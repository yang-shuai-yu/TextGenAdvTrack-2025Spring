import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class FreezeLMWithClassifier(nn.Module):
    def __init__(self, lm_model, hidden_size, num_labels=2, pooling='cls'):
        super().__init__()
        self.lm = lm_model
        self.pooling = pooling
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Freeze所有参数
        for param in self.lm.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, **kwargs):
        with torch.no_grad():
            outputs = self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            last_hidden = outputs.hidden_states[-1]

        # 池化方式
        if self.pooling == 'cls':
            pooled = last_hidden[:, 0, :]  # (batch, hidden)
        elif self.pooling == 'mean':
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).expand(last_hidden.size())
                sum_hidden = (last_hidden * mask).sum(1)
                lens = mask.sum(1)
                pooled = sum_hidden / lens
            else:
                pooled = last_hidden.mean(dim=1)
        else:
            raise ValueError("Unknown pooling method")

        # 关键改动：保证pooled和classifier在同一设备
        pooled = pooled.to(self.classifier.weight.device)

        logits = self.classifier(pooled)
        return logits

# model.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class LMModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        # 加载预训练语言模型
        self.lm = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

    def forward(self, input_ids, attention_mask=None, **kwargs):
        with torch.no_grad():
            # 获取模型输出，包括隐藏层状态
            outputs = self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            # 使用最后一层隐藏状态
            last_hidden = outputs.hidden_states[-1]
        return last_hidden


class Classifier(nn.Module):
    def __init__(self, hidden_size, num_labels=2, pooling='cls'):
        super().__init__()
        self.pooling = pooling
        # 分类头
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, last_hidden, attention_mask=None):
        # 池化方式
        if self.pooling == 'cls':
            pooled = last_hidden[:, 0, :]  # (batch, hidden)
        elif self.pooling == 'mean':
            if attention_mask is not None:
                # 按mask加权均值
                # mask = attention_mask.unsqueeze(-1).expand(last_hidden.size())
                mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden.dtype)
                sum_hidden = (last_hidden * mask).sum(1)
                lens = mask.sum(1)
                pooled = sum_hidden / lens
            else:
                pooled = last_hidden.mean(dim=1)
        else:
            raise ValueError("Unknown pooling method")

        # 分类
        pooled = pooled.float()    # half -> float
        logits = self.classifier(pooled)
        return logits

class PolyLMClassifier(nn.Module):
    def __init__(self, model_path, hidden_size, pooling='mean'):
        """
        Args:
            model_path (str): 本地模型路径
            hidden_size (int): PolyLM 的 hidden size
            pooling (str): 'cls' 或 'mean'
        """
        super().__init__()
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, legacy=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # 加载 PolyLM 模型
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.lm.resize_token_embeddings(len(self.tokenizer))
        
        self.pooling = pooling
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 冻结 PolyLM 参数
        for param in self.lm.parameters():
            param.requires_grad = False

    def forward(self, texts):
        """
        Args:
            texts (List[str]): 已经拼好的文本序列列表
        Returns:
            probs: 概率输出 (batch_size, )
        """
        # Tokenize
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        device = next(self.parameters()).device
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)

        # Debug: 检查输入文本和 attention_mask
        # if any([t.strip() == "" for t in texts]):
        #     print("WARNING: 有空文本输入！texts:", texts)
        # # 打印每个样本的有效长度
        # print("DEBUG: attention_mask.sum(1):", attention_mask.sum(1).tolist())

        with torch.no_grad():
            outputs = self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            last_hidden = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden)

        # 池化
        if self.pooling == 'cls':
            pooled = last_hidden[:, 0, :]  # 取首 token
        elif self.pooling == 'mean':
            mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden.dtype)
            sum_hidden = (last_hidden * mask).sum(1)
            lens = mask.sum(1).clamp(min=1e-6)  # 防止除0
            # Debug: 检查 lens 是否有0
            if (lens == 1e-6).any():
                print("WARNING: 有样本 attention_mask 全0！对应文本:", [t for i, t in enumerate(texts) if lens[i] == 1e-6])
            pooled = sum_hidden / lens
        else:
            raise ValueError("Unknown pooling method")

        pooled = pooled.float()  # 转回 float32
        pooled = torch.clamp(pooled, -1e2, 1e2)
        logits = self.classifier(pooled).squeeze(-1)  # (batch_size, )

        probs = torch.sigmoid(logits)  # 转为概率

        # Debug: 检查概率是否在[0,1]内和有无nan
        if torch.isnan(probs).any():
            print("ERROR: 出现 nan！probs:", probs)
            print("相关 pooled:", pooled)
            print("相关 lens:", lens)
            print("相关输入文本:", texts)
        if (probs < 0).any() or (probs > 1).any():
            print("ERROR: 概率超出[0,1]！probs:", probs)
        
        return probs