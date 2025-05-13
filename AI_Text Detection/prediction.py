import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from utils.model_utils import adjust_prediction_score
from utils.my_models import PolyLMClassifier  # 你的新模型类
import torch
import torch.nn.functional as F

def get_opts():
    arg = argparse.ArgumentParser()
    arg.add_argument(
        "--your-team-name",
        type=str,
        default="生成文本队",
        help="Your team name"
    )
    arg.add_argument(
        "--data_path",
        type=str,
        help="Path to the CSV dataset",
        default="./data/UCAS_AISAD_TEXT-test2.csv"
    )
    arg.add_argument(
        "--hidden_size",
        help="Hidden size of model",
        default=2048
    )
    arg.add_argument(
        "--model_type",
        type=str,
        choices=["polylm", "RADAR"],  # ✅ 添加 RADAR
        help="Type of model to use",
        default="polylm"
    )
    arg.add_argument(
        "--model_path",
        help="Your model path, change with model-type",
        default="./models/polylm-1.7b"  # RADAR-Vicuna-7b
    )
    arg.add_argument(
        "--checkpoint_path",
        help="Checkpoint path",
        default='./checkpoints/best_polylm_classifier_m4_fusion.pt'
    )
    arg.add_argument(
        "--result-path",
        type=str,
        help="Path to save the results",
        default="./results"
    )
    arg.add_argument(
        "--device",
        default="cuda"
    )
    opts = arg.parse_args()
    return opts

def get_dataset(opts):
    print(f"Loading dataset from {opts.data_path}...")
    data = pd.read_csv(opts.data_path)
    dataset = data[['prompt', 'text']].dropna().copy()
    print(f"Prepared dataset with {len(dataset)} prompts")
    return dataset

def get_model(opts):
    print(f"Loading {opts.model_type} detector model...")
    device = opts.device

    if opts.model_type == "polylm":
        hidden_size = 2048
        model = PolyLMClassifier(
            model_path=opts.model_path,
            hidden_size=hidden_size,
            pooling='mean'
        ).to(device)
        model.load_state_dict(torch.load(opts.checkpoint_path))
        print("PolyLMClassifier model loaded successfully")
        return model, None  # tokenizer 是 None

    elif opts.model_type == "RADAR":
        detector = AutoModelForSequenceClassification.from_pretrained(opts.model_path)
        tokenizer = AutoTokenizer.from_pretrained(opts.model_path)
        detector.eval()
        detector.to(device)
        print("RADAR model loaded successfully")
        return detector, tokenizer

    else:
        raise ValueError(f"Unknown model type: {opts.model_type}")

def run_prediction(model, tokenizer, dataset, opts):
    print("Starting prediction process...")
    prompts = dataset['prompt'].tolist()
    texts = dataset['text'].tolist()
    text_predictions = []
    device = opts.device
    model_type = opts.model_type

    start_time = pd.Timestamp.now()

    print("Processing texts...")
    for text in tqdm(texts, desc="Text predictions"):
        try:
            if model_type == "polylm":
                prediction = model(text)
                pred_score = prediction[0].detach().cpu()
                pred_label = (pred_score > 0.5).long()
                pred_score = pred_score.item()
                final_score = adjust_prediction_score(pred_label, pred_score, model_type)

            elif model_type == "RADAR":
                with torch.no_grad():
                    inputs = tokenizer(
                        text,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                    output_probs = F.log_softmax(outputs.logits, -1)[:, 0].exp().tolist()
                    prob_ai = output_probs[0]  # AI 概率
                    # RADAR 是 AI 概率，我们要按人类概率调整（1 - prob_ai）
                    pred_label = int((1-prob_ai)>0.5)
                    final_score = 1 - prob_ai

            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            text_predictions.append(final_score)

        except Exception as e:
            print(f"Error processing text: {str(e)[:100]}...")
            text_predictions.append(None)

    end_time = pd.Timestamp.now()
    processing_time = (end_time - start_time).total_seconds()

    results_data = {
        'prompt': prompts,
        'text_prediction': text_predictions
    }
    results = {
        "predictions_data": results_data,
        "time": processing_time
    }
    print(f"Predictions completed in {processing_time:.2f} seconds")
    return results

if __name__ == "__main__":
    opts = get_opts()
    dataset = get_dataset(opts)
    model, tokenizer = get_model(opts)
    results = run_prediction(model, tokenizer, dataset, opts)

    # Save results
    os.makedirs(opts.result_path, exist_ok=True)
    writer = pd.ExcelWriter(os.path.join(opts.result_path, opts.your_team_name + ".xlsx"), engine='openpyxl')
    prediction_frame = pd.DataFrame(data=results["predictions_data"])
    prediction_frame = prediction_frame.dropna()
    time_frame = pd.DataFrame(data={
        "Data Volume": [len(prediction_frame)],
        "Time": [results["time"]],
    })
    prediction_frame.to_excel(writer, sheet_name="predictions", index=False)
    time_frame.to_excel(writer, sheet_name="time", index=False)
    writer.close()
    print(f"Results saved to {os.path.join(opts.result_path, opts.your_team_name + '2' + '.xlsx')}")


# import os
# import argparse
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# from transformers import pipeline
# from utils.model_utils import adjust_prediction_score
# from utils.my_models import PolyLMClassifier  # 你的新模型类
# import torch

# def get_opts():
#     arg = argparse.ArgumentParser()
#     arg.add_argument(
#         "--your-team-name",
#         type=str,
#         default="生成文本队",
#         help="Your team name"
#     )
#     arg.add_argument(
#         "--data_path",
#         type=str,
#         help="Path to the CSV dataset",
#         default="./data/UCAS_AISAD_TEXT-val.csv"
#     )
#     arg.add_argument(
#         "--hidden_size",
#         help="Hidden size of model",
#         default=2048
#     )
#     arg.add_argument(
#         "--model-type",
#         type=str,
#         choices=[ "your model"],
#         help="Type of model to use",
#         default="your model"
#     )
#     arg.add_argument(
#         "--model_path",
#         help="Your model path",
#         default="./models/polylm-1.7b"
#     )
#     arg.add_argument(
#         "--checkpoint_path",
#         help="Checkpoint path",
#         default='./checkpoints/best_polylm_classifier_m4_fusion.pt'
#     )
#     arg.add_argument(
#         "--result-path",
#         type=str,
#         help="Path to save the results",
#         default="./results"
#     )
#     arg.add_argument(
#         "--device",
#         default="cuda"
#     )
#     opts = arg.parse_args()
#     return opts

# def get_dataset(opts):
#     print(f"Loading dataset from {opts.data_path}...")
#     data = pd.read_csv(opts.data_path)

#     # New format: prompt, text
#     dataset = data[['prompt', 'text']].dropna().copy()
#     print(f"Prepared dataset with {len(dataset)} prompts")
    
#     return dataset

# def get_model(opts):
#     print(f"Loading {opts.model_type} detector model...")
    
#     '''
#     You should load your model here!
#     '''
#     hidden_size = 2048  # 改成你的 PolyLM 的 hidden size
#     model = PolyLMClassifier(
#         model_path=opts.model_path,
#         hidden_size=hidden_size,
#         pooling='mean'  # 推荐 mean 池化
#     ).to(opts.device)
#     model.load_state_dict(torch.load(opts.checkpoint_path))
#     print("Model loaded successfully")
#     return model

# def run_prediction(model, dataset, model_type):
#     print("Starting prediction process...")
#     prompts = dataset['prompt'].tolist()
#     texts = dataset['text'].tolist()
    
#     text_predictions = []
    
#     start_time = pd.Timestamp.now()
    
#     # Process texts
#     print("Processing texts...")
#     for text in tqdm(texts, desc="Text predictions"):
#         try:
#             prediction = model(text)
#             # print(prediction)
#             # Get label and score
#             # pred_label = prediction[0]['label']
#             # pred_score = prediction[0]['score']
#             pred_score = prediction[0].detach().cpu()
#             pred_label = (pred_score > 0.5).long()
#             pred_score = pred_score.item()
            
#             # Adjust score - higher values indicate more likely human-written text
#             final_score = adjust_prediction_score(pred_label, pred_score, model_type)
                
#             text_predictions.append(final_score)
#         except Exception as e:
#             print(f"Error processing text: {str(e)[:100]}...")
#             text_predictions.append(None)
    
#     end_time = pd.Timestamp.now()
#     processing_time = (end_time - start_time).total_seconds()
    
#     # Create results in the requested format
#     results_data = {
#         'prompt': prompts,
#         'text_prediction': text_predictions
#     }
    
#     # Create results dictionary
#     results = {
#         "predictions_data": results_data,
#         "time": processing_time
#     }
    
#     print(f"Predictions completed in {processing_time:.2f} seconds")
#     return results

# if __name__ == "__main__":
#     opts = get_opts()
#     dataset = get_dataset(opts)
#     model = get_model(opts)
#     results = run_prediction(model, dataset, opts.model_type)
    
#     # Save results
#     os.makedirs(opts.result_path, exist_ok=True)
#     writer = pd.ExcelWriter(os.path.join(opts.result_path, opts.your_team_name + ".xlsx"), engine='openpyxl')
    
#     # Create prediction dataframe with the required columns
#     prediction_frame = pd.DataFrame(
#         data = results["predictions_data"]
#     )
    
#     # Filter out rows with None values
#     prediction_frame = prediction_frame.dropna()
    
#     time_frame = pd.DataFrame(
#         data = {
#             "Data Volume": [len(prediction_frame)],
#             "Time": [results["time"]],
#         }
#     )
    
#     prediction_frame.to_excel(writer, sheet_name="predictions", index=False)
#     time_frame.to_excel(writer, sheet_name="time", index=False)
#     writer.close()
    
#     print(f"Results saved to {os.path.join(opts.result_path, opts.your_team_name + '.xlsx')}")
