# AI_Text Detection

This is the official code for AI_Text Detection in TextGenAdvTrack, the practice session of the course Artificial Intelligence Security, Attacks and Defenses (UCAS, spring 2025).

## ⚡ How to start quickly

1. Clone the repository
```
git clone https://github.com/UCASAISecAdv/TextGenAdvTrack-2025Spring.git
cd TextGenAdvTrack-2025Spring/AI_Text Detection
```

2. Prepare the environment
```
conda create -n AISAD python=3.8
conda activate AISAD
pip install -r requirements.txt
```

3. Download the dataset \
Please acquire the download link from our Wechat. 
- **Training Dataset**: You may use any dataset for training  **EXCEPT M4 AND HC3**. \
  Using M4 and HC3 for training is strictly **PROHIBITED**. \
  You should declare any additional data sources in your final report.
- **Validation Set**: UCAS_AISAD_TEXT-val. Selected samples from M4 and HC3 datasets with labels provided.
- **Test Set 1**: UCAS_AISAD_TEXT-test_1. Created by applying evasion attacks (such as paraphrasing, synonym replacement, etc.) to the validation set without labels provided.
- **Test Set 2**: UCAS_AISAD_TEXT-test_2. Additional samples collected from the evasion track of this assessment and will be released at the last week of the practice.

Example data format:
```csv
prompt,human_text,machine_text
"Explain quantum computing","Quantum computing uses quantum bits or qubits...","Quantum computing is a type of computation that harnesses..."
"Describe climate change","Climate change refers to long-term shifts...","Climate change is the long-term alteration in Earth's..."
"解释量子计算的原理","量子计算利用量子比特或称量子位作为基本计算单元...","量子计算是一种利用量子力学原理进行信息处理的技术..."
"描述全球气候变化","全球气候变化是指地球气候系统的长期变化...","全球气候变化是指地球气候系统的统计特性随时间变化..."
"Объяснение принципов квантовых вычислений","Квантовые вычисления используют квантовые биты, или кубиты, в качестве основных вычислительных единиц...","Квантовые вычисления - это технология, использующая принципы квантовой механики для обработки информации..."
"Описание глобального изменения климата","Глобальное изменение климата относится к долгосрочным изменениям климатической системы Земли...","Глобальное изменение климата относится к изменению статистических характеристик климатической системы Земли во времени..."
...
```

4. Run model inference
```
python inference.py \
    --your-team-name $YOUR_TEAM_NAME \
    --data-path $YOUR_DATASET_PATH/test1 \
    --model-type $MODEL \
    --result-path $YOUR_SAVE_PATH/
```

5. Evaluate model performance \
We evaluate a model according to AUC. Please refer to the corresponding file.
```
python evaluate.py \
    --submit-path ${YOUR_SAVE_PATH}/${YOUR_TEAM_NAME} \
```

## 📊 File Format Specifications

### Input Dataset Format
CSV file with columns: `prompt`, `human_text`, `machine_text`

### Output Format
Excel file (`<your-team-name>.xlsx`) with two sheets:
- `predictions` sheet containing:
  - `prompt`: Original prompt text
  - `human_text_prediction`: Probability of human authorship (higher = more likely human)
  - `machine_text_prediction`: Probability of human authorship (higher = more likely human)
- `time` sheet containing:
  - `Data Volume`: Number of processed examples
  - `Time`: Total processing time in seconds


## 📈 Evaluation Metrics
Models are evaluated based on:
- Machine AUC: AUC score for detecting machine-generated text
- Human AUC: AUC score for identifying human-written text
- Combined AUC: Average of Machine AUC and Human AUC (ranking metric)
- Avg Time (s): Processing time per example


## ⚠️ Caution
1. Do not modify the column names in the output files
2. Higher probability scores should indicate higher likelihood of human authorship
3. The leaderboard ranks teams by Combined AUC in descending order


## 🔧 Available Models
- `argugpt`: SJTU-CL/RoBERTa-large-ArguGPT-sent
- `openai`: openai-community/roberta-base-openai-detector
- `radar`: SJTU-CL/RoBERTa-large-ArguGPT
- tips: you can load models from 'huggingface' or 'local'.

## Acknowledgements
- This code is based on [LLMDA](https://github.com/ShushantaTUD/LLMDA).
