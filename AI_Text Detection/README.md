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
- **Validation Set**: UCAS_AISAD_TEXT-val. 6,000 samples selected from M4 and HC3 datasets with labels provided.
- **Test Set 1**: UCAS_AISAD_TEXT-test_1. Created by applying evasion attacks (such as paraphrasing, synonym replacement, etc.) to the validation set without labels provided.
- **Test Set 2**: UCAS_AISAD_TEXT-test_2. Additional samples collected from the evasion track of this assessment and will be released at the last week of the practice.


4. Run model prediction
```
python prediction.py \
    --your-team-name $YOUR_TEAM_NAME \
    --data-path $YOUR_DATASET_PATH/test1 \
    --model-type $MODEL \
    --result-path $YOUR_SAVE_PATH/
```

**You should load your detectiion model!**

5. Evaluate model performance \
We evaluate a model according to AUC. Please refer to the corresponding file.
```
python evaluate.py \
    --submit-path ${YOUR_SAVE_PATH}/${YOUR_TEAM_NAME} \
    --gt-name $PATH_TO_GROUND_TRUTH_WITHOUT_EXTENSION
```

## 📊 File Format Specifications
### Input Dataset Format
'UCAS_AISAD_TEXT-val','UCAS_AISAD_TEXT-test_1' and 'UCAS_AISAD_TEXT-test_2' : \
CSV file with columns: `prompt`, `text`
```csv
prompt,text,label
"Explain quantum computing","Quantum computing uses quantum bits or qubits...",0
"Describe climate change","Climate change refers to long-term shifts...",1
"解释量子计算的原理","量子计算利用量子比特或称量子位作为基本计算单元...",1
"描述全球气候变化","全球气候变化是指地球气候系统的长期变化...",0
"Объяснение принципов квантовых вычислений","Квантовые вычисления используют квантовые биты, или кубиты, в качестве основных вычислительных единиц...",1
"Описание глобального изменения климата","Глобальное изменение климата относится к долгосрочным изменениям климатической системы Земли...",0
...
```
'0' stands for 'machine_text', '1' stands for 'human_text'.

### Ground Truth Format
'gt' : \
CSV file with columns:  `label`:
```csv
label
0
1
1
1
...
```

### Output Format
Excel file (`<your-team-name>.xlsx`) with two sheets:
- `predictions` sheet containing :
  - `prompt`: Original prompt text
  - `text_prediction`: Probability of human authorship (higher = more likely human)
```csv
prompt,text_prediction
"Explain quantum computing",0.95
"Describe climate change",0.68
...
```

- `time` sheet containing:
  - `Data Volume`: Number of processed examples
  - `Time`: Total processing time in seconds
```csv
Data Volume,Time
"6000",53.21
```

## 📈 Evaluation Metrics
Models are evaluated based on:
- AUC: Area Under the ROC Curve for the unified dataset (combines both human and machine text detection)
- Acc.(%): Accuracy percentage (correctly classified samples / total samples)
- F1: F1 score measuring the balance between precision and recall
- Final_Score = 0.6 * AUC + 0.3 * Acc.(%)  + 0.1 * F1) / 100

The leaderboard ranks teams by Final_Score in descending order. Higher values indicate better performance.


## ⚠️ Caution
1. Do not modify the column names in the output files
2. Do not change the order of the data or it will affect the evaluation results
3. Higher probability scores should indicate higher likelihood of human_text. Please check the 'utils/model_utils.py' to get the correct result


## 🔧 Available Models
- `argugpt`: SJTU-CL/RoBERTa-large-ArguGPT-sent
- tips: you can load models from 'huggingface' or 'local'.

