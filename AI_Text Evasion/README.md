# AI_Text Evasion

This is the official code for AI_Text Evasion in TextGenAdvTrack, the practice session of the course Artificial Intelligence Security, Attacks and Defenses (UCAS, spring 2025).

## ⚡ How to start

1. Download the dataset

Please acquire the download link of UCAS_AISAD_TEXT-val dataset from our Wechat. UCAS_AISAD_TEXT-val dataset contains 6,000 prompt, machine-generated and human-written texts sampled from M4 and HC3 dataset.

- **Training Dataset**: You may use any dataset for training  **EXCEPT M4 AND HC3**. Using M4 and HC3 for training is strictly **PROHIBITED**. \
   You should declare any additional data sources in your final report.
 - **Validation Set**: UCAS_AISAD_TEXT-val. 6,000 samples selected from M4 and HC3 datasets *with labels provided*.
 - **Test Set 1**: UCAS_AISAD_TEXT-test_1. 6,000 prompt, machine-generated and human-written texts *without labels provided*.
 - **Test Set 2**: UCAS_AISAD_TEXT-test_2. Additional samples collected from the evasion track of this assessment and will be released at the last week of the practice.

'UCAS_AISAD_TEXT-val','UCAS_AISAD_TEXT-test_1' and 'UCAS_AISAD_TEXT-test_2' : \
CSV file with columns: `prompt`, `text`, `label`(optional)
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


2. Prepare your evasion algorithm

Develop strategies to modify the machine_text which label is `0` in Validation Set: `UCAS_AISAD_TEXT-val` to evade detection systems. You may use various techniques including:
- Paraphrasing
- Synonym Replacement
- Misspelling
- Other techniques of your choice
- ...

3. Submit your results

Generate the evasion results following the format described below, and submit a file named `YOUR_TEAM_NAME_test_1.csv` to the TA.


4. (Optional) To evaluate the eavsion effectiveness, run `evaluate.py` following the instructions in `/detection/README.md`


## Format of results

Your submission should be a CSV file with the following format:

CSV file with columns: `prompt`, `text`
```csv
prompt,text
"Explain quantum computing","Quantum computing uses quantum bits or qubits..."
"Describe climate change","Climate change refers to long-term shifts..."
"解释量子计算的原理","量子计算利用量子比特或称量子位作为基本计算单元..."
"描述全球气候变化","全球气候变化是指地球气候系统的长期变化..."
"Объяснение принципов квантовых вычислений","Квантовые вычисления используют квантовые биты, или кубиты, в качестве основных вычислительных единиц..."
"Описание глобального изменения климата","Глобальное изменение климата относится к долгосрочным изменениям климатической системы Земли..."
...
```
**Important**: 
- Only modify the machine_text which label is `0`. The prompt and human_text which label is `1` must remain exactly the same as in the original dataset.
- Please confirme that the csv file encoding method is **utf-8**

## Naming Convention
Your submission must be named `your-team-name_test_2.csv`, where YOUR_TEAM_NAME is your team's identifier.

## Evaluation Criteria
We evaluate the evasion effectiveness by calculating **1-AUC** from there different models. The overall rating score is their average. For fairness, we do not provide the evaluation script.



## ⚠️ Caution
1. Do not modify the column names in the output files
2. Do not change the order of the data or it will affect the evaluation results
3. Modified texts should preserve the original meaning to the extent possible.
4. Submissions not following the required format will be rejected and may affect your evaluation results.

