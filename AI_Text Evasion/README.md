# AI_Text Evasion

This is the official code for AI_Text Evasion in TextGenAdvTrack, the practice session of the course Artificial Intelligence Security, Attacks and Defenses (UCAS, spring 2025).

## ⚡ How to start

1. Download the dataset

Please acquire the download link of UCAS_AISAD_TEXT-val dataset from our Wechat. UCAS_AISAD_TEXT-val dataset contains 6,000 prompt, machine-generated and human-written texts sampled from M4 and HC3 dataset.


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


2. Prepare your evasion algorithm

Develop strategies to modify AI-generated text(column 'machine_text') to evade detection systems. You may use various techniques including:
- Paraphrasing
- Synonym Replacement
- Misspelling
- Article Deletion
- Space Infiltration Strategy
- Homoglyph Attack
- Other techniques of your choice
- ...

3. Submit your results

Generate the evasion results following the format described below, and submit a file named `YOUR_TEAM_NAME_test_1.csv` to the TA.

## Format of results

Your submission should be a CSV file with the following format:

```csv
prompt,machine_text,human_text
prompt_1,machine_text_after_evasion_1,human_text_1
prompt_2,machine_text_after_evasion_2,human_text_2
...
```

**Important**: Only modify the machine_text column. The prompt and human_text columns must remain exactly the same as in the original dataset.

## Naming Convention
Your submission must be named `YOUR_TEAM_NAME_test_1.csv`, where YOUR_TEAM_NAME is your team's identifier.

## Evaluation Criteria
We evaluate the evasion effectiveness by calculating **1-AUC** from there different models. The overall rating score is their average. For fairness, we do not provide the evaluation script.

## ⚠️ Caution
1. **DO NOT modify anything other than the machine_text column**. The prompt and human_text columns must remain unchanged.
2. All submissions must maintain the exact same number of samples as in the original dataset.
3. Modified texts should preserve the original meaning to the extent possible.
4. Submissions not following the required format will be rejected and may affect your evaluation results.

