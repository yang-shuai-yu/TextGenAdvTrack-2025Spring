import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import pipeline
from utils.model_utils import adjust_prediction_score

def get_opts():
    arg = argparse.ArgumentParser()
    arg.add_argument(
        "--your-team-name",
        type=str,
    )
    arg.add_argument(
        "--data-path",
        type=str,
        help="Path to the CSV dataset"
    )
    arg.add_argument(
        "--model-type",
        type=str,
        choices=["argugpt", "openai", "radar"],
        help="Type of model to use"
    )
    arg.add_argument(
        "--result-path",
        type=str,
    )
    opts = arg.parse_args()
    return opts

def get_dataset(opts):
    print(f"Loading dataset from {opts.data_path}...")
    data = pd.read_csv(opts.data_path)

    # Your format: prompt, human_text, machine_text
    dataset = data[['prompt', 'human_text', 'machine_text']].dropna().copy()
    print(f"Prepared dataset with {len(dataset)} prompts")
    
    return dataset

def get_model(opts):
    print(f"Loading {opts.model_type} detector model...")
    
    if opts.model_type == "argugpt":
        model = pipeline("text-classification", model="SJTU-CL/RoBERTa-large-ArguGPT-sent", 
                         max_length=512, truncation=True)
    elif opts.model_type == "openai":
        model = pipeline("text-classification", model="roberta-base-openai-detector", 
                         max_length=512, truncation=True)
    elif opts.model_type == "radar":
        model = pipeline("text-classification", model="TrustSafeAI/RADAR-Vicuna-7B")
    
    print("Model loaded successfully")
    return model

def run_inference(model, dataset, model_type):
    print("Starting prediction process...")
    prompts = dataset['prompt'].tolist()
    machine_texts = dataset['machine_text'].tolist()
    human_texts = dataset['human_text'].tolist()
    
    machine_predictions = []
    human_predictions = []
    
    start_time = pd.Timestamp.now()
    
    # Process machine texts
    print("Processing machine-generated texts...")
    for text in tqdm(machine_texts, desc="Machine text predictions"):
        try:
            prediction = model(text)
            
            # Get label and score
            pred_label = prediction[0]['label']
            pred_score = prediction[0]['score']
            
            # Adjust score - higher values indicate more likely human-written text
            final_score = adjust_prediction_score(pred_label, pred_score, model_type)
                
            machine_predictions.append(final_score)
        except Exception as e:
            print(f"Error processing machine text: {str(e)[:100]}...")
            machine_predictions.append(None)
    
    # Process human texts
    print("Processing human-written texts...")
    for text in tqdm(human_texts, desc="Human text predictions"):
        try:
            prediction = model(text)
            
            # Get label and score
            pred_label = prediction[0]['label']
            pred_score = prediction[0]['score']
            
            # Adjust score - higher values indicate more likely human-written text
            final_score = adjust_prediction_score(pred_label, pred_score, model_type)
                
            human_predictions.append(final_score)
        except Exception as e:
            print(f"Error processing human text: {str(e)[:100]}...")
            human_predictions.append(None)
    
    end_time = pd.Timestamp.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # Create combined results in the requested format
    results_data = {
        'prompt': prompts,
        'human_text_prediction': human_predictions,
        'machine_text_prediction': machine_predictions
    }
    
    # Create results dictionary
    results = {
        "predictions_data": results_data,
        "time": processing_time
    }
    
    print(f"Predictions completed in {processing_time:.2f} seconds")
    return results

if __name__ == "__main__":
    opts = get_opts()
    dataset = get_dataset(opts)
    model = get_model(opts)
    results = run_inference(model, dataset, opts.model_type)
    
    # Save results
    os.makedirs(opts.result_path, exist_ok=True)
    writer = pd.ExcelWriter(os.path.join(opts.result_path, opts.your_team_name + ".xlsx"), engine='openpyxl')
    
    # Create prediction dataframe with the required 3 columns
    prediction_frame = pd.DataFrame(
        data = results["predictions_data"]
    )
    
    # Filter out rows with None values
    prediction_frame = prediction_frame.dropna()
    
    time_frame = pd.DataFrame(
        data = {
            "Data Volume": [len(prediction_frame)],
            "Time": [results["time"]],
        }
    )
    
    prediction_frame.to_excel(writer, sheet_name="predictions", index=False)
    time_frame.to_excel(writer, sheet_name="time", index=False)
    writer.close()
    
    print(f"Results saved to {os.path.join(opts.result_path, opts.your_team_name + '.xlsx')}")
