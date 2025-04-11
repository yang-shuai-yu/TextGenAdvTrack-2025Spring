import os
import argparse

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

def eval_func(gt_name="ground_truth", path=""):
    """
    Evaluate LLMDA detector results from multiple methods
    
    Args:
        gt_name (str): Name of the ground truth file (without extension)
        path (str): Path to directory containing result files
        
    Returns:
        dict: Dictionary with team/method names as keys and their metrics as values
    """
    ret = {}

    # Get all files in the directory
    teams = os.listdir(path)
    
    # Filter out ground truth and leaderboard files
    if gt_name + ".xlsx" in teams:
        teams.remove(gt_name + ".xlsx")
    if "LeaderBoard.xlsx" in teams:
        teams.remove("LeaderBoard.xlsx")

    # Read ground truth file
    gts = pd.read_excel(
        os.path.join(path, gt_name + ".xlsx"),
        sheet_name="labels"
    )
    
    # Extract machine and human ground truth
    human_gt = gts["human_label"].values
    machine_gt = gts["machine_label"].values

    print(f"Found {len(teams)} team/method submissions to evaluate")
    
    for team in teams:
        try:
            # Read predictions sheet
            data = pd.read_excel(
                os.path.join(path, team),
                sheet_name="predictions",
            )
            
            # Extract machine and human predictions
            human_pred = data["human_text_prediction"].values
            machine_pred = data["machine_text_prediction"].values
            
            # Calculate AUC for machine text detection
            machine_auc = roc_auc_score(machine_gt, machine_pred)
            
            # Calculate AUC for human text detection
            human_auc = roc_auc_score(human_gt, human_pred)
            
            # Calculate combined AUC
            combined_auc = (machine_auc + human_auc) / 2
            
            # Read time information
            time_data = pd.read_excel(
                os.path.join(path, team),
                sheet_name="time",
            )
            
            # Calculate average processing time per sample
            mean_time = time_data["Time"][0] / time_data["Data Volume"][0]
            
            # Store results
            ret[team.split(".")[0]] = {
                "machine_auc": machine_auc,
                "human_auc": human_auc, 
                "combined_auc": combined_auc,
                "mean_time": mean_time
            }
            
            print(f"Evaluated {team}: Machine AUC={machine_auc:.4f}, Human AUC={human_auc:.4f}, Combined={combined_auc:.4f}")
            
        except Exception as e:
            print(f"Error processing {team}: {str(e)}")
            continue

    return ret

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument(
        "--submit-path",
        type=str,
        help="Path to directory containing submission files"
    )
    arg.add_argument(
        "--gt-name",
        type=str,
        default="ground_truth",
        help="Name of ground truth file (without extension)"
    )
    opts = arg.parse_args()
    
    results = eval_func(gt_name=opts.gt_name, path=opts.submit_path)

    # Create leaderboard Excel file
    writer = pd.ExcelWriter(os.path.join(opts.submit_path, "LeaderBoard.xlsx"), engine="openpyxl")
    
    leaderboard_data = {
        "Team/Method": results.keys(),
        "Machine AUC": [res["machine_auc"] for res in results.values()],
        "Human AUC": [res["human_auc"] for res in results.values()],
        "Combined AUC": [res["combined_auc"] for res in results.values()],
        "Avg Time (s)": [res["mean_time"] for res in results.values()]
    }
    
    # Sort by combined AUC (descending)
    leaderboard_df = pd.DataFrame(data=leaderboard_data)
    leaderboard_df = leaderboard_df.sort_values(by="Combined AUC", ascending=False)
    
    leaderboard_df.to_excel(writer, index=False)
    writer.close()
    
    print(f"Leaderboard saved to {os.path.join(opts.submit_path, 'LeaderBoard.xlsx')}")
