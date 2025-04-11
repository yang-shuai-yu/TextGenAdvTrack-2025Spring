"""
Utility functions for LLM detection models
"""

def adjust_prediction_score(pred_label, pred_score, model_type):
    """
    Adjusts the prediction score to a standardized format where higher values
    indicate higher likelihood of human-written text.
    
    Args:
        pred_label (str): Prediction label from the model
        pred_score (float): Prediction score/probability
        model_type (str): Model type ("argugpt", "radar", "openai")
        
    Returns:
        float: Adjusted score where higher values indicate more likely human-written text
    """
    if model_type == "argugpt":
        # ArguGPT: LABEL_1=machine(0), LABEL_0=human(1)
        if pred_label == "LABEL_0":  
            return pred_score
        else:  
            return 1 - pred_score
    elif model_type == "radar":
        # RADAR: LABEL_1=human(1), LABEL_0=machine(0)
        if pred_label == "LABEL_1": 
            return pred_score
        else:  
            return 1 - pred_score
    elif model_type == "openai":
        # OpenAI: Real=human(1), Fake=machine(0)
        if pred_label == "Real":  
            return pred_score
        else:  
            return 1 - pred_score
    else:
        # Default handling, return original score
        return pred_score
