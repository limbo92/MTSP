#%%
import os, sys, json, argparse, pickle, time, copy, math, re, random, shutil, glob
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from data_loader import load_text_data, load_label_data

# Configure OpenAI client, pointing to local vLLM server
client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8001/v1"
)


def try_load_json(path: Path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(e)
        return None

def try_load_pickle(path: Path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(e)
        return None

def try_load_torch_pt(path: Path):
    try:
        return torch.load(path)
    except Exception as e:
        print(e)
        return None

# Task instruction for the model
TASK_INSTRUCTION = """
# Task:
Based on the given interview conversation excerpts with the patient, assign a PHQ-8 score (0-3) for each of the 8 factors below.

If a factor is not explicitly mentioned in the interview, you must still estimate a reasonable score by inferring from the overall patient situation and context.

# PHQ-8 items and scoring rules:
Over the last 2 weeks, how often has the patient been bothered by any of the following problems?

# Items:
1. NoInterest – Little interest or pleasure in doing things.
2. Depressed – Feeling down, depressed, or hopeless.
3. Sleep – Trouble falling or staying asleep, or sleeping too much.
4. Tired – Feeling tired or having little energy.
5. Appetite – Poor appetite or overeating.
6. Failure – Feeling bad about yourself, or that you are a failure, or have let yourself or your family down.
7. Concentrating – Trouble concentrating on things, such as reading the newspaper or watching television.
8. Moving – Moving or speaking so slowly that other people could have noticed, or the opposite: being so fidgety or restless that you have been moving around a lot more than usual.

# Each PHQ-8 item is rated on a 4-point scale:
0 = Not at all
1 = Several days
2 = More than half the days
3 = Nearly every day

# Response Format:
After thinking, you must output all 8 factors in the following format without any additional text or explanation. The score for each factor should be an integer between 0 and 3, inclusive.
- NoInterest: 0
- Depressed: 0
- Sleep: 0
- Tired: 0
- Appetite: 0
- Failure: 0
- Concentrating: 0
- Moving: 0
"""

def extract_scores(text: str):
    """
    Extract PHQ-8's 8 factor scores from model output text.
    
    Args:
        text: Model output text, format should be:
            - NoInterest: 0
            - Depressed: 1
            ...
    
    Returns:
        list: List of 8 scores, returns [1.5] * 8 if parsing fails
    """
    try:
        lines = text.split('\n')
        response_dict = {}
        for line in lines:
            if ':' not in line:
                continue
            item, score = line.split(':', 1)  # Only split at the first colon
            item = item.strip().lstrip('-').strip()  # Remove possible leading dash
            score = score.strip()
            
            # Try to convert to float or int
            try:
                score_value = float(score)
                response_dict[item] = score_value
            except ValueError:
                continue
        
        # Extract scores in PHQ-8 order
        phq8_items = ['NoInterest', 'Depressed', 'Sleep', 'Tired', 
                      'Appetite', 'Failure', 'Concentrating', 'Moving']
        
        scores = []
        for item in phq8_items:
            if item in response_dict:
                scores.append(response_dict[item])
            else:
                scores.append(1.5)  # Default value
        
        return scores if len(scores) == 8 else [1.5] * 8
    except Exception as e:
        print(f'Error extracting scores: {e}')
        return [1.5] * 8

def construct_example(data):
    """
    Construct a few-shot example.
    
    Args:
        data: Data dictionary containing 'transcript' and 'scores'
    
    Returns:
        str: Formatted example text
    """
    transcript = data['transcript']
    scores = data['scores']
    example = f"""
Transcript: {transcript}

Scores:
- NoInterest: {scores[0]}
- Depressed: {scores[1]}
- Sleep: {scores[2]}
- Tired: {scores[3]}
- Appetite: {scores[4]}
- Failure: {scores[5]}
- Concentrating: {scores[6]}
- Moving: {scores[7]}
""".strip()
    return example

# %%
# No longer need to load local model, use vLLM's OpenAI interface
MODEL_NAME = "qwen3-14B"  # Model name configured in vLLM server

#%%
# Read training data from CSV file (for few-shot examples)
X_train = load_text_data('data/X_test_text.csv')
y_train_df = pd.read_csv('data/y_test_text.csv')

# Build training data list
train_data_list = []
for i, text in enumerate(X_train):
    train_data_list.append({
        'transcript': text,
        'participant_id': y_train_df.iloc[i]['participant_id'] if 'participant_id' in y_train_df.columns else i,
        'scores': [
            y_train_df.iloc[i]['NoInterest'] if 'NoInterest' in y_train_df.columns else 0,
            y_train_df.iloc[i]['Depressed'] if 'Depressed' in y_train_df.columns else 0,
            y_train_df.iloc[i]['Sleep'] if 'Sleep' in y_train_df.columns else 0,
            y_train_df.iloc[i]['Tired'] if 'Tired' in y_train_df.columns else 0,
            y_train_df.iloc[i]['Appetite'] if 'Appetite' in y_train_df.columns else 0,
            y_train_df.iloc[i]['Failure'] if 'Failure' in y_train_df.columns else 0,
            y_train_df.iloc[i]['Concentrating'] if 'Concentrating' in y_train_df.columns else 0,
            y_train_df.iloc[i]['Moving'] if 'Moving' in y_train_df.columns else 0,
        ],
        'Total': y_train_df.iloc[i]['Total'] if 'Total' in y_train_df.columns else 0
    })

# Read test data from CSV file
X_test = load_text_data('data/X_test_text.csv')
y_test_df = pd.read_csv('data/y_test_text.csv')

# Build test data list
data_list = []
for i, text in enumerate(X_test):
    data_list.append({
        'transcript': text,
        'participant_id': y_test_df.iloc[i]['participant_id'] if 'participant_id' in y_test_df.columns else i,
        'scores': [
            y_test_df.iloc[i]['NoInterest'] if 'NoInterest' in y_test_df.columns else 0,
            y_test_df.iloc[i]['Depressed'] if 'Depressed' in y_test_df.columns else 0,
            y_test_df.iloc[i]['Sleep'] if 'Sleep' in y_test_df.columns else 0,
            y_test_df.iloc[i]['Tired'] if 'Tired' in y_test_df.columns else 0,
            y_test_df.iloc[i]['Appetite'] if 'Appetite' in y_test_df.columns else 0,
            y_test_df.iloc[i]['Failure'] if 'Failure' in y_test_df.columns else 0,
            y_test_df.iloc[i]['Concentrating'] if 'Concentrating' in y_test_df.columns else 0,
            y_test_df.iloc[i]['Moving'] if 'Moving' in y_test_df.columns else 0,
        ],
        'Total': y_test_df.iloc[i]['Total'] if 'Total' in y_test_df.columns else 0
    })

# Few-shot configuration
# k_few_shot: Number of few-shot examples to use
# - 0: Zero-shot (no examples)
# - 1-5: Few-shot (use 1-5 training samples as examples)
# - Recommended: 3-5 examples work well
k_few_shot = 0  # Modify here to enable/disable few-shot
few_shot_data = train_data_list[:k_few_shot] if k_few_shot > 0 else []

# DEBUG mode: only test first 2 samples
DEBUG_MODE = False
if DEBUG_MODE:
    data_list = data_list[:2]

print(f'Number of training samples: {len(train_data_list)}')
print(f'Number of few-shot examples: {len(few_shot_data)}')
print(f'Number of test samples: {len(data_list)}')

#%%
# Use OpenAI API for batch inference
def generate_predictions(data_list, few_shot_data=None):
    """
    Use OpenAI API interface to perform batch prediction on data list.
    
    Args:
        data_list: Test data list
        few_shot_data: Few-shot example data list (optional)
    """
    responses = []
    texts = []
    pred_scores = []
    
    # If there are few-shot examples, build few-shot section first
    few_shot_text = ""
    if few_shot_data and len(few_shot_data) > 0:
        few_shot_examples = '\n\n'.join([construct_example(d) for d in few_shot_data])
        few_shot_text = f"\n\nHere are some examples:\n\n{few_shot_examples}\n\n"
    
    for i, data in enumerate(tqdm(data_list, desc="Generating predictions")):
        transcript = data['transcript']
        
        # Build prompt: TASK_INSTRUCTION + few-shot examples + transcript
        if few_shot_text:
            prompt_text = f"{TASK_INSTRUCTION}{few_shot_text}Now, please score this transcript:\n\nTranscript: {transcript}"
        else:
            prompt_text = f"{TASK_INSTRUCTION}\n\nTranscript: {transcript}"
        
        try:
            # Call OpenAI API
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in mental health assessment."},
                    {"role": "user", "content": prompt_text},
                ],
                # Disable thinking mode, output results directly
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    }
                },
                stream=False,
                temperature=0.1,
                max_tokens=500,  # PHQ-8 scoring doesn't require many tokens
            )
            
            answer = response.choices[0].message.content
            
            # Check if answer is empty
            if answer is None or not isinstance(answer, str):
                print(f'Warning: Sample {i} received None or non-string answer.')
                answer = ""
                scores = [1.5] * 8  # Default value
            else:
                # Extract scores
                scores = extract_scores(answer)
            
            responses.append(response)
            texts.append(answer)
            pred_scores.append(scores)
            
        except Exception as e:
            print(f'Error processing sample {i}: {e}')
            texts.append("")
            pred_scores.append([1.5] * 8)
    
    return texts, pred_scores

#%%
# %%
if __name__ == '__main__':
    # Generate predictions (pass few-shot data)
    texts, pred_scores = generate_predictions(data_list, few_shot_data=few_shot_data)
    
    # Save results
    results = []
    for data, text, scores in zip(data_list, texts, pred_scores):
        results.append({
            'participant_id': int(data['participant_id']) if isinstance(data['participant_id'], (np.integer, np.int64)) else data['participant_id'],
            'transcript': data['transcript'][:200] + '...',  # Only save first 200 characters
            'ground_truth': [float(s) if isinstance(s, (np.integer, np.floating)) else s for s in data['scores']],
            'predicted': [float(s) if isinstance(s, (np.integer, np.floating)) else s for s in scores],
            'response_text': text
        })
    
    with open('outputs.json', 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    # Calculate evaluation metrics
    all_gt_scores = np.array([d['scores'] for d in data_list])
    all_pred_scores = np.array(pred_scores)
    
    # MAE metrics
    # Micro MAE: Mean absolute error for each factor
    micro_mae = np.abs(all_gt_scores - all_pred_scores).mean(axis=0)
    # Macro MAE: Average of total error per sample
    macro_mae = np.abs(all_gt_scores - all_pred_scores).sum(axis=1).mean(axis=0)
    # Overall MAE: Mean absolute error of all predictions
    overall_mae = np.abs(all_gt_scores - all_pred_scores).mean()
    
    # RMSE metrics
    # Micro RMSE: Root mean square error for each factor
    micro_rmse = np.sqrt(((all_gt_scores - all_pred_scores) ** 2).mean(axis=0))
    # Macro RMSE: Average RMSE per sample
    macro_rmse = np.sqrt(((all_gt_scores - all_pred_scores) ** 2).mean(axis=1)).mean()
    # Macro RMSE (total per sample): Root mean square error of total score error per sample
    total_errors_per_sample = (all_gt_scores - all_pred_scores).sum(axis=1)  # Total score error per sample
    macro_rmse_total = np.sqrt((total_errors_per_sample ** 2).mean())
    # Overall RMSE: Root mean square error of all predictions
    overall_rmse = np.sqrt(((all_gt_scores - all_pred_scores) ** 2).mean())
    
    print(f'\n{"="*50}')
    print(f'Evaluation Results:')
    print(f'{"="*50}')
    print(f'\n--- MAE (Mean Absolute Error) ---')
    print(f'Micro MAE (per factor):')
    print(f'  NoInterest:    {micro_mae[0]:.3f}')
    print(f'  Depressed:     {micro_mae[1]:.3f}')
    print(f'  Sleep:         {micro_mae[2]:.3f}')
    print(f'  Tired:         {micro_mae[3]:.3f}')
    print(f'  Appetite:      {micro_mae[4]:.3f}')
    print(f'  Failure:       {micro_mae[5]:.3f}')
    print(f'  Concentrating: {micro_mae[6]:.3f}')
    print(f'  Moving:        {micro_mae[7]:.3f}')
    print(f'\nMacro MAE (total per sample): {macro_mae:.3f}')
    print(f'Overall MAE: {overall_mae:.3f}')
    
    print(f'\n--- RMSE (Root Mean Squared Error) ---')
    print(f'Micro RMSE (per factor):')
    print(f'  NoInterest:    {micro_rmse[0]:.3f}')
    print(f'  Depressed:     {micro_rmse[1]:.3f}')
    print(f'  Sleep:         {micro_rmse[2]:.3f}')
    print(f'  Tired:         {micro_rmse[3]:.3f}')
    print(f'  Appetite:      {micro_rmse[4]:.3f}')
    print(f'  Failure:       {micro_rmse[5]:.3f}')
    print(f'  Concentrating: {micro_rmse[6]:.3f}')
    print(f'  Moving:        {micro_rmse[7]:.3f}')
    print(f'\nMacro RMSE (per sample): {macro_rmse:.3f}')
    print(f'Macro RMSE (total per sample): {macro_rmse_total:.3f}')
    print(f'Overall RMSE: {overall_rmse:.3f}')
    print(f'{"="*50}\n')
    
    print(f'Results saved to outputs.json')
