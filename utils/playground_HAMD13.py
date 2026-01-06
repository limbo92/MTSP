"""
Playground script for evaluating LLMs (e.g., Qwen3-14B) on HAMD-13 datasets.
This script performs zero-shot evaluation of large language models on depression assessment tasks.

Usage:
    # Evaluate on EvaluateTape dataset
    python playground_HAMD13.py --dataset evaluatetape
    
    # Evaluate on PDCH dataset
    python playground_HAMD13.py --dataset pdch
"""

import argparse
import os
import json
import numpy as np
from typing import List, Dict
from tqdm import tqdm


def load_hamd13_data(dataset_name: str = "evaluatetape", split: str = "test"):
    """
    Load HAMD-13 dataset.
    
    Args:
        dataset_name: "evaluatetape" or "pdch"
        split: "train", "val", or "test"
    
    Returns:
        List of samples with transcript and scores
    """
    data_dir = "../data"
    
    if dataset_name.lower() == "evaluatetape":
        # EvaluateTape dataset (summarized format)
        file_path = os.path.join(data_dir, "evaluatetape", f"eval_summary_{split}.json")
    elif dataset_name.lower() == "pdch":
        file_path = os.path.join(data_dir, "pdch", f"pdch_original_{split}.json")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def calculate_mae_rmse(predictions: List[List[int]], ground_truth: List[List[int]]):
    """
    Calculate MAE and RMSE for subscales and total score.
    
    Args:
        predictions: List of predicted scores for each sample
        ground_truth: List of ground truth scores for each sample
    
    Returns:
        Dictionary with MAE and RMSE metrics
    """
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Per-subscale metrics
    subscale_mae = np.mean(np.abs(predictions - ground_truth), axis=0)
    subscale_rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2, axis=0))
    
    # Total score metrics
    pred_total = predictions.sum(axis=1)
    true_total = ground_truth.sum(axis=1)
    total_mae = np.mean(np.abs(pred_total - true_total))
    total_rmse = np.sqrt(np.mean((pred_total - true_total) ** 2))
    
    # Average subscale metrics
    avg_subscale_mae = np.mean(subscale_mae)
    avg_subscale_rmse = np.mean(subscale_rmse)
    
    return {
        'subscale_mae': subscale_mae.tolist(),
        'subscale_rmse': subscale_rmse.tolist(),
        'avg_subscale_mae': avg_subscale_mae,
        'avg_subscale_rmse': avg_subscale_rmse,
        'total_mae': total_mae,
        'total_rmse': total_rmse
    }


def evaluate_llm_zero_shot(data: List[Dict], model_name: str = "Qwen3-14B"):
    """
    Evaluate LLM in zero-shot setting.
    
    This is a placeholder function. In practice, you would:
    1. Load the LLM model (e.g., using transformers or vLLM)
    2. Create prompts for each sample
    3. Generate predictions
    4. Parse the outputs to extract scores
    
    Args:
        data: List of samples with transcript and scores
        model_name: Name of the LLM to evaluate
    
    Returns:
        Dictionary with predictions and metrics
    """
    print(f"Evaluating {model_name} in zero-shot setting...")
    print(f"Note: This is a placeholder. Implement actual LLM inference here.")
    
    # Placeholder: Return random predictions for demonstration
    # In practice, replace this with actual LLM inference
    predictions = []
    ground_truth = []
    
    for sample in tqdm(data, desc="Evaluating"):
        # Get ground truth
        if 'scores' in sample:
            scores = sample['scores']
        elif 'subscales' in sample:
            scores = sample['subscales']
        else:
            continue
        
        # Ensure 13 scores
        if len(scores) != 13:
            scores = scores[:13] + [0] * (13 - len(scores))
        
        ground_truth.append(scores)
        
        # Placeholder prediction (replace with actual LLM inference)
        # For demonstration, use a simple heuristic or random values
        # In practice, you would:
        # 1. Create a prompt from the transcript
        # 2. Call the LLM API or model
        # 3. Parse the response to extract HAMD-13 scores
        pred_scores = [max(0, min(4, int(s + np.random.normal(0, 0.5)))) for s in scores]
        predictions.append(pred_scores)
    
    # Calculate metrics
    metrics = calculate_mae_rmse(predictions, ground_truth)
    
    return {
        'predictions': predictions,
        'ground_truth': ground_truth,
        'metrics': metrics
    }


def create_prompt(transcript: str, dataset_name: str = "evaluatetape"):
    """
    Create a prompt for LLM evaluation.
    
    Args:
        transcript: Patient interview transcript
        dataset_name: Dataset name to determine prompt format
    
    Returns:
        Formatted prompt string
    """
    if dataset_name.lower() == "evaluatetape":
        # Summarized format - transcript already contains subscale descriptions
        prompt = f"""请根据以下患者访谈摘要，评估HAMD-13量表的13个子量表分数。

访谈摘要：
{transcript}

请为以下13个子量表分别打分（每个子量表的分数范围已标注）：

1. 罪恶感 (Guilt): 0-4分
2. 自杀意念 (Suicide): 0-4分
3. 入睡困难 (Insomnia_Initial): 0-2分
4. 睡眠中段 (Insomnia_Middle): 0-2分
5. 早醒 (Insomnia_Late): 0-2分
6. 工作和兴趣 (Work_Interests): 0-4分
7. 精神性焦虑 (Psychic_Anxiety): 0-4分
8. 胃肠道症状 (GI_Symptoms): 0-2分
9. 躯体症状 (Somatic_Symptoms): 0-2分
10. 性症状 (Genital_Symptoms): 0-2分
11. 疑病 (Hypochondriasis): 0-4分
12. 体重减轻 (Weight_Loss): 0-2分
13. 自知力 (Insight): 0-2分

请以JSON格式输出，格式如下：
{{"scores": [分数1, 分数2, ..., 分数13]}}
"""
    else:
        # Dialogue format
        prompt = f"""请根据以下医患对话，评估HAMD-13量表的13个子量表分数。

对话内容：
{transcript}

请为以下13个子量表分别打分（每个子量表的分数范围已标注）：

1. 罪恶感 (Guilt): 0-4分
2. 自杀意念 (Suicide): 0-4分
3. 入睡困难 (Insomnia_Initial): 0-2分
4. 睡眠中段 (Insomnia_Middle): 0-2分
5. 早醒 (Insomnia_Late): 0-2分
6. 工作和兴趣 (Work_Interests): 0-4分
7. 精神性焦虑 (Psychic_Anxiety): 0-4分
8. 胃肠道症状 (GI_Symptoms): 0-2分
9. 躯体症状 (Somatic_Symptoms): 0-2分
10. 性症状 (Genital_Symptoms): 0-2分
11. 疑病 (Hypochondriasis): 0-4分
12. 体重减轻 (Weight_Loss): 0-2分
13. 自知力 (Insight): 0-2分

请以JSON格式输出，格式如下：
{{"scores": [分数1, 分数2, ..., 分数13]}}
"""
    
    return prompt


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLMs on HAMD-13 datasets")
    parser.add_argument("--dataset", type=str, default="evaluatetape",
                       choices=["evaluatetape", "pdch"],
                       help="Dataset name")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "val", "test"],
                       help="Data split to evaluate")
    parser.add_argument("--model", type=str, default="Qwen3-14B",
                       help="LLM model name")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file path for results")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading {args.dataset} dataset ({args.split} split)...")
    data = load_hamd13_data(args.dataset, args.split)
    print(f"Loaded {len(data)} samples")
    
    # Evaluate LLM
    results = evaluate_llm_zero_shot(data, args.model)
    
    # Print results
    print("\n" + "="*70)
    print(f"Evaluation Results: {args.model} on {args.dataset} ({args.split})")
    print("="*70)
    print(f"\nAverage Subscale MAE: {results['metrics']['avg_subscale_mae']:.4f}")
    print(f"Average Subscale RMSE: {results['metrics']['avg_subscale_rmse']:.4f}")
    print(f"Total Score MAE: {results['metrics']['total_mae']:.4f}")
    print(f"Total Score RMSE: {results['metrics']['total_rmse']:.4f}")
    
    print("\nPer-subscale MAE:")
    subscale_names = [
        "Guilt", "Suicide", "Insomnia_Initial", "Insomnia_Middle", "Insomnia_Late",
        "Work_Interests", "Psychic_Anxiety", "GI_Symptoms", "Somatic_Symptoms",
        "Genital_Symptoms", "Hypochondriasis", "Weight_Loss", "Insight"
    ]
    for i, (name, mae) in enumerate(zip(subscale_names, results['metrics']['subscale_mae'])):
        print(f"  {i+1:2d}. {name:20s}: {mae:.4f}")
    
    # Save results
    if args.output:
        output_data = {
            'dataset': args.dataset,
            'split': args.split,
            'model': args.model,
            'num_samples': len(data),
            'metrics': results['metrics'],
            'predictions': results['predictions'],
            'ground_truth': results['ground_truth']
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")
    else:
        # Default output path
        output_path = f"llm_results_{args.model}_{args.dataset}_{args.split}.json"
        output_data = {
            'dataset': args.dataset,
            'split': args.split,
            'model': args.model,
            'num_samples': len(data),
            'metrics': results['metrics'],
            'predictions': results['predictions'],
            'ground_truth': results['ground_truth']
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")
    
    print("\n" + "="*70)
    print("Note: This script currently uses placeholder predictions.")
    print("To use actual LLM inference, implement the evaluate_llm_zero_shot function")
    print("with your LLM API or model loading code.")
    print("="*70)


if __name__ == "__main__":
    main()

