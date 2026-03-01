"""
Playground script for evaluating LLMs (e.g., Qwen3-14B) on HAMD-13 datasets.
This script performs zero-shot evaluation of large language models on depression assessment tasks.

Usage:
    # Evaluate on CIDH or PDCH dataset (all data)
    python playground_HAMD13_all.py --dataset cidh
    
    # Custom output file
    python playground_HAMD13_all.py --dataset cidh --output my_results.xlsx
"""

import argparse
import os
import json
import re
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
from openai import OpenAI


# Subscale names for HAMD-13
SUBSCALE_NAMES = [
    "Guilt", "Suicide", "Insomnia_Initial", "Insomnia_Middle", "Insomnia_Late",
    "Work_Interests", "Psychic_Anxiety", "GI_Symptoms", "Somatic_Symptoms",
    "Genital_Symptoms", "Hypochondriasis", "Weight_Loss", "Insight"
]


client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8001/v1"
)


MODEL_NAME = "qwen3-14B"


def load_hamd13_data(dataset_name: str = "cidh"):
    """
    Load HAMD-13 dataset from all.json file.
    
    Args:
        dataset_name: "cidh" or "pdch"
    
    Returns:
        List of samples with transcript and scores
    """
    # Use relative path from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    
    if dataset_name.lower() == "cidh":
        # cidh dataset - use all.json
        file_path = os.path.join(data_dir, "cidh", "cidh_summary_all.json")
    elif dataset_name.lower() == "pdch":
        file_path = os.path.join(data_dir, "pdch", "pdch_summary_all.json")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def validate_scores(scores: List[int]) -> List[int]:
    max_scores = [4, 4, 2, 2, 2, 4, 4, 2, 2, 2, 4, 2, 2]
    
    validated = []
    for i, score in enumerate(scores[:13]):
        max_val = max_scores[i] if i < len(max_scores) else 2
        validated.append(max(0, min(max_val, int(score))))
    while len(validated) < 13:
        validated.append(0)
    
    return validated[:13]


def remove_json_comments(json_str: str) -> str:
    json_str = re.sub(r'```json\s*', '', json_str)
    json_str = re.sub(r'```\s*$', '', json_str, flags=re.MULTILINE)

    lines = []
    in_string = False
    escape_next = False
    
    for line in json_str.split('\n'):
        cleaned_line = []
        i = 0
        while i < len(line):
            char = line[i]
            
            if escape_next:
                cleaned_line.append(char)
                escape_next = False
                i += 1
                continue
            
            if char == '\\':
                cleaned_line.append(char)
                escape_next = True
                i += 1
                continue
            
            if char == '"':
                in_string = not in_string
                cleaned_line.append(char)
                i += 1
                continue
            
            if not in_string and i < len(line) - 1 and line[i:i+2] == '//':
                break
            
            cleaned_line.append(char)
            i += 1
        
        lines.append(''.join(cleaned_line))
    
    return '\n'.join(lines)


def extract_scores_from_json(text: str):
    try:
        json_match = re.search(r'\{[^{}]*"scores"[^{}]*\[.*?\][^{}]*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            json_str = remove_json_comments(json_str)
            try:
                result = json.loads(json_str)
                if 'scores' in result and isinstance(result['scores'], list):
                    scores = [int(s) for s in result['scores']]
                    if len(scores) >= 13:
                        return validate_scores(scores[:13])
                    elif len(scores) > 0:
                        scores = scores + [0] * (13 - len(scores))
                        return validate_scores(scores)
            except json.JSONDecodeError:
                pass
        
        scores_start = text.find('"scores"')
        if scores_start != -1:
            bracket_start = text.find('[', scores_start)
            if bracket_start != -1:
                bracket_count = 0
                bracket_end = bracket_start
                for i in range(bracket_start, len(text)):
                    if text[i] == '[':
                        bracket_count += 1
                    elif text[i] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            bracket_end = i
                            break
                
                if bracket_end > bracket_start:
                    array_content = text[bracket_start + 1:bracket_end]
                    numbers = re.findall(r'\b(\d+)\b', array_content)
                    if numbers:
                        scores = [int(n) for n in numbers]
                        if len(scores) >= 13:
                            return validate_scores(scores[:13])
                        elif len(scores) > 0:
                            scores = scores + [0] * (13 - len(scores))
                            return validate_scores(scores)
        
        array_match = re.search(r'\[[\d\s,//\-\u4e00-\u9fff]*\]', text)
        if array_match:
            array_str = array_match.group(0)
            numbers = re.findall(r'\b(\d+)\b', array_str)
            if numbers:
                scores = [int(n) for n in numbers]
                if len(scores) >= 13:
                    return validate_scores(scores[:13])
                elif len(scores) > 0:
                    scores = scores + [0] * (13 - len(scores))
                    return validate_scores(scores)
        
        print(f'Warning: Failed to extract scores from text: {text[:300]}...')
        return [0] * 13
        
    except Exception as e:
        print(f'Error extracting scores: {e}')
        print(f'Text: {text[:300]}...')
        return [0] * 13


def evaluate_llm_zero_shot(data: List[Dict], model_name: str = "Qwen3-14B", dataset_name: str = "cidh"):
    """
    Evaluate LLM in zero-shot setting using vLLM server.
    
    Args:
        data: List of samples with transcript and scores
        model_name: Name of the LLM to evaluate
        dataset_name: Dataset name for prompt formatting
    
    Returns:
        Dictionary with predictions and sample_ids
    """
    print(f"Evaluating {model_name} in zero-shot setting...")
    print(f"Using vLLM server at http://localhost:8001/v1")
    
    predictions = []
    sample_ids = []
    error_count = 0
    
    for idx, sample in enumerate(tqdm(data, desc="Evaluating")):
        # Get sample ID
        sample_id = sample.get('id', sample.get('sample_id', f"sample_{idx}"))
        sample_ids.append(sample_id)
        
        # Get transcript
        transcript = sample.get('transcript', '')
        if not transcript:
            print(f'Warning: Sample {sample_id} has no transcript, using default scores')
            predictions.append([0] * 13)
            continue
        
        # Create prompt
        prompt_text = create_prompt(transcript, dataset_name)
        
        try:
            # Call OpenAI API (vLLM server)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in mental health assessment."},
                    {"role": "user", "content": prompt_text},
                ],
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    }
                },
                stream=False,
                temperature=0.1,
                max_tokens=500,
            )
            
            answer = response.choices[0].message.content
            
            if answer is None or not isinstance(answer, str):
                print(f'Warning: Sample {sample_id} received None or non-string answer.')
                pred_scores = [0] * 13
            else:
                pred_scores = extract_scores_from_json(answer)
            
            predictions.append(pred_scores)
            
        except Exception as e:
            error_count += 1
            print(f'Error processing sample {sample_id}: {e}')
            predictions.append([0] * 13)
    
    if error_count > 0:
        print(f'\nWarning: {error_count} samples encountered errors during inference.')
    
    return {
        'predictions': predictions,
        'sample_ids': sample_ids
    }


def create_prompt(transcript: str, dataset_name: str = "cidh"):
    """
    Create a prompt for LLM evaluation.
    
    Args:
        transcript: Patient interview transcript
        dataset_name: Dataset name to determine prompt format
    
    Returns:
        Formatted prompt string
    """
    if dataset_name.lower() == "cidh":
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


def save_results_to_excel(results: Dict, output_path: str):
    """
    Save prediction results to Excel file.
    Only contains sample ID and 13 prediction scores.
    
    Args:
        results: Dictionary containing predictions and sample_ids
        output_path: Path to save the Excel file
    """
    predictions = results['predictions']
    sample_ids = results['sample_ids']
    
    # Create DataFrame with sample ID and 13 prediction scores
    data = {
        'Sample_ID': sample_ids
    }
    
    # Add 13 subscale predictions
    for i, name in enumerate(SUBSCALE_NAMES):
        data[name] = [pred[i] for pred in predictions]
    
    df = pd.DataFrame(data)
    
    # Save to Excel
    df.to_excel(output_path, index=False, engine='openpyxl')
    
    print(f"\nResults saved to Excel: {output_path}")
    print(f"Total samples: {len(sample_ids)}")
    print(f"Columns: Sample_ID + {len(SUBSCALE_NAMES)} subscale predictions")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLMs on HAMD-13 datasets")
    parser.add_argument("--dataset", type=str, default="cidh",
                       choices=["cidh", "pdch"],
                       help="Dataset name")
    parser.add_argument("--model", type=str, default="Qwen3-14B",
                       help="LLM model name")
    parser.add_argument("--output", type=str, default=None,
                       help="Output Excel file path for results")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading {args.dataset} dataset (all data)...")
    try:
        data = load_hamd13_data(args.dataset)
        print(f"Loaded {len(data)} samples")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Evaluate LLM
    results = evaluate_llm_zero_shot(data, args.model, args.dataset)
    
    # Save results to Excel
    if args.output:
        output_path = args.output
    else:
        # Default output path
        output_path = f"predictions_{args.model}_{args.dataset}.xlsx"
    
    save_results_to_excel(results, output_path)
    
    print("\n" + "="*70)
    print("Evaluation completed!")
    print("="*70)


if __name__ == "__main__":
    main()
