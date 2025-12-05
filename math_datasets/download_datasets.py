"""
Download math datasets from Hugging Face or kvpress repository.
Supports: aime24, aime25, math500, math1000, gsm8k, and other math datasets.
This script downloads the datasets and saves them in the format expected by math_pipeline.py.
"""

import argparse
import json
import os
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    import requests
except ImportError:
    requests = None


def download_from_github(dataset_name: str, output_path: str) -> bool:
    """Download dataset from kvpress GitHub repository"""
    if requests is None:
        return False
    
    base_url = "https://raw.githubusercontent.com/minghui-liu/kvpress/decode/reason"
    github_url = f"{base_url}/{dataset_name}.json"
    
    try:
        print(f"  Trying GitHub: {github_url}")
        response = requests.get(github_url, timeout=30)
        response.raise_for_status()
        
        # Save the file
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        # Verify it's valid JSON
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"  Downloaded {len(data) if isinstance(data, list) else 'unknown'} items from GitHub")
        return True
    except Exception as e:
        print(f"  GitHub download failed: {e}")
        return False


def download_aime24(output_path: str = "aime24.json"):
    """Download AIME24 dataset from Hugging Face or GitHub"""
    print("Downloading AIME24 dataset...")
    
    # Try Hugging Face first
    if load_dataset is not None:
        try:
            dataset = load_dataset("openai/aime24", split="test")
            problems = []
            for item in dataset:
                problems.append({
                    "id": item.get("id", len(problems) + 1),
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "solution": item.get("solution", ""),
                })
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(problems, f, indent=2, ensure_ascii=False)
            
            print(f"Downloaded {len(problems)} problems to {output_path}")
            return output_path
        except Exception as e:
            print(f"  Hugging Face download failed: {e}")
            print("  Trying GitHub...")
    
    # Try GitHub
    if download_from_github("aime24", output_path):
        return output_path
    
    print("Error: Could not download AIME24 from Hugging Face or GitHub")
    print("Please download manually from: https://github.com/minghui-liu/kvpress/tree/decode/reason")
    return None


def download_aime25(output_path: str = "aime25.json"):
    """Download AIME25 dataset from Hugging Face or GitHub"""
    print("Downloading AIME25 dataset...")
    
    # Try Hugging Face first
    if load_dataset is not None:
        try:
            dataset = load_dataset("openai/aime25", split="test")
            problems = []
            for item in dataset:
                problems.append({
                    "id": item.get("id", len(problems) + 1),
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "solution": item.get("solution", ""),
                })
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(problems, f, indent=2, ensure_ascii=False)
            
            print(f"Downloaded {len(problems)} problems to {output_path}")
            return output_path
        except Exception as e:
            print(f"  Hugging Face download failed: {e}")
            print("  Trying GitHub...")
    
    # Try GitHub
    if download_from_github("aime25", output_path):
        return output_path
    
    print("Error: Could not download AIME25 from Hugging Face or GitHub")
    print("Please download manually from: https://github.com/minghui-liu/kvpress/tree/decode/reason")
    return None


def download_math500(output_path: str = "math500.json", num_problems: int = 500):
    """Download MATH dataset (first N problems from MATH dataset) or from GitHub"""
    print(f"Downloading MATH{num_problems} dataset...")
    
    # Try Hugging Face first
    if load_dataset is not None:
        try:
            # Load test split and take first N problems
            dataset = load_dataset("hendrycks/competition_math", split="test")
            problems = []
            for i, item in enumerate(dataset):
                if i >= num_problems:
                    break
                # Extract answer from solution (format: ...#### answer)
                solution = item.get("solution", "")
                answer = ""
                if "####" in solution:
                    answer = solution.split("####")[-1].strip()
                else:
                    # Try to extract final answer from solution
                    answer = solution.strip().split("\n")[-1] if solution else ""
                
                problems.append({
                    "id": i + 1,
                    "question": item.get("problem", ""),
                    "answer": answer,
                    "solution": solution,
                })
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(problems, f, indent=2, ensure_ascii=False)
            
            print(f"Downloaded {len(problems)} problems to {output_path}")
            return output_path
        except Exception as e:
            print(f"  Hugging Face download failed: {e}")
            print("  Trying GitHub...")
    
    # Try GitHub (only for math500, not math1000)
    if num_problems == 500:
        if download_from_github("math500", output_path):
            return output_path
    
    print(f"Error: Could not download MATH{num_problems} from Hugging Face or GitHub")
    print("Please download manually from: https://github.com/minghui-liu/kvpress/tree/decode/reason")
    return None


def download_gsm8k(output_path: str = "gsm8k.json", split: str = "test"):
    """Download GSM8K dataset from Hugging Face"""
    print(f"Downloading GSM8K dataset ({split} split)...")
    try:
        dataset = load_dataset("gsm8k", "main", split=split)
        problems = []
        for item in dataset:
            # GSM8K answer field contains the solution with final answer after ####
            answer_text = item.get("answer", "")
            answer = ""
            solution = answer_text
            
            if "####" in answer_text:
                parts = answer_text.split("####")
                solution = parts[0].strip()
                answer = parts[-1].strip()
            else:
                # If no #### separator, try to extract final number
                answer = answer_text.strip()
            
            problems.append({
                "id": item.get("id", len(problems) + 1),
                "question": item.get("question", ""),
                "answer": answer,
                "solution": solution,
            })
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(problems, f, indent=2, ensure_ascii=False)
        
        print(f"Downloaded {len(problems)} problems to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading GSM8K: {e}")
        print("Note: If dataset not found, you may need to download manually from kvpress repository")
        return None


def download_dataset(dataset_name: str, output_path: str = None):
    """
    Download a dataset by name.
    Supported: aime24, aime25, math500, math1000, gsm8k, gsm8k_train
    """
    if output_path is None:
        output_path = f"{dataset_name}.json"
    
    dataset_name_lower = dataset_name.lower()
    
    if dataset_name_lower == "aime24":
        return download_aime24(output_path)
    elif dataset_name_lower == "aime25":
        return download_aime25(output_path)
    elif dataset_name_lower == "math500":
        return download_math500(output_path, num_problems=500)
    elif dataset_name_lower == "math1000":
        return download_math500(output_path, num_problems=1000)
    elif dataset_name_lower == "gsm8k" or dataset_name_lower == "gsm8k_test":
        return download_gsm8k(output_path, split="test")
    elif dataset_name_lower == "gsm8k_train":
        return download_gsm8k(output_path, split="train")
    else:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Supported datasets: aime24, aime25, math500, math1000, gsm8k, gsm8k_train")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download math datasets for the math_pipeline.py pipeline"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["aime24", "aime25", "math500", "gsm8k"],
        help="Dataset names to download (default: aime24 aime25 math500 gsm8k). "
             "Supported: aime24, aime25, math500, math1000, gsm8k, gsm8k_train",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available datasets",
    )
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    print("=" * 80)
    print("Downloading Math Datasets")
    print("=" * 80)
    
    # Determine which datasets to download
    if args.all:
        datasets_to_download = ["aime24", "aime25", "math500", "math1000", "gsm8k", "gsm8k_train"]
    else:
        datasets_to_download = args.datasets
    
    # Download datasets
    results = {}
    for dataset_name in datasets_to_download:
        output_path = os.path.join(script_dir, f"{dataset_name}.json")
        result = download_dataset(dataset_name, output_path)
        results[dataset_name] = result
        print()  # Empty line between downloads
    
    print("=" * 80)
    successful = [name for name, path in results.items() if path is not None]
    failed = [name for name, path in results.items() if path is None]
    
    if successful:
        print("Successfully downloaded:")
        for name in successful:
            print(f"  - {name}: {results[name]}")
    
    if failed:
        print("\nFailed to download:")
        for name in failed:
            print(f"  - {name}")
        print("\nTo download manually:")
        print("1. Visit: https://github.com/minghui-liu/kvpress/tree/decode/reason")
        print("2. Click on the dataset file (e.g., aime24.json)")
        print("3. Click 'Raw' button to download")
        print("4. Save the file to:", script_dir)
        print("\nOr use curl/wget:")
        for name in failed:
            print(f"  curl -o {script_dir}/{name}.json https://raw.githubusercontent.com/minghui-liu/kvpress/decode/reason/{name}.json")
    
    print("=" * 80)
