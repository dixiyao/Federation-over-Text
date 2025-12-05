"""
Math Problem Solving Pipeline
Uses client.py to learn skills from dataset1, aggregates with server.py,
and uses generate_server.py to solve problems in dataset2.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from client import ChainOfThoughtReader
from generate_server import GenerateServer
from server import SkillAggregationServer


class MathPipeline:
    """
    Pipeline for learning skills from math problems and applying them to new problems.
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        device: Optional[str] = None,
        output_dir: str = "math_output",
    ):
        self.model_name = model_name
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.client = None
        self.server = None
        self.generate_server = None

    def load_math_dataset(self, dataset_path: str) -> List[Dict]:
        """
        Load math dataset from JSON file.
        
        Expected formats:
        1. List of problems: [{"question": "...", "answer": "...", ...}, ...]
        2. Dict with problems: {"problems": [...], ...}
        3. Dict with data: {"data": [...], ...}
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle different dataset formats
        problems = []
        if isinstance(data, list):
            problems = data
        elif isinstance(data, dict):
            # Try common keys (kvpress style)
            if "problems" in data:
                problems = data["problems"]
            elif "data" in data:
                problems = data["data"]
            elif "test" in data:
                problems = data["test"]
            elif "train" in data:
                problems = data["train"]
            else:
                # Assume it's a single problem
                problems = [data]
        else:
            raise ValueError(f"Unexpected dataset format: {type(data)}")
        
        # Normalize problem format to have consistent keys
        normalized = []
        for idx, problem in enumerate(problems):
            # Handle different field names
            normalized_problem = {
                "id": problem.get("id", problem.get("problem_id", idx + 1)),
                "problem": problem.get("problem", problem.get("question", problem.get("text", ""))),
                "question": problem.get("question", problem.get("problem", problem.get("text", ""))),
                "solution": problem.get("solution", problem.get("step_by_step", "")),
                "answer": problem.get("answer", problem.get("final_answer", "")),
            }
            # Copy any additional fields
            for key, value in problem.items():
                if key not in normalized_problem:
                    normalized_problem[key] = value
            
            normalized.append(normalized_problem)
        
        return normalized

    def learn_skills_from_dataset1(
        self, dataset1_path: str, max_problems: Optional[int] = None
    ) -> str:
        """
        Step 1: Use client.py to learn skills from dataset1 (e.g., aime25).
        Each question uses a client to extract skills.
        
        Returns:
            Path to directory containing skill JSON files
        """
        print("\n" + "=" * 80)
        print("STEP 1: Learning Skills from Dataset1")
        print("=" * 80)
        
        # Load dataset1
        problems = self.load_math_dataset(dataset1_path)
        if max_problems:
            problems = problems[:max_problems]
        
        print(f"Loaded {len(problems)} problems from {dataset1_path}")
        
        # Create client instance
        self.client = ChainOfThoughtReader(
            model_name=self.model_name,
            device=self.device,
        )
        
        # Output directory for skill books
        skills_output_dir = os.path.join(self.output_dir, "skills")
        os.makedirs(skills_output_dir, exist_ok=True)
        
        # Process each problem to extract skills
        print(f"\nProcessing {len(problems)} problems to extract skills...")
        for idx, problem_data in enumerate(problems, 1):
            # Try both 'problem' and 'question' fields (normalized in load_math_dataset)
            problem_text = problem_data.get("problem") or problem_data.get("question", "")
            if not problem_text:
                print(f"Warning: Problem {idx} has no problem/question field. Skipping...")
                continue
            
            print(f"\n[{idx}/{len(problems)}] Processing problem...")
            print(f"Problem: {problem_text[:100]}...")
            
            try:
                # Use client to extract skills from this problem
                # The task/question is the problem itself
                result = self.client.read_paper(task=problem_text, paper_content=None)
                
                # Save skill book for this problem
                skill_book = result.get("behavior_book", {})
                if skill_book:
                    output_data = {
                        "problem": problem_text,
                        "problem_id": problem_data.get("id", idx),
                        "behaviors": [
                            {
                                "behavior": name.replace("behavior_", ""),
                                "description": desc
                            }
                            for name, desc in skill_book.items()
                        ],
                        "behavior_book": skill_book,
                    }
                    
                    output_path = os.path.join(skills_output_dir, f"problem_{idx:04d}.json")
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(output_data, f, indent=2, ensure_ascii=False)
                    
                    print(f"  Extracted {len(skill_book)} skills -> {output_path}")
                else:
                    print(f"  No skills extracted from this problem")
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"  Error processing problem {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\nCompleted skill extraction. Skills saved to: {skills_output_dir}")
        return skills_output_dir

    def aggregate_skills(self, skills_dir: str, r1: float = 0.9, r2: float = 0.4) -> str:
        """
        Step 2: Use server.py to aggregate skills from all problems.
        
        Returns:
            Path to encyclopedia file
        """
        print("\n" + "=" * 80)
        print("STEP 2: Aggregating Skills")
        print("=" * 80)
        
        # Create server instance
        self.server = SkillAggregationServer(
            model_name=self.model_name,
            device=self.device,
            input_dir=skills_dir,
        )
        
        # Aggregate skills
        result = self.server.aggregate_and_build_encyclopedia(
            json_files=None,  # Use all JSON files in skills_dir
            r1=r1,
            r2=r2,
            output_dir=self.output_dir,
        )
        
        # Encyclopedia path
        encyclopedia_path = os.path.join(self.output_dir, "encyclopedia.txt")
        
        print(f"\nSkills aggregated. Encyclopedia saved to: {encyclopedia_path}")
        return encyclopedia_path

    def solve_dataset2(
        self,
        dataset2_path: str,
        encyclopedia_path: str,
        max_problems: Optional[int] = None,
    ) -> List[Dict]:
        """
        Step 3: Use generate_server.py to solve problems in dataset2 (e.g., math500).
        Uses the learned encyclopedia with minimal tokens.
        
        Returns:
            List of results with predictions
        """
        print("\n" + "=" * 80)
        print("STEP 3: Solving Dataset2 with Learned Skills")
        print("=" * 80)
        
        # Load dataset2
        problems = self.load_math_dataset(dataset2_path)
        if max_problems:
            problems = problems[:max_problems]
        
        print(f"Loaded {len(problems)} problems from {dataset2_path}")
        
        # Create generate server instance
        self.generate_server = GenerateServer(
            model_name=self.model_name,
            device=self.device,
        )
        
        # Load encyclopedia
        self.generate_server.load_encyclopedia(encyclopedia_path)
        
        # Optimize prompt for minimal tokens while maintaining effectiveness
        # Load and parse encyclopedia to extract only relevant skills
        with open(encyclopedia_path, "r", encoding="utf-8") as f:
            encyclopedia_content = f.read()
        
        # Try to parse as JSON to extract skills more intelligently
        try:
            encyclopedia_json = json.loads(encyclopedia_content)
            # Extract skills from JSON structure
            all_skills = []
            if "merged_skills" in encyclopedia_json:
                for skill in encyclopedia_json["merged_skills"]:
                    all_skills.append(f"{skill.get('skill_name', '')}: {skill.get('description', '')}")
            if "clusters" in encyclopedia_json:
                for cluster in encyclopedia_json["clusters"]:
                    for skill in cluster.get("skills", []):
                        all_skills.append(f"{skill.get('skill_name', '')}: {skill.get('description', '')}")
            if "standalone_skills" in encyclopedia_json:
                for skill in encyclopedia_json["standalone_skills"]:
                    all_skills.append(f"{skill.get('skill_name', '')}: {skill.get('description', '')}")
            
            # Create compact skills text (limit to ~3000 chars for minimal tokens)
            skills_text = "\n".join(all_skills[:50])  # Limit to top 50 skills
            if len(skills_text) > 3000:
                skills_text = skills_text[:3000] + "..."
        except:
            # Fallback: use raw encyclopedia but truncate
            skills_text = encyclopedia_content[:3000] + "..." if len(encyclopedia_content) > 3000 else encyclopedia_content
        
        # Override the generation prompt to be more concise
        original_prompt_method = self.generate_server._get_generation_prompt
        
        def optimized_prompt(query: str) -> str:
            """Optimized prompt with minimal tokens - focuses on relevant skills"""
            # Use a very concise prompt that minimizes token usage
            prompt = f"""Skills:
{skills_text}

Q: {query}

Solve using relevant skills. Be concise.

## Answer:
"""
            return prompt
        
        self.generate_server._get_generation_prompt = optimized_prompt
        
        # Solve each problem
        results = []
        print(f"\nSolving {len(problems)} problems...")
        
        for idx, problem_data in enumerate(problems, 1):
            # Try both 'problem' and 'question' fields (normalized in load_math_dataset)
            problem_text = problem_data.get("problem") or problem_data.get("question", "")
            if not problem_text:
                print(f"Warning: Problem {idx} has no problem/question field. Skipping...")
                continue
            
            print(f"\n[{idx}/{len(problems)}] Solving problem...")
            print(f"Problem: {problem_text[:100]}...")
            
            try:
                # Generate answer using encyclopedia
                answer = self.generate_server.generate(problem_text)
                
                # Extract answer from response
                answer_text = answer
                if "## Answer:" in answer:
                    start_idx = answer.find("## Answer:") + len("## Answer:")
                    end_idx = answer.find("## End of Answer:")
                    if end_idx == -1:
                        end_idx = len(answer)
                    answer_text = answer[start_idx:end_idx].strip()
                
                result = {
                    "problem_id": problem_data.get("id", idx),
                    "problem": problem_text,
                    "question": problem_text,  # Keep both for compatibility
                    "predicted_answer": answer_text,
                    "full_response": answer,
                    "ground_truth": problem_data.get("answer") or problem_data.get("solution", ""),
                    "ground_truth_solution": problem_data.get("solution", ""),
                }
                
                results.append(result)
                print(f"  Generated answer: {answer_text[:100]}...")
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"  Error solving problem {idx}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "problem_id": problem_data.get("id", idx),
                    "problem": problem_text,
                    "predicted_answer": "",
                    "error": str(e),
                })
                continue
        
        # Save results
        results_path = os.path.join(self.output_dir, "dataset2_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {results_path}")
        return results

    def run_full_pipeline(
        self,
        dataset1_path: str,
        dataset2_path: str,
        max_dataset1: Optional[int] = None,
        max_dataset2: Optional[int] = None,
        r1: float = 0.9,
        r2: float = 0.4,
    ) -> Dict:
        """
        Run the complete pipeline:
        1. Learn skills from dataset1
        2. Aggregate skills
        3. Solve dataset2 using learned skills
        
        Returns:
            Dictionary with results and statistics
        """
        start_time = time.time()
        
        # Step 1: Learn skills from dataset1
        skills_dir = self.learn_skills_from_dataset1(dataset1_path, max_problems=max_dataset1)
        
        # Step 2: Aggregate skills
        encyclopedia_path = self.aggregate_skills(skills_dir, r1=r1, r2=r2)
        
        # Step 3: Solve dataset2
        results = self.solve_dataset2(dataset2_path, encyclopedia_path, max_problems=max_dataset2)
        
        # Calculate statistics
        total_time = time.time() - start_time
        num_correct = sum(
            1 for r in results
            if r.get("predicted_answer") and r.get("ground_truth")
            and self._check_answer_match(r["predicted_answer"], r["ground_truth"])
        )
        
        summary = {
            "dataset1_path": dataset1_path,
            "dataset2_path": dataset2_path,
            "num_dataset1_problems": max_dataset1 or "all",
            "num_dataset2_problems": len(results),
            "num_correct": num_correct,
            "accuracy": num_correct / len(results) if results else 0.0,
            "total_time_seconds": total_time,
            "encyclopedia_path": encyclopedia_path,
            "results_path": os.path.join(self.output_dir, "dataset2_results.json"),
        }
        
        # Save summary
        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(f"Dataset1 problems processed: {max_dataset1 or 'all'}")
        print(f"Dataset2 problems solved: {len(results)}")
        print(f"Correct answers: {num_correct}/{len(results)}")
        print(f"Accuracy: {summary['accuracy']:.2%}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Summary saved to: {summary_path}")
        print("=" * 80)
        
        return summary

    def _check_answer_match(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if predicted answer matches ground truth.
        For math problems, extract final numerical answer and compare.
        """
        # Extract numbers from both answers
        import re
        
        def extract_numbers(text: str) -> List[float]:
            # Find all numbers (including decimals and negatives)
            numbers = re.findall(r'-?\d+\.?\d*', text)
            return [float(n) for n in numbers if n]
        
        pred_nums = extract_numbers(predicted)
        gt_nums = extract_numbers(ground_truth)
        
        if not pred_nums or not gt_nums:
            # Fallback to string comparison
            return predicted.strip().lower() == ground_truth.strip().lower()
        
        # Check if any predicted number matches any ground truth number
        for p in pred_nums:
            for g in gt_nums:
                if abs(p - g) < 1e-6:  # Allow for floating point errors
                    return True
        
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Math Problem Solving Pipeline - Learn skills from dataset1 and apply to dataset2"
    )
    parser.add_argument(
        "--dataset1",
        type=str,
        default="aime25",
        help="Dataset name for learning skills (e.g., aime25). File should be in math_datasets/{name}.json (default: aime25)",
    )
    parser.add_argument(
        "--dataset2",
        type=str,
        default="math500",
        help="Dataset name for testing (e.g., math500). File should be in math_datasets/{name}.json (default: math500)",
    )
    parser.add_argument(
        "--max-dataset1",
        type=int,
        default=None,
        help="Maximum number of problems to process from dataset1 (default: all)",
    )
    parser.add_argument(
        "--max-dataset2",
        type=int,
        default=None,
        help="Maximum number of problems to solve from dataset2 (default: all)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Hugging Face model name (default: deepseek-ai/DeepSeek-R1-Distill-Llama-8B)",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda' or 'cpu' (default: auto-detect)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="math_output",
        help="Output directory for results (default: math_output)",
    )
    parser.add_argument(
        "--r1",
        type=float,
        default=0.95,
        help="Threshold r1 for same skills (default: 0.95)",
    )
    parser.add_argument(
        "--r2",
        type=float,
        default=0.6,
        help="Threshold r2 for linked skills (default: 0.6)",
    )

    args = parser.parse_args()

    # Convert dataset names to file paths
    # If it's already a path (contains / or .json), use it as-is
    # Otherwise, treat it as a dataset name and look in math_datasets/
    if "/" in args.dataset1 or args.dataset1.endswith(".json"):
        dataset1_path = args.dataset1
    else:
        dataset1_path = f"math_datasets/{args.dataset1}.json"
    
    if "/" in args.dataset2 or args.dataset2.endswith(".json"):
        dataset2_path = args.dataset2
    else:
        dataset2_path = f"math_datasets/{args.dataset2}.json"

    # Create pipeline
    pipeline = MathPipeline(
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir,
    )

    try:
        # Run full pipeline
        summary = pipeline.run_full_pipeline(
            dataset1_path=dataset1_path,
            dataset2_path=dataset2_path,
            max_dataset1=args.max_dataset1,
            max_dataset2=args.max_dataset2,
            r1=args.r1,
            r2=args.r2,
        )

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you have:")
        print("1. Installed required packages: pip install -r requirements.txt")
        print("2. Dataset files in JSON format")
        print("3. For GPU support, ensure CUDA is properly installed")
        print("\nExample usage:")
        print("  # Use default datasets (aime25 and math500)")
        print("  python math_pipeline.py")
        print("  # Use custom dataset names")
        print("  python math_pipeline.py --dataset1 aime25 --dataset2 math500")
        print("  # With limits")
        print("  python math_pipeline.py --max-dataset1 10 --max-dataset2 50")
        print("  # Use full paths if needed")
        print("  python math_pipeline.py --dataset1 math_datasets/aime25.json --dataset2 math_datasets/math500.json")
        print("\nTo download datasets:")
        print("  python math_datasets/download_datasets.py")

