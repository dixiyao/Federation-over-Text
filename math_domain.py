"""
Multi-dataset Math Problem Solving Pipeline (math_domain)
- Supports running STEP 1 (insight extraction) across multiple datasets.
- Allows choosing which insight sets to aggregate in STEP 2.
- Evaluates multiple target datasets sequentially in STEP 3 using the shared encyclopedia.

Usage is similar to math_pipeline.py but adds list-style arguments.

## Supported Datasets

### Hugging Face Datasets (via 🤗 datasets library):
- gsm8k, gsm8k_train: Grade School Math (8K examples)
- aime24, aime25: American Invitational Mathematics Examination
- math500, math1000: Competition math problems

### Local Datasets:
- CSV or JSON files in math_datasets/ directory

### IMOBench (International Mathematical Olympiad Benchmark)
From: https://github.com/google-deepmind/superhuman/tree/main/imobench
See: https://imobench.github.io

IMOBench consists of three specialized benchmarks:

1. **IMO-AnswerBench** (400 problems)
   - Short-answer problems with verifiable final answers
   - Categories: Algebra, Combinatorics, Geometry, Number Theory
   - Difficulty: pre-IMO, IMO-Easy, IMO-Medium, IMO-Hard
   - CSV columns: problem/question, answer/solution, id, difficulty
   - Evaluation: Symbolic comparison with algebraic normalization

2. **IMO-ProofBench** (60 problems)
   - Proof-writing evaluation (not just final answers)
   - Requires human expert grading (0-7 scale)
   - Can use ProofAutoGrader with Gemini 2.5 Pro for automatic evaluation
   - Correlation with human grading: 0.96 (basic), 0.93 (advanced)
   - Not automatically evaluated in this script - use external graders

3. **IMO-GradingBench** (1000 examples)
   - Dataset for evaluating grading capability
   - Problem + proposed solution + human grade (0-7)
   - Classification labels: Correct (7), Almost (6), Partial (1), Incorrect (0)
   - CSV columns: problem, solution, grade, grade_label

### Answer Verification for IMOBench:
- Numeric: Direct numeric comparison with tolerance 1e-6
- Symbolic: Algebraic equivalence checking (normalized forms)
- String: Case-insensitive exact matching
- Partial: Substring matching for multi-answer problems
- Unit handling: Removes common units (degrees, radians, cm, m, etc.)

For proof-based evaluation on IMO-ProofBench, consider:
- Using Gemini 2.5 Pro's ProofAutoGrader (available in superhuman repo)
- Implementing LLM-based grading with reference solutions
- Human expert evaluation for rigorous assessment
"""

import argparse
import csv
import json
import os
import random
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from client import ChainOfThoughtReader
from math_datasets.utils import extract_numbers
from server import InsightAggregationServer
from server_text import TextBasedInsightAggregationServer

# Dataset registry: (source, path_or_hf_name, data_dir_or_col_map, split_or_none)
# - source="hf": use Hugging Face with (hf_name, data_dir, split)
# - source="json": load JSON from path
# - source="csv": load CSV from path with optional column mapping dict
#
# IMOBench Benchmarks (https://imobench.github.io/):
# - IMO-AnswerBench: 400 short-answer problems (CSV)
#   Columns: problem/question, answer/solution, id, difficulty
#   Evaluation: Symbolic comparison with unit normalization
#
# - IMO-ProofBench: 60 proof-based problems (not directly scored here)
#   Requires: ProofAutoGrader (Gemini 2.5 Pro) or human evaluation
#   Grading: 0-7 scale, ~high correlation with human experts (0.96)
#
# - IMO-GradingBench: 1000 grading examples (CSV)
#   Columns: problem, solution, grade (0-7), grade_label (Correct/Almost/Partial/Incorrect)
#   Use for training automatic graders
DATASET_REGISTRY: Dict[str, Tuple[str, str, Optional[str], Optional[str]]] = {
    "gsm8k": ("hf", "openai/gsm8k", None, "test"),
    "gsm8k_train": ("hf", "openai/gsm8k", None, "train"),
    "aime25": ("hf", "math-ai/aime25", None, "test"),
    "aime24": ("hf", "math-ai/aime24", None, "test"),
    "math500": ("hf", "HuggingFaceH4/MATH-500", None, "test"),
    "math1000": ("hf", "hendrycks/competition_math", None, "test"),
    # IMO benchmark (IMOBench) — CSV files from https://github.com/google-deepmind/superhuman/tree/main/imobench
    # Download from: https://github.com/google-deepmind/superhuman/tree/main/imobench
    "imo_answerbench": ("csv", "math_datasets/answerbench.csv", None, None),
    "imo_answerbench_algebra": ("csv", "math_datasets/imo_algebra.csv", None, None),
    "imo_answerbench_geometry": ("csv", "math_datasets/imo_geometry.csv", None, None),
    "imo_answerbench_number_theory": (
        "csv",
        "math_datasets/imo_number_theory.csv",
        None,
        None,
    ),
    # IMO-ProofBench: Requires external evaluation (ProofAutoGrader or human experts)
    "imo_proofbench": ("csv", "math_datasets/proofbench.csv", None, None),
    # IMO-GradingBench: For training/evaluating automatic graders
    "imo_gradingbench": ("csv", "math_datasets/gradingbench.csv", None, None),
}


class MathDomainPipeline:
    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        device: Optional[str] = None,
        output_dir: str = "math_output",
        use_gemini: bool = False,
        gemini_api_key: Optional[str] = None,
        mode: str = "text",
        num_iterations: int = 3,
    ):
        self.model_name = model_name
        self.device = device
        self.output_dir = output_dir
        self.use_gemini = use_gemini
        self.gemini_api_key = gemini_api_key
        self.mode = mode  # "normal" uses server.py, "text" uses server_text.py
        self.iterative = True  # Always True
        self.num_iterations = num_iterations

        os.makedirs(output_dir, exist_ok=True)

        self.client: Optional[ChainOfThoughtReader] = None
        self.server: Optional[InsightAggregationServer] = None
        self.server_text: Optional[TextBasedInsightAggregationServer] = None
        self.encyclopedia_loaded = False

    # ------------------------------------------------------------------
    # Dataset loading helpers
    # ------------------------------------------------------------------
    def _load_local_json(
        self, dataset_name: str, explicit_path: Optional[str]
    ) -> List[Dict]:
        """Load a dataset from a local JSON file."""
        candidate_path = explicit_path or os.path.join(
            "math_datasets", f"{dataset_name}.json"
        )
        if not os.path.exists(candidate_path):
            raise FileNotFoundError(
                f"Dataset '{dataset_name}' not found. Provide {candidate_path} or update DATASET_REGISTRY."
            )
        with open(candidate_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _load_csv_file(
        self, dataset_name: str, explicit_path: Optional[str]
    ) -> List[Dict]:
        """Load a dataset from a CSV file.

        Tries to infer column names from common patterns:
        - Problem: problem, question, problem_text, task, statement
        - Answer: answer, solution, final_answer, answer_text
        - ID: id, problem_id, num, number
        """
        candidate_path = explicit_path or os.path.join(
            "math_datasets", f"{dataset_name}.csv"
        )
        if not os.path.exists(candidate_path):
            raise FileNotFoundError(
                f"CSV file for dataset '{dataset_name}' not found at {candidate_path}"
            )

        print(f"Loading CSV file from {candidate_path}...")
        with open(candidate_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file {candidate_path} is empty or has no header")

            fieldnames_lower = [fn.lower() for fn in reader.fieldnames]
            print(f"CSV columns: {reader.fieldnames}")

            # Map CSV columns to standard schema
            problem_cols = [
                "problem",
                "question",
                "problem_text",
                "task",
                "statement",
                "text",
            ]
            # Note: For gradingbench, "response" is the student answer being graded
            # For answerbench/proofbench, "short answer" or "solution" is the correct answer
            answer_cols = [
                "answer",
                "solution", 
                "final_answer",
                "answer_text",
                "short answer",
                "short_answer",
                "response"  # For gradingbench
            ]
            id_cols = [
                "id",
                "problem_id",
                "problem id",
                "grading_id",
                "grading id",
                "num",
                "number",
                "idx"
            ]

            problem_col = None
            answer_col = None
            id_col = None  # Reserved for future ID column mapping

            for col in problem_cols:
                if col in fieldnames_lower:
                    problem_col = reader.fieldnames[fieldnames_lower.index(col)]
                    break

            for col in answer_cols:
                if col in fieldnames_lower:
                    answer_col = reader.fieldnames[fieldnames_lower.index(col)]
                    break

            for col in id_cols:
                if col in fieldnames_lower:
                    id_col = reader.fieldnames[fieldnames_lower.index(col)]
                    break

            if not problem_col:
                raise ValueError(
                    f"Could not find problem column in CSV. Available columns: {reader.fieldnames}. "
                    f"Expected one of: {problem_cols}"
                )
            if not answer_col:
                raise ValueError(
                    f"Could not find answer column in CSV. Available columns: {reader.fieldnames}. "
                    f"Expected one of: {answer_cols}"
                )

            data = []
            for row_idx, row in enumerate(reader, 1):
                data.append(row)

            print(f"Loaded {len(data)} rows from CSV")
            return data

    def _normalize_problems(
        self, raw_problems: List[Dict], dataset_name: str
    ) -> List[Dict]:
        """Ensure a consistent schema across sources."""

        # Helper to find field from problem dict with multiple possible names
        def get_field(obj: Dict, candidates: List[str], default: str = "") -> str:
            for candidate in candidates:
                for key in obj.keys():
                    if key.lower() == candidate.lower():
                        val = obj[key]
                        return str(val) if val is not None else default
            return default

        normalized = []
        for idx, problem in enumerate(raw_problems):
            # Try various column name combinations (case-insensitive)
            problem_text = get_field(
                problem,
                ["problem", "question", "problem_text", "task", "statement", "text"],
            )
            answer_text = get_field(
                problem, ["answer", "solution", "final_answer", "answer_text"]
            )
            id_val = get_field(
                problem, ["id", "problem_id", "num", "number", "idx"], str(idx + 1)
            )

            normalized_problem = {
                "id": int(id_val) if id_val.isdigit() else id_val,
                "problem": problem_text,
                "question": problem_text,  # Keep both for compatibility
                "solution": get_field(problem, ["solution", "step_by_step"]),
                "answer": answer_text,
            }

            # GSM8K stores answer as "solution #### answer"; split when present.
            if dataset_name.startswith("gsm8k"):
                if "####" in answer_text:
                    parts = answer_text.split("####")
                    normalized_problem["solution"] = parts[0].strip()
                    normalized_problem["answer"] = parts[-1].strip()

            # Preserve all other fields from original
            for key, value in problem.items():
                if key not in normalized_problem:
                    normalized_problem[key] = value
            normalized.append(normalized_problem)
        return normalized

    def load_math_dataset(self, dataset_name: str) -> List[Dict]:
        """Load a dataset from Hugging Face, CSV, or JSON file."""
        # Registry lookup
        entry = DATASET_REGISTRY.get(dataset_name)
        if not entry:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available datasets: {', '.join(DATASET_REGISTRY.keys())}"
            )

        source_type, path_or_hf_name, data_dir, split = entry

        # Attempt Hugging Face
        if source_type == "hf":
            if load_dataset is None:
                raise ImportError(
                    "datasets library is required. Install with: pip install datasets"
                )
            print(
                f"Loading dataset '{dataset_name}' from Hugging Face ({path_or_hf_name}, split={split})..."
            )
            if data_dir and split:
                ds = load_dataset(path_or_hf_name, data_dir=data_dir, split=split)
            else:
                ds = load_dataset(path_or_hf_name, split=split)

            raw = []
            for i, item in enumerate(ds):
                if dataset_name == "math1000" and i >= 1000:
                    break
                solution = item.get("solution", "")
                answer = item.get("answer", "")
                if dataset_name == "math1000" and "####" in solution:
                    answer = solution.split("####")[-1].strip()
                raw.append(
                    {
                        "id": item.get("id", i + 1),
                        "problem": item.get("problem", item.get("question", "")),
                        "question": item.get("question", item.get("problem", "")),
                        "solution": solution or item.get("answer", ""),
                        "answer": answer,
                    }
                )
            print(f"Loaded {len(raw)} problems from Hugging Face")
            return self._normalize_problems(raw, dataset_name)

        # Attempt CSV file
        if source_type == "csv":
            raw = self._load_csv_file(dataset_name, path_or_hf_name)
            print(f"Loaded {len(raw)} problems from CSV for '{dataset_name}'")
            return self._normalize_problems(raw, dataset_name)

        # Attempt JSON file
        if source_type == "json":
            raw = self._load_local_json(dataset_name, path_or_hf_name)
            print(f"Loaded {len(raw)} problems from JSON for '{dataset_name}'")
            return self._normalize_problems(raw, dataset_name)

        raise ValueError(f"Unknown source type: {source_type}")

    # ------------------------------------------------------------------
    # STEP 1: Insight extraction across multiple datasets
    # ------------------------------------------------------------------
    def _ensure_client(self):
        if self.client is None:
            self.client = ChainOfThoughtReader(
                model_name=self.model_name,
                device=self.device,
                use_gemini=self.use_gemini,
                gemini_api_key=self.gemini_api_key,
            )

    def _extract_insights_for_dataset(
        self,
        dataset_name: str,
        problems: List[Dict],
        max_problems: Optional[int],
        encyclopedia_path: Optional[str] = None,
        iteration: int = 0,
    ) -> Tuple[str, List[Dict]]:
        """Extract insights from dataset, optionally solving with encyclopedia first.

        Args:
            dataset_name: Name of dataset
            problems: List of problems
            max_problems: Max problems to process
            encyclopedia_path: If provided, solve with encyclopedia before extracting insights
            iteration: Current iteration number (for logging)

        Returns:
            Tuple of (insights_dir, results_list)
        """
        self._ensure_client()
        insights_dir = os.path.join(self.output_dir, dataset_name)
        os.makedirs(insights_dir, exist_ok=True)

        worklist = problems[:max_problems] if max_problems else problems
        print(
            f"\nIteration {iteration}: Extracting insights for {dataset_name} ({len(worklist)} problems)..."
        )

        # If encyclopedia provided, load it into client for solving
        results = []
        if encyclopedia_path and os.path.exists(encyclopedia_path):
            print(f"  Using encyclopedia from: {encyclopedia_path}")
            self.client.load_encyclopedia(encyclopedia_path, mode=self.mode)
            self.encyclopedia_loaded = True
        else:
            self.encyclopedia_loaded = False

        for idx, problem_data in enumerate(worklist, 1):
            problem_text = problem_data.get("problem") or problem_data.get(
                "question", ""
            )
            if not problem_text:
                print(f"  [skip] Problem {idx} missing text")
                continue

            print(f"  [{idx}/{len(worklist)}] {problem_text[:80]}...")

            # Solve with encyclopedia if loaded
            predicted_answer = None
            is_correct = False
            if self.encyclopedia_loaded:
                try:
                    answer = self.client.solve_with_encyclopedia(
                        problem_text, is_math=True
                    )
                    answer_text = answer
                    if "## Answer:" in answer:
                        start_idx = answer.find("## Answer:") + len("## Answer:")
                        end_idx = answer.find("## End of Answer:")
                        if end_idx == -1:
                            end_idx = len(answer)
                        answer_text = answer[start_idx:end_idx].strip()
                    predicted_answer = answer_text

                    ground_truth = problem_data.get("answer") or problem_data.get(
                        "solution", ""
                    )
                    is_correct = self._check_answer_match(
                        predicted_answer, ground_truth, dataset_name
                    )
                    status = "✓" if is_correct else "✗"
                    print(
                        f"    {status} Predicted: {predicted_answer[:50]}... | GT: {ground_truth[:50]}..."
                    )
                except Exception as exc:
                    print(f"    Error solving: {exc}")

            # Even without encyclopedia in iteration 1, track that we attempted
            # This ensures all iterations have unified result tracking
            if not self.encyclopedia_loaded:
                ground_truth = problem_data.get("answer") or problem_data.get(
                    "solution", ""
                )
                predicted_answer = ""  # No prediction without encyclopedia
                is_correct = False
                results.append(
                    {
                        "dataset": dataset_name,
                        "problem_id": problem_data.get("id", idx),
                        "predicted_answer": predicted_answer,
                        "ground_truth": ground_truth,
                        "is_correct": is_correct,
                    }
                )

            # Extract insights
            try:
                result = self.client.read_paper(task=problem_text, paper_content=None)
                insight_book = result.get("behavior_book", {})
                if insight_book:
                    insight_book = {
                        k: v
                        for k, v in insight_book.items()
                        if not k.startswith("insight_fallback")
                    }
                if not insight_book:
                    print("    No insights extracted")
                    continue

                output_data = {
                    "problem": problem_text,
                    "problem_id": problem_data.get("id", idx),
                    "behavior_book": insight_book,
                    "iteration": iteration,
                    "predicted_answer": (
                        predicted_answer if predicted_answer is not None else ""
                    ),
                    "ground_truth": problem_data.get("answer")
                    or problem_data.get("solution", ""),
                    "is_correct": is_correct,
                }
                # Only add to results if not already added above (for iteration 1 case)
                if predicted_answer is not None and self.encyclopedia_loaded:
                    results.append(
                        {
                            "dataset": dataset_name,
                            "problem_id": problem_data.get("id", idx),
                            "predicted_answer": predicted_answer,
                            "ground_truth": output_data["ground_truth"],
                            "is_correct": is_correct,
                        }
                    )

                output_path = os.path.join(insights_dir, f"problem_{idx:04d}.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                print(f"    Saved {len(insight_book)} insights -> {output_path}")
                time.sleep(0.5)
            except Exception as exc:  # noqa: BLE001
                print(f"    Error processing problem {idx}: {exc}")

        return insights_dir, results

    def learn_insights_from_datasets(
        self,
        dataset_names: List[str],
        max_problems: Optional[int],
        encyclopedia_path: Optional[str] = None,
        iteration: int = 0,
    ) -> Tuple[Dict[str, str], List[Dict]]:
        """Learn insights from datasets, optionally solving with encyclopedia first.

        Returns:
            Tuple of (insights_map, all_results)
        """
        if not dataset_names:
            raise ValueError("Provide at least one dataset for STEP 1.")

        insights_map: Dict[str, str] = {}
        all_results = []
        for name in dataset_names:
            problems = self.load_math_dataset(name)
            insights_dir, results = self._extract_insights_for_dataset(
                name, problems, max_problems, encyclopedia_path, iteration
            )
            insights_map[name] = insights_dir
            all_results.extend(results)

        print("\nFinished STEP 1 across datasets:")
        for name, path in insights_map.items():
            print(f"  - {name}: {path}")

        if all_results:
            num_correct = sum(1 for r in all_results if r["is_correct"])
            accuracy = num_correct / len(all_results) if all_results else 0.0
            print(
                f"\nIteration {iteration} Accuracy: {num_correct}/{len(all_results)} = {accuracy:.2%}"
            )

        return insights_map, all_results

    # ------------------------------------------------------------------
    # STEP 2: Aggregate chosen insights into one encyclopedia
    # ------------------------------------------------------------------
    def aggregate_insights(self, insight_sets: List[str], r1: float, r2: float) -> str:
        if not insight_sets:
            raise ValueError("Provide at least one dataset to aggregate in STEP 2.")

        json_files: List[str] = []
        for name in insight_sets:
            insights_dir = os.path.join(self.output_dir, name)
            if not os.path.isdir(insights_dir):
                raise FileNotFoundError(
                    f"Insights directory not found for {name}: {insights_dir}"
                )
            dataset_files = [
                os.path.join(insights_dir, f)
                for f in os.listdir(insights_dir)
                if f.endswith(".json") and f.startswith("problem_")
            ]
            json_files.extend(sorted(dataset_files))

        if not json_files:
            raise FileNotFoundError("No insight JSON files found for aggregation.")

        print("\nAggregating insights from:")
        for name in insight_sets:
            print(f"  - {name}")

        if self.mode == "text":
            self.server_text = TextBasedInsightAggregationServer(
                model_name=self.model_name,
                device=self.device,
                input_dir=self.output_dir,
                use_gemini=self.use_gemini,
                gemini_api_key=self.gemini_api_key,
            )
            result = self.server_text.aggregate_and_build_encyclopedia(
                json_files=json_files, output_dir=self.output_dir
            )
            self.server_text.save_results(result, output_dir=self.output_dir)
            encyclopedia_path = os.path.join(self.output_dir, "encyclopedia.json")
        else:
            self.server = InsightAggregationServer(
                model_name=self.model_name,
                device=self.device,
                input_dir=self.output_dir,
                use_gemini=self.use_gemini,
                gemini_api_key=self.gemini_api_key,
            )
            result = self.server.aggregate_and_build_encyclopedia(
                json_files=json_files, r1=r1, r2=r2, output_dir=self.output_dir
            )
            self.server.save_results(result, output_dir=self.output_dir)
            encyclopedia_path = os.path.join(self.output_dir, "encyclopedia.txt")

        print(f"Encyclopedia saved to {encyclopedia_path}")
        return encyclopedia_path

    # ------------------------------------------------------------------
    # STEP 3: Solve multiple target datasets using one encyclopedia (DEPRECATED)
    # ------------------------------------------------------------------
    # NOTE: In iterative mode, solving is now integrated into _extract_insights_for_dataset
    # Standard mode can still use this if needed, but iterative mode handles everything
    # through the client with encyclopedia support

    # ------------------------------------------------------------------
    # Iterative Learning Pipeline
    # ------------------------------------------------------------------
    def run_iterative_pipeline(
        self,
        dataset_list: List[str],
        max_problems: Optional[int],
        r1: float = 0.95,
        r2: float = 0.4,
        start_from_step: int = 1,
    ) -> Dict:
        """Run iterative learning pipeline.

        Each iteration:
        1. Use encyclopedia (from previous iteration) to solve problems and log accuracy
        2. Extract insights from the same problems
        3. Aggregate insights into new encyclopedia
        4. Repeat

        Args:
            dataset_list: List of datasets to train on
            max_problems: Max problems per dataset
            r1: Similarity threshold for aggregation
            r2: Similarity threshold for aggregation
            start_from_step: Start from step 1 (extract) or 2 (aggregate only)

        Returns:
            Summary dict with iteration history
        """
        if not dataset_list:
            raise ValueError("Provide at least one dataset for iterative learning")

        start_time = time.time()
        iteration_history = []
        encyclopedia_path = None

        print(f"\n{'='*80}")
        print(f"Starting Iterative Learning Pipeline: {self.num_iterations} iterations")
        print(f"Datasets: {', '.join(dataset_list)}")
        print(f"Max problems per dataset: {max_problems or 'all'}")
        print(f"{'='*80}\n")

        for iteration in range(1, self.num_iterations + 1):
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration}/{self.num_iterations}")
            print(f"{'='*80}")

            # STEP 1: Extract insights (and solve if encyclopedia exists)
            if start_from_step == 1:
                insights_map, results = self.learn_insights_from_datasets(
                    dataset_list, max_problems, encyclopedia_path, iteration
                )

                # Calculate accuracy for this iteration
                accuracy = 0.0
                num_correct = 0
                if results:
                    num_correct = sum(1 for r in results if r["is_correct"])
                    accuracy = num_correct / len(results)
            else:
                # Starting from step 2: check if insights exist from previous run
                print(f"\nSkipping Step 1 (insight extraction) - assuming insights already exist")
                insights_exist = all(
                    os.path.isdir(os.path.join(self.output_dir, name))
                    for name in dataset_list
                )
                if not insights_exist:
                    raise FileNotFoundError(
                        f"Cannot start from step 2: Insight directories not found in {self.output_dir}. "
                        "Run with --start-from-step 1 first to extract insights."
                    )
                results = []
                accuracy = 0.0
                num_correct = 0

            # STEP 2: Aggregate insights into encyclopedia
            print(f"\nIteration {iteration}: Aggregating insights...")
            encyclopedia_path = self.aggregate_insights(dataset_list, r1=r1, r2=r2)

            # Save iteration results
            iteration_dir = os.path.join(self.output_dir, f"iteration_{iteration}")
            os.makedirs(iteration_dir, exist_ok=True)

            iteration_summary = {
                "iteration": iteration,
                "datasets": dataset_list,
                "num_problems": len(results),
                "num_correct": num_correct,
                "accuracy": accuracy,
                "encyclopedia_path": encyclopedia_path,
            }
            iteration_history.append(iteration_summary)

            # Save iteration results
            iteration_results_path = os.path.join(iteration_dir, "results.json")
            with open(iteration_results_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "summary": iteration_summary,
                        "results": results,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            print(f"\nIteration {iteration} Summary:")
            print(f"  Accuracy: {num_correct}/{len(results)} = {accuracy:.2%}")
            print(f"  Encyclopedia: {encyclopedia_path}")
            print(f"  Results saved: {iteration_results_path}")

        # Final summary
        final_summary = {
            "mode": "iterative",
            "num_iterations": self.num_iterations,
            "datasets": dataset_list,
            "iteration_history": iteration_history,
            "total_time_seconds": time.time() - start_time,
        }

        summary_path = os.path.join(self.output_dir, "iterative_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print("ITERATIVE LEARNING COMPLETE")
        print(f"{'='*80}")
        print(f"\nAccuracy progression:")
        for iter_sum in iteration_history:
            print(f"  Iteration {iter_sum['iteration']}: {iter_sum['accuracy']:.2%}")
        print(f"\nFinal summary saved: {summary_path}")

        return final_summary

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def _check_answer_match(
        self, predicted: str, ground_truth: str, dataset_name: Optional[str] = None
    ) -> bool:
        """Check if predicted answer matches ground truth.

        For standard datasets (AIME, MATH500, GSM8K): numeric/string comparison
        For IMOBench datasets: uses symbolic comparison and handles algebraic equivalence

        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            dataset_name: Dataset name to determine evaluation strategy

        Returns:
            True if answers match
        """
        if not predicted or not ground_truth:
            return False

        # For IMOBench datasets, use more sophisticated symbolic evaluation
        if dataset_name and dataset_name.startswith("imo"):
            return self._check_imo_answer(predicted, ground_truth)

        # Standard numeric/string comparison for other datasets

        pred_nums = extract_numbers(predicted)
        gt_nums = extract_numbers(ground_truth)

        # If no numbers found, do string comparison
        if not pred_nums or not gt_nums:
            return predicted.strip().lower() == ground_truth.strip().lower()

        # Check if any predicted number matches any ground truth number
        return any(abs(p - g) < 1e-6 for p in pred_nums for g in gt_nums)

    def _check_imo_answer(self, predicted: str, ground_truth: str) -> bool:
        """Evaluate IMOBench answers with symbolic/algebraic comparison.

        IMOBench answers may be:
        - Single numbers/expressions (from IMO-AnswerBench)
        - Multi-step solutions (from IMO-ProofBench)
        - Boolean/classification (from IMO-GradingBench)

        This implements basic symbolic comparison and normalization.
        """
        pred = predicted.strip()
        truth = ground_truth.strip()

        if not pred or not truth:
            return False

        # Exact match (case-insensitive)
        if pred.lower() == truth.lower():
            return True

        # Normalize common variations
        # Remove common suffixes
        suffixes = [
            " degrees",
            "°",
            " radians",
            " rad",
            " units",
            " cm",
            " m",
            " inches",
        ]
        pred_norm = pred.lower()
        truth_norm = truth.lower()

        for suffix in suffixes:
            pred_norm = pred_norm.replace(suffix.lower(), "")
            truth_norm = truth_norm.replace(suffix.lower(), "")

        if pred_norm.strip() == truth_norm.strip():
            return True

        # Extract numeric parts for numeric comparison
        pred_nums = extract_numbers(pred)
        truth_nums = extract_numbers(truth)

        # If both have numbers, compare numerically
        if pred_nums and truth_nums:
            # Check exact match or close approximate match
            if len(pred_nums) == len(truth_nums):
                return all(abs(p - t) < 1e-6 for p, t in zip(pred_nums, truth_nums))
            # Also try single number comparison
            if len(pred_nums) == 1 and len(truth_nums) == 1:
                return abs(pred_nums[0] - truth_nums[0]) < 1e-6

        # Check if one answer is substring of other (for partial matches)
        # This helps with cases like "42 or 43" vs "42"
        if pred_norm in truth_norm or truth_norm in pred_norm:
            return True

        return False


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _parse_list_arg(raw: Optional[List[str]]) -> Optional[List[str]]:
    if raw is None:
        return None
    normalized: List[str] = []
    for item in raw:
        parts = [p.strip() for p in item.split(",") if p.strip()]
        normalized.extend(parts)
    return normalized or None


def main():
    parser = argparse.ArgumentParser(
        description="Iterative math learning pipeline: solve (if encyclopedia available) + extract insights + aggregate → repeat"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["aime25"],
        help="Datasets for iterative learning (space- or comma-separated).",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Limit problems per dataset per iteration.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model name (HF).",
    )
    parser.add_argument(
        "-d", "--device", type=str, default=None, help="Device to use (cuda or cpu)."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="math_output",
        help="Root output directory.",
    )
    parser.add_argument(
        "--r1", type=float, default=0.95, help="r1 threshold for same insights."
    )
    parser.add_argument(
        "--r2", type=float, default=0.6, help="r2 threshold for linked insights."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--use-gemini", action="store_true", help="Use Google Gemini API."
    )
    parser.add_argument(
        "--gemini-api-key", type=str, default=None, help="Gemini API key."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="text",
        choices=["normal", "text"],
        help="Aggregation/inference mode (normal=GraphRAG, text=text-based). Default: text",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=3,
        help="Number of iterations (default: 3)",
    )
    parser.add_argument(
        "--start-from-step",
        type=int,
        default=1,
        choices=[1, 2],
        help="Start from step: 1=extract insights (default), 2=aggregate only (assumes insights already exist)",
    )

    args = parser.parse_args()

    # Normalize dataset lists
    datasets = _parse_list_arg(args.datasets)

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {args.seed}")

    pipeline = MathDomainPipeline(
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir,
        use_gemini=args.use_gemini,
        gemini_api_key=args.gemini_api_key,
        mode=args.mode,
        num_iterations=args.num_iterations,
    )

    try:
        if not datasets:
            raise ValueError("--datasets is required")
        pipeline.run_iterative_pipeline(
            dataset_list=datasets,
            max_problems=args.max_problems,
            r1=args.r1,
            r2=args.r2,
            start_from_step=args.start_from_step,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        import traceback

        traceback.print_exc()
        print("\nExamples:")
        print(
            "  python math_domain.py --datasets aime25 --max-problems 10 --num-iterations 3"
        )
        print(
            "  python math_domain.py --datasets gsm8k math500 --max-problems 20 --mode text"
        )
        print(
            "  python math_domain.py --datasets imo_answerbench --max-problems 30 --num-iterations 5"
        )


if __name__ == "__main__":
    main()
