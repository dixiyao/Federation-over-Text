"""
Behavior Curation Pipeline Client
Implements metacognitive reuse for LLM reasoning based on:
"Metacognitive Reuse: Turning Recurring LLM Reasoning Into Concise Behaviors"
Uses a three-stage pipeline: Solution → Reflection → Behavior Extraction
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import PyPDF2
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChainOfThoughtReader:
    """
    A client that reads papers using behavior curation pipeline based on
    "Metacognitive Reuse: Turning Recurring LLM Reasoning Into Concise Behaviors"
    Implements three-stage pipeline: Solution → Reflection → Behavior Extraction
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1",
        task: Optional[str] = None,
        device: Optional[str] = None,
        papers_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.papers_dir = papers_dir or "data/papers/iclr23_top5"
        self.reasoning_steps = []
        self.task = (
            task
            or "Analyze this paper and identify key contributions, limitations, and potential future research directions."
        )
        self.behavior_book = {}  # Store extracted behaviors
        
        # Model and tokenizer will be loaded lazily on first use
        self.model = None
        self.tokenizer = None
        self.device = device or ("cuda" if self._check_cuda() else "cpu")
        
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _load_model(self):
        """Lazy load the Hugging Face model and tokenizer"""
        if self.model is not None and self.tokenizer is not None:
            return
        
        try:
            print(f"Loading model: {self.model_name}")
            print(f"Device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            
            # Load model
            # Use torch_dtype instead of dtype for from_pretrained
            model_kwargs = {
                "trust_remote_code": True,
            }

            if self.device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **model_kwargs
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("Model loaded successfully!")
            
        except ImportError:
            raise ImportError(
                "transformers and torch are required. Install with: pip install transformers torch"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file"""
        try:
            text = ""
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except ImportError:
            print("PyPDF2 not installed. Installing...")
            os.system("pip install PyPDF2")
            return self._extract_text_from_pdf(pdf_path)
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""

    def _get_pdf_files(self, directory: str, max_papers: Optional[int] = None) -> list:
        """Get list of PDF files from a directory"""
        pdf_files = []
        dir_path = Path(directory)

        if not dir_path.exists():
            print(f"Warning: Directory {directory} does not exist")
            return pdf_files

        # Find all PDF files
        for pdf_file in dir_path.glob("*.pdf"):
            pdf_files.append(str(pdf_file))

        # Sort for consistent ordering
        pdf_files.sort()

        # Limit number of papers if specified
        if max_papers is not None and max_papers > 0:
            pdf_files = pdf_files[:max_papers]

        return pdf_files

    def _call_model(self, prompt: str, system_prompt: Optional[str] = None, max_new_tokens: Optional[int] = None) -> str:
        """
        Call the Hugging Face language model.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt (will be prepended if provided)
                          NOTE: For DeepSeek-R1 models, avoid system prompts - put all in user prompt
            max_new_tokens: Maximum number of new tokens to generate. If None, uses default 32768.
        
        Returns:
            Generated text response
        """
        # Load model if not already loaded
        self._load_model()
        
        # For DeepSeek-R1: Avoid system prompts, put all instructions in user prompt
        # If system_prompt is provided, combine it into the user prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        try:
            
            # Tokenize input
            # For papers with ~80k characters, we need ~20-30k tokens
            # DeepSeek-R1 supports large context windows (64k+ tokens)
            # Use 65536 (64k) to handle very large papers
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=65536,  # 64k tokens - enough for papers up to ~200k characters
            ).to(self.device)

            # Get input token count
            input_token_count = inputs["input_ids"].shape[1]

            # Use provided max_new_tokens or default to 32768
            if max_new_tokens is None:
                max_new_tokens = 32768  # Default: 32k tokens

            print(
                f"Input tokens: {input_token_count}, Max new tokens: {max_new_tokens}"
            )
            
            # Generate response
            with torch.no_grad():
                # DeepSeek-R1 recommendations: temperature 0.5-0.7 (0.6 recommended)
                # Check if model name contains "DeepSeek-R1" to use recommended settings
                is_deepseek_r1 = "DeepSeek-R1" in self.model_name
                
                if is_deepseek_r1:
                    # DeepSeek-R1 recommended settings
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        top_p=0.9,
                        temperature=0.7,
                        repetition_penalty=1.1,
                        use_cache=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                else:
                    # Default settings for other models
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.1,  # Penalize repetition to avoid loops
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
            
            # Decode response
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"Error calling model: {e}")
            return f"[Error] Model generation failed: {str(e)}"

    def _get_solution_prompt(self, problem: str) -> str:
        """Get the Solution Prompt as defined in the paper"""
        prompt = f"""Please reason step by step and put the final answer in <answer_box>.

Problem: {problem}"""
        return prompt

    def _get_reflection_prompt(self, problem: str, solution: str) -> str:
        """Get the Reflection Prompt as defined in the paper"""
        prompt = f"""
You are given:

Problem:
{problem}

Proposed Solution:
{solution}

Definition:
A **“skill”** is a general, reusable method/strategy/technique — not a one-off answer. A skill should help solve this problem and many analogous problems.

Your task: Produce a structured, thorough critique of the proposed solution. Your output should have **three clearly labeled sections**:  
---

### I. Correctness Analysis  
- Verify whether the solution is correct under established scientific / mathematical / physical principles.  
- Check every calculation and algebraic step for errors.  
- Evaluate the logical soundness and justification of each step.  
- Assess if the reasoning is generalizable (or is overly ad-hoc to this instance).  
- Identify any conceptual flaws, misuse of assumptions, or violations of domain context or constraints.  
- If relevant: assess whether the solution respects ethical, contextual, or domain-specific constraints.  

### II. Missing Skills / Techniques Analysis  
- Identify any generalizable skills or techniques (per above definition) that the author could have used but did not.  
- For each missing skill:  
    * Describe what the skill is.  
    * Explain **how** using it could have simplified the solution (e.g., reduced length, improved clarity).  
    * Explain **how** it could have prevented errors or improved robustness.  
    * Explain **why** that skill is generally useful for this kind of problem or class of problems.  
    * Optionally, show how a solution using that skill might look (concise sketch).  

### III. New Skill Suggestions for Future Problems  
- Propose **new, reusable skills** (techniques/approaches/heuristics) that would help with this and similar problems.  
- For each proposed skill:  
    * Name it with prefix `skill_`.  
    * Provide a clear, actionable description of when and how to apply it.  
    * Explain **why** this skill is valuable and in what kinds of problems or contexts it helps.   
"""
        return prompt

    def _get_behavior_prompt(self, problem: str, solution: str, reflection: str) -> str:
        """Get the Behavior Prompt as defined in the paper"""
        # Use string replacement to avoid f-string issues with curly braces in reflection
        prompt_template = """
You are given:

- Problem: {problem}  
- Proposed Solution: {solution}  
- Reflection / Critique: {reflection}

Your task:  
Generate a JSON object whose keys are skill names and whose values are skill descriptions. Each skill name **must** begin with `skill_`. Each skill should be a single line, and the format is "skill_[name]: [description]". Each description **must** be a **single line** (no newline characters). Do **not** invent skills outside of what can be inferred from the problem, solution, and reflection. Do **not** include any commentary or extra text — output **only** the JSON object.

Example format:
{{
  "skill_exampleName": "Description of the skill.",
  "skill_anotherOne": "Another skill description."
                     }}
                """
        prompt = prompt_template.format(
            problem=problem, solution=solution, reflection=reflection
        )
        return prompt

    def _step_solution(self, problem: str) -> Dict:
        """Step 1: Generate solution using Solution Prompt"""
        prompt = self._get_solution_prompt(problem)

        system_prompt = None
        # Step 1: Use 2048 tokens for solution generation (sufficient for math problems)
        response = self._call_model(prompt, system_prompt, max_new_tokens=2048)
        print(f"Solution Response: {response}")

        step_result = {
            "step": 1,
            "name": "Solution Generation",
            "prompt": prompt,
            "response": response,
            "timestamp": time.time(),
        }

        self.reasoning_steps.append(step_result)
        return step_result

    def _step_reflection(self, problem: str, solution: str) -> Dict:
        """Step 2: Generate reflection using Reflection Prompt"""
        prompt = self._get_reflection_prompt(problem, solution)

        system_prompt = None
        # Step 2: Use 32768 tokens for reflection (needs more tokens for detailed critique)
        response = self._call_model(prompt, system_prompt, max_new_tokens=32768)
        print(f"Reflection Response: {response}")

        step_result = {
            "step": 2,
            "name": "Reflection and Critique",
            "prompt": prompt,
            "response": response,
            "timestamp": time.time(),
        }

        self.reasoning_steps.append(step_result)
        return step_result

    def _step_behavior_extraction(
        self, problem: str, solution: str, reflection: str
    ) -> Dict:
        """Step 3: Extract behaviors using Behavior Prompt"""
        prompt = self._get_behavior_prompt(problem, solution, reflection)

        system_prompt = None
        # Step 3: Use 32768 tokens for behavior extraction (needs more tokens for comprehensive skill extraction)
        response = self._call_model(prompt, system_prompt, max_new_tokens=32768)
        print(f"Behavior Extraction Response: {response}")

        # Try to parse behaviors from JSON response
        behaviors = {}
        formatted_json_array = None  # Store the original formatted JSON array

        try:
            # First, try to extract JSON from markdown code blocks (```json ... ```)
            json_code_block = re.search(
                r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", response, re.DOTALL
            )
            if json_code_block:
                json_str = json_code_block.group(1)
            else:
                # Try to find JSON array [...]
                json_array_match = re.search(r"\[[\s\S]*?\]", response)
                if json_array_match:
                    json_str = json_array_match.group(0)
                else:
                    # Try to find JSON object {...}
                    json_object_match = re.search(r"\{[\s\S]*?\}", response)
                    if json_object_match:
                        json_str = json_object_match.group(0)
                    else:
                        json_str = None

            if json_str:
                try:
                    json_data = json.loads(json_str)
                    # Store the original formatted JSON array if it's an array
                    if isinstance(json_data, list):
                        formatted_json_array = json_data
                        # Also convert to behavior_book format for internal use
                        for item in json_data:
                            if isinstance(item, dict):
                                # Handle format: {"behavior": "name", "description": "desc"}
                                if "skill" in item and "description" in item:
                                    skill_name = item["skill"]
                                    # Ensure skill name starts with "skill_"
                                    if not skill_name.startswith("skill_"):
                                        skill_name = f"skill_{skill_name}"
                                    behaviors[skill_name] = item["description"]
                                else:
                                    behaviors.update(item)
                    elif isinstance(json_data, dict):
                        behaviors = json_data

                    if behaviors:
                        self.behavior_book.update(behaviors)
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error: {e}")

            # If still no behaviors, try manual extraction
            if not behaviors:
                print(
                    "Warning: Could not parse JSON from behavior extraction. Attempting manual extraction."
                )
                # Try to extract behaviors manually (look for behavior_* patterns)
                behavior_pattern = (
                    r'["\']?(behavior_\w+)["\']?\s*[:=]\s*["\']?([^"\']+)["\']?'
                )
                matches = re.findall(behavior_pattern, response)
                for name, desc in matches:
                    behaviors[name] = desc.strip()
                if behaviors:
                    self.behavior_book.update(behaviors)

        except Exception as e:
            print(f"Warning: Error parsing behaviors: {e}. Storing raw response.")
            if not behaviors:
                behaviors = {"raw_response": response}

        step_result = {
            "step": 3,
            "name": "Behavior Extraction",
            "prompt": prompt,
            "response": response,
            "behaviors": behaviors,
            "formatted_json_array": formatted_json_array,  # Store the original formatted JSON array
            "timestamp": time.time(),
        }

        self.reasoning_steps.append(step_result)
        return step_result

    def read_paper(
        self, task: Optional[str] = None, paper_content: Optional[str] = None
    ) -> Dict:
        """
        Main method to process a question/problem using behavior curation pipeline.
        Implements the three-stage pipeline: Solution → Reflection → Behavior Extraction

        Args:
            task: Optional task/question/problem to solve. If None, uses the default task.
            paper_content: Optional paper content to combine with the task/question.

        Returns:
            Dictionary containing all reasoning steps, behaviors, and behavior book.
        """
        # Update task if provided
        if task is not None:
            self.task = task

        # Combine task/question with paper content
        if paper_content:
            problem = f"""User Question: {self.task}

Paper Content:
{paper_content}

Please answer the user's question based on the paper content provided above."""
        else:
            # The task/question is treated as the problem to solve
            problem = self.task

        print(f"Problem/Question: {self.task}\n")
        if paper_content:
            print(f"Paper content length: {len(paper_content)} characters\n")

        # Reset reasoning steps and behavior book
        self.reasoning_steps = []
        self.behavior_book = {}

        # Step 1: Solution Generation
        print("Step 1: Generating solution...")
        step1 = self._step_solution(problem)
        solution = step1["response"]
        time.sleep(1)  # Rate limiting

        # Step 2: Reflection
        print("Step 2: Generating reflection and critique...")
        step2 = self._step_reflection(problem, solution)
        reflection = step2["response"]
        time.sleep(1)  # Rate limiting

        # Step 3: Behavior Extraction
        print("Step 3: Extracting behaviors...")
        step3 = self._step_behavior_extraction(problem, solution, reflection)
        time.sleep(1)

        # Compile results
        result = {
            "problem": problem,
            "task": self.task,
            # "reasoning_steps": self.reasoning_steps,
            # "solution": solution,
            # "reflection": reflection,
            # "behaviors": step3.get("behaviors", {}),
            "behavior_book": self.behavior_book,
            "total_steps": len(self.reasoning_steps),
            # "complete_reasoning": self._format_complete_reasoning(),
        }

        return result

    def process_multiple_papers(
        self,
        question: str,
        papers_dir: Optional[str] = None,
        num_papers: Optional[int] = None,
    ) -> list:
        """
        Process multiple papers with a given question sequentially.
        Each paper will generate its own behavior book.

        Args:
            question: The user-provided question to answer for each paper
            papers_dir: Directory containing PDF papers. If None, uses self.papers_dir
            num_papers: Number of papers to process. If None, processes all papers.

        Returns:
            List of results, one for each paper processed
        """
        # Use provided directory or default
        directory = papers_dir or self.papers_dir

        # Get list of PDF files
        pdf_files = self._get_pdf_files(directory, num_papers)

        if not pdf_files:
            print(f"No PDF files found in {directory}")
            return []

        print(f"Found {len(pdf_files)} PDF file(s) to process")
        print(f"Question: {question}\n")
        print("=" * 80)

        all_results = []
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        # Process papers sequentially
        for idx, pdf_path in enumerate(pdf_files, 1):
            paper_name = Path(pdf_path).stem
            print(f"\n{'='*80}")
            print(f"Processing Paper {idx}/{len(pdf_files)}: {paper_name}")
            print(f"{'='*80}\n")

            try:
                # Extract text from PDF
                print(f"Extracting text from {paper_name}...")
                paper_content = self._extract_text_from_pdf(pdf_path)

                if not paper_content:
                    print(
                        f"Warning: Could not extract text from {pdf_path}. Skipping..."
                    )
                    continue

                print(f"Extracted {len(paper_content)} characters from paper\n")

                # Process this paper with the question
                result = self.read_paper(task=question, paper_content=paper_content)

                # Add paper metadata to result
                result["paper_path"] = pdf_path
                result["paper_name"] = paper_name
                result["paper_index"] = idx

                # Get the formatted JSON array from the behavior extraction step
                formatted_json = None
                for step in result.get("reasoning_steps", []):
                    if step.get("step") == 3 and "formatted_json_array" in step:
                        formatted_json = step.get("formatted_json_array")
                        break

                # If no formatted JSON array found, fall back to behavior_book format
                if formatted_json is None:
                    formatted_json = [
                        {"behavior": name.replace("behavior_", ""), "description": desc}
                        for name, desc in result.get("behavior_book", {}).items()
                    ]

                # Save JSON
                output_data = {
                    "file_number": idx,
                    "paper_name": paper_name,
                    "behaviors": formatted_json,
                }

                output_path = output_dir / f"paper_{idx:02d}.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)

                print(f"Saved behavior book to: {output_path}")
                print(f"\nCompleted paper {idx}/{len(pdf_files)}: {paper_name}")
                print(f"Behaviors extracted: {len(result.get('behavior_book', {}))}")
                print("-" * 80)

                all_results.append(result)

            except Exception as e:
                print(f"Error processing {paper_name}: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Save summary
        summary = {
            "total_papers": len(all_results),
            "papers": [
                {
                    "paper_index": r.get("paper_index", 0),
                    "paper_name": r.get("paper_name", "Unknown"),
                    "paper_path": r.get("paper_path", ""),
                    "behavior_book": r.get("behavior_book", {}),
                }
                for r in all_results
            ],
            "total_behaviors": sum(
                len(r.get("behavior_book", {})) for r in all_results
            ),
        }

        summary_path = output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"Completed processing {len(all_results)}/{len(pdf_files)} papers")
        print(f"Total behaviors extracted: {summary['total_behaviors']}")
        print(f"Summary saved to: {summary_path}")
        print(f"{'='*80}")

        return all_results

    def _format_complete_reasoning(self) -> str:
        """Format all reasoning steps into a complete reasoning process"""
        formatted = []
        for step in self.reasoning_steps:
            formatted.append(f"\n{'='*80}")
            formatted.append(f"STEP {step['step']}: {step['name']}")
            formatted.append(f"{'='*80}\n")
            formatted.append(step["response"])
            formatted.append("\n")
        return "\n".join(formatted)

    def save_reasoning(self, reasoning_result: Dict, output_path: Optional[str] = None):
        """Save the complete reasoning process and behavior book to files"""
        if output_path is None:
            # Create a safe filename from the problem/question
            safe_name = re.sub(
                r"[^\w\s-]", "", reasoning_result.get("problem", "reasoning")[:50]
            )
            safe_name = re.sub(r"[-\s]+", "_", safe_name)
            output_path = f"reasoning_{safe_name}.txt"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Problem/Question: {reasoning_result.get('problem', 'N/A')}\n")
            f.write(f"\n{'='*80}\n")
            f.write("BEHAVIOR CURATION PIPELINE RESULTS\n")
            f.write(f"{'='*80}\n\n")
            f.write(reasoning_result["complete_reasoning"])
            f.write(f"\n\n{'='*80}\n")
            f.write("BEHAVIOR BOOK\n")
            f.write(f"{'='*80}\n\n")
            for behavior_name, behavior_desc in reasoning_result.get(
                "behavior_book", {}
            ).items():
                f.write(f"{behavior_name}: {behavior_desc}\n")

        # Also save as JSON for structured access
        json_path = output_path.replace(".txt", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(reasoning_result, f, indent=2, ensure_ascii=False)

        # Save behavior book separately as JSON
        behavior_book_path = output_path.replace(".txt", "_behavior_book.json")
        with open(behavior_book_path, "w", encoding="utf-8") as f:
            json.dump(
                reasoning_result.get("behavior_book", {}),
                f,
                indent=2,
                ensure_ascii=False,
            )

        print("\nResults saved to:")
        print(f"  - {output_path}")
        print(f"  - {json_path}")
        print(f"  - {behavior_book_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Behavior Curation Pipeline - Metacognitive Reuse for LLM Reasoning"
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        default=None,
        help="Problem/question to solve (required for behavior extraction)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1",
        help="Hugging Face model name to use (default: deepseek-ai/DeepSeek-R1)",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda' or 'cpu' (default: auto-detect)",
    )
    parser.add_argument(
        "-p",
        "--papers-dir",
        type=str,
        default=None,
        help="Directory containing PDF papers to process (default: data/papers/iclr23_top5)",
    )
    parser.add_argument(
        "-n",
        "--num-papers",
        type=int,
        default=None,
        help="Number of papers to process (default: all papers in directory)",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Process a single question without papers (legacy mode)",
    )

    args = parser.parse_args()

    # Example usage
    reader = ChainOfThoughtReader(
        model_name=args.model, 
        task=args.task,
        device=args.device,
        papers_dir=args.papers_dir,
    )

    try:
        if not args.task:
            print("Error: Please provide a problem/question using -t or --task")
            print("Example: python client.py -t 'What are the key contributions?' -n 5")
            exit(1)

        # If --single flag is set or no papers directory specified, use legacy single mode
        if args.single or (args.papers_dir is None and args.num_papers is None):
            result = reader.read_paper(task=args.task)
        reader.save_reasoning(result)

        print("\n" + "=" * 80)
        print("BEHAVIOR CURATION PIPELINE COMPLETE")
        print("=" * 80)
        print(result["complete_reasoning"])
        print("\n" + "=" * 80)
        print("EXTRACTED BEHAVIORS")
        print("=" * 80)
        for behavior_name, behavior_desc in result.get("behavior_book", {}).items():
            print(f"\n{behavior_name}: {behavior_desc}")
        print("\n" + "=" * 80)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        print("\nMake sure you have:")
        print("1. Installed required packages: pip install -r requirements.txt")
        print("2. For GPU support, ensure CUDA is properly installed")
        print("\nExample usage:")
        print("  # Process multiple papers:")
        print("  python client.py -t 'What are the key contributions?' -n 5")
        print(
            "  python client.py -t 'Analyze this paper' -p data/papers/iclr23_top5 -n 10"
        )
        print("  # Single question mode:")
        print(
            "  python client.py -t 'Find the area of a circle with radius 4' --single"
        )
