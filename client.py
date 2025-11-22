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
from typing import Dict, Optional

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
    ):
        self.model_name = model_name
        self.papers_dir = "data/papers/iclr23_top5"
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
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
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

    def _call_model(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Call the Hugging Face language model.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt (will be prepended if provided)
        
        Returns:
            Generated text response
        """
        # Load model if not already loaded
        self._load_model()
        
        # Combine system prompt and user prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        try:
            
            # Tokenize input
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096, 
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,  
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
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
Problem: {problem}

Solution: {solution}
        
Here is the definition of a behavior:
- A note or skill to keep in mind while solving math problems.
- A strategy, a trick, or a technique.
- A general rule or a common sense principle.
- Not a solution to the problem, but it can be used to solve the problem.

For example - if the problem is "Find the area of a circle with radius 4", one useful behaviour could be:
{{"behavior_area_of_circle": "area of a circle is pi*r^2"}}

Given a problem and the corresponding solution, reflect and critique the solutions along the following dimensions:

1. Correctness Analysis:
   - Is the solution mathematically correct?
   - Are there any calculation errors?
   - Is the reasoning logically sound?
   - Are all steps properly justified?
   - What mistakes, if any, were made?

2. Missing Behaviors Analysis:
   - What behaviors should have been used but weren't?
   - Remember: a behavior is a note or instruction for quickly using concepts without deriving them from scratch.
   - For each missing behavior, explain:
     * How would it have reduced the length of the answer?
     * How would it have prevented errors?
     * Why is it crucial for similar problems?
     * How would it have made the solution more elegant (even if correct)?

3. New Behavior Suggestions:
   - Suggest specific new behaviors for similar problems.
   - For each new behavior:
     * The name must start with 'behavior_'
     * Provide clear and actionable instructions
     * Include examples where helpful
     * Make sure it's general enough for similar problems
     * Explain why this behavior is valuable
"""
        return prompt

    def _get_behavior_prompt(self, problem: str, solution: str, reflection: str) -> str:
        """Get the Behavior Prompt as defined in the paper"""
        prompt = f"""Problem: {problem}

Solution: {solution}

Reflection: {reflection}

Now, given this reflection generate a list of behaviors and corresponding instructions/explanations in json format. Each behavior should be a single line, and the format is "behavior_[name]: [description]". The list should be in json format, and each behavior should be a key-value pair, where the key is the behavior name and the value is the description."""
        return prompt

    def _step_solution(self, problem: str) -> Dict:
        """Step 1: Generate solution using Solution Prompt"""
        prompt = self._get_solution_prompt(problem)
        
        system_prompt = None
        response = self._call_model(prompt, system_prompt)

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
        response = self._call_model(prompt, system_prompt)

        step_result = {
            "step": 2,
            "name": "Reflection and Critique",
            "prompt": prompt,
            "response": response,
            "timestamp": time.time(),
        }

        self.reasoning_steps.append(step_result)
        return step_result

    def _step_behavior_extraction(self, problem: str, solution: str, reflection: str) -> Dict:
        """Step 3: Extract behaviors using Behavior Prompt"""
        prompt = self._get_behavior_prompt(problem, solution, reflection)
        
        system_prompt = None
        response = self._call_model(prompt, system_prompt)

        # Try to parse behaviors from JSON response
        behaviors = {}
        try:
            # Try to extract JSON from the response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                behaviors = json.loads(json_match.group())
                # Update behavior book
                self.behavior_book.update(behaviors)
            else:
                print("Warning: Could not parse JSON from behavior extraction. Attempting manual extraction.")
                # Try to extract behaviors manually (look for behavior_* patterns)
                behavior_pattern = r'["\']?(behavior_\w+)["\']?\s*[:=]\s*["\']?([^"\']+)["\']?'
                matches = re.findall(behavior_pattern, response)
                for name, desc in matches:
                    behaviors[name] = desc.strip()
                self.behavior_book.update(behaviors)
        except Exception as e:
            print(f"Warning: Error parsing behaviors: {e}. Storing raw response.")
            behaviors = {"raw_response": response}

        step_result = {
            "step": 3,
            "name": "Behavior Extraction",
            "prompt": prompt,
            "response": response,
            "behaviors": behaviors,
            "timestamp": time.time(),
        }

        self.reasoning_steps.append(step_result)
        return step_result

    def read_paper(
        self, task: Optional[str] = None
    ) -> Dict:
        """
        Main method to process a question/problem using behavior curation pipeline.
        Implements the three-stage pipeline: Solution → Reflection → Behavior Extraction

        Args:
            task: Optional task/question/problem to solve. If None, uses the default task.

        Returns:
            Dictionary containing all reasoning steps, behaviors, and behavior book.
        """
        # Update task if provided
        if task is not None:
            self.task = task

        # The task/question is treated as the problem to solve
        problem = self.task

        print(f"Problem/Question: {problem}\n")

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
            "reasoning_steps": self.reasoning_steps,
            "solution": solution,
            "reflection": reflection,
            "behaviors": step3.get("behaviors", {}),
            "behavior_book": self.behavior_book,
            "total_steps": len(self.reasoning_steps),
            "complete_reasoning": self._format_complete_reasoning(),
        }

        return result

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
            safe_name = re.sub(r'[^\w\s-]', '', reasoning_result.get('problem', 'reasoning')[:50])
            safe_name = re.sub(r'[-\s]+', '_', safe_name)
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
            for behavior_name, behavior_desc in reasoning_result.get("behavior_book", {}).items():
                f.write(f"{behavior_name}: {behavior_desc}\n")

        # Also save as JSON for structured access
        json_path = output_path.replace(".txt", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(reasoning_result, f, indent=2, ensure_ascii=False)

        # Save behavior book separately as JSON
        behavior_book_path = output_path.replace(".txt", "_behavior_book.json")
        with open(behavior_book_path, "w", encoding="utf-8") as f:
            json.dump(reasoning_result.get("behavior_book", {}), f, indent=2, ensure_ascii=False)

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

    args = parser.parse_args()

    # Example usage
    reader = ChainOfThoughtReader(
        model_name=args.model, 
        task=args.task,
        device=args.device
    )

    try:
        if not args.task:
            print("Error: Please provide a problem/question using -t or --task")
            print("Example: python client.py -t 'Find the area of a circle with radius 4'")
            exit(1)
        
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
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Installed required packages: pip install -r requirements.txt")
        print("2. For GPU support, ensure CUDA is properly installed")
        print("\nExample usage:")
        print("  python client.py -t 'Find the area of a circle with radius 4'")
        print("  python client.py -t 'Solve: x^2 + 5x + 6 = 0' -m deepseek-ai/DeepSeek-R1")
