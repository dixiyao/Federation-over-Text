"""
Chain-of-Thought Paper Reader Client
Uses a language model (e.g., Deepseek-r1) to read papers with multi-step reasoning.
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
    A client that reads papers using chain-of-thought reasoning with multiple steps.
    Similar to how modern AI assistants (like Cursor) break down complex tasks.
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
        self.generated_prompts = {}  # Store prompts generated in Step 0
        
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

    def _step_0_generate_prompts(self, title: str, paper_excerpt: str) -> Dict:
        """Step 0: Generate prompts for all subsequent steps"""
        # Minimal prompt - just the essential information
        prompt = f"""Paper Title: {title}
                     Paper Excerpt: {paper_excerpt[:2000]}
                     Task/Question: {self.task}

                     Generate 4 prompts for a 4-step paper analysis process. Return JSON format:
                     {{
                        "step1_prompt": "...",
                        "step2_prompt": "...",
                        "step3_prompt": "...",
                        "step4_prompt": "..."
                     }}
                """

        system_prompt = None  # No predefined system prompt

        response = self._call_model(prompt, system_prompt)

        # Try to parse JSON from response
        try:
            # Try to extract JSON from the response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                prompts_dict = json.loads(json_match.group())
                self.generated_prompts = prompts_dict
            else:
                # If JSON parsing fails, store the raw response and try to extract prompts manually
                print(
                    "Warning: Could not parse JSON from Step 0 response. Using raw response."
                )
                self.generated_prompts = {
                    "step1_prompt": response,
                    "step2_prompt": response,
                    "step3_prompt": response,
                    "step4_prompt": response,
                }
        except Exception as e:
            print(
                f"Warning: Error parsing prompts from Step 0: {e}. Using raw response."
            )
            self.generated_prompts = {
                "step1_prompt": response,
                "step2_prompt": response,
                "step3_prompt": response,
                "step4_prompt": response,
            }

        step_result = {
            "step": 0,
            "name": "Generate Prompts for All Steps",
            "prompt": prompt,
            "response": response,
            "generated_prompts": self.generated_prompts,
            "timestamp": time.time(),
        }

        self.reasoning_steps.append(step_result)
        return step_result

    def _step_1_summary_and_plan(self, paper_text: str, title: str) -> Dict:
        """Step 1: Summarize paper, extract learnings, identify limitations and future work, generate plan"""
        # Get the generated prompt from Step 0
        step1_prompt_template = self.generated_prompts.get("step1_prompt", "")

        # Replace placeholders in the prompt with actual values
        prompt = step1_prompt_template.replace("{title}", title)
        prompt = prompt.replace("{paper_text}", paper_text)
        prompt = prompt.replace("{paper_content}", paper_text)
        prompt = prompt.replace("{task}", self.task)
        prompt = prompt.replace("{question}", self.task)

        # If no prompt was generated, use a minimal fallback
        if not step1_prompt_template:
            prompt = f"Title: {title}\n\nPaper Content: {paper_text}\n\nTask: {self.task}\n\nAnalyze this paper."

        system_prompt = None  # No predefined system prompt

        response = self._call_model(prompt, system_prompt)

        step_result = {
            "step": 1,
            "name": "Summary, Learning, Limitations, Future Work, and Plan Generation",
            "prompt": prompt,
            "response": response,
            "timestamp": time.time(),
        }

        self.reasoning_steps.append(step_result)
        return step_result

    def _step_2_find_relevant_content(
        self, paper_text: str, step1_result: Dict
    ) -> Dict:
        """Step 2: Find content that can help and support answering the question"""
        # Get the generated prompt from Step 0
        step2_prompt_template = self.generated_prompts.get("step2_prompt", "")

        # Replace placeholders in the prompt with actual values
        prompt = step2_prompt_template.replace(
            "{title}", step1_result.get("paper_title", "")
        )
        prompt = prompt.replace("{paper_text}", paper_text)
        prompt = prompt.replace("{paper_content}", paper_text)
        prompt = prompt.replace("{task}", self.task)
        prompt = prompt.replace("{question}", self.task)
        prompt = prompt.replace("{step1_result}", step1_result["response"])
        prompt = prompt.replace("{step1_analysis}", step1_result["response"])

        # If no prompt was generated, use a minimal fallback
        if not step2_prompt_template:
            prompt = f"Task: {self.task}\n\nStep 1 Analysis: {step1_result['response']}\n\nPaper Content: {paper_text}\n\nFind relevant content."

        system_prompt = None  # No predefined system prompt

        response = self._call_model(prompt, system_prompt)

        step_result = {
            "step": 2,
            "name": "Find Relevant Content and Evidence",
            "prompt": prompt,
            "response": response,
            "timestamp": time.time(),
        }

        self.reasoning_steps.append(step_result)
        return step_result

    def _step_3_find_answer(self, paper_text: str, previous_steps: List[Dict]) -> Dict:
        """Step 3: Follow the plan and evidence from step 2 to find the answer"""
        # Get the generated prompt from Step 0
        step3_prompt_template = self.generated_prompts.get("step3_prompt", "")

        # Build previous steps context
        previous_steps_context = chr(10).join(
            [
                f"=== Step {s['step']}: {s['name']} ==={chr(10)}{s['response']}{chr(10)}"
                for s in previous_steps
            ]
        )

        # Replace placeholders in the prompt with actual values
        prompt = step3_prompt_template.replace(
            "{title}",
            previous_steps[0].get("paper_title", "") if previous_steps else "",
        )
        prompt = prompt.replace("{paper_text}", paper_text)
        prompt = prompt.replace("{paper_content}", paper_text)
        prompt = prompt.replace("{task}", self.task)
        prompt = prompt.replace("{question}", self.task)
        prompt = prompt.replace("{previous_steps}", previous_steps_context)
        prompt = prompt.replace(
            "{step1_result}", previous_steps[0]["response"] if previous_steps else ""
        )
        prompt = prompt.replace(
            "{step2_result}",
            previous_steps[1]["response"] if len(previous_steps) > 1 else "",
        )

        # If no prompt was generated, use a minimal fallback
        if not step3_prompt_template:
            prompt = f"Task: {self.task}\n\nPrevious Steps:\n{previous_steps_context}\n\nPaper Content: {paper_text}\n\nFind the answer."

        system_prompt = None  # No predefined system prompt

        response = self._call_model(prompt, system_prompt)

        step_result = {
            "step": 3,
            "name": "Find Answer Using Plan and Evidence",
            "prompt": prompt,
            "response": response,
            "timestamp": time.time(),
        }

        self.reasoning_steps.append(step_result)
        return step_result

    def _step_4_polish_and_novelty(
        self, paper_text: str, all_steps: List[Dict]
    ) -> Dict:
        """Step 4: Polish the answer and think of novelty"""
        # Get the generated prompt from Step 0
        step4_prompt_template = self.generated_prompts.get("step4_prompt", "")

        # Build all steps context
        all_steps_context = chr(10).join(
            [
                f"=== Step {s['step']}: {s['name']} ==={chr(10)}{s['response']}{chr(10)}"
                for s in all_steps
            ]
        )

        # Replace placeholders in the prompt with actual values
        prompt = step4_prompt_template.replace(
            "{title}", all_steps[0].get("paper_title", "") if all_steps else ""
        )
        prompt = prompt.replace("{paper_text}", paper_text)
        prompt = prompt.replace("{paper_content}", paper_text)
        prompt = prompt.replace("{task}", self.task)
        prompt = prompt.replace("{question}", self.task)
        prompt = prompt.replace("{all_steps}", all_steps_context)
        prompt = prompt.replace("{complete_reasoning}", all_steps_context)
        prompt = prompt.replace(
            "{step1_result}", all_steps[0]["response"] if all_steps else ""
        )
        prompt = prompt.replace(
            "{step2_result}", all_steps[1]["response"] if len(all_steps) > 1 else ""
        )
        prompt = prompt.replace(
            "{step3_result}", all_steps[2]["response"] if len(all_steps) > 2 else ""
        )

        # If no prompt was generated, use a minimal fallback
        if not step4_prompt_template:
            prompt = f"Task: {self.task}\n\nComplete Reasoning:\n{all_steps_context}\n\nPaper Content: {paper_text}\n\nPolish answer and identify novelty."

        system_prompt = None  # No predefined system prompt

        response = self._call_model(prompt, system_prompt)

        step_result = {
            "step": 4,
            "name": "Polish Answer and Identify Novelty",
            "prompt": prompt,
            "response": response,
            "timestamp": time.time(),
        }

        self.reasoning_steps.append(step_result)
        return step_result

    def read_paper(
        self, paper_path: Optional[str] = None, task: Optional[str] = None
    ) -> Dict:
        """
        Main method to read a paper with chain-of-thought reasoning.

        Args:
            paper_path: Path to the paper PDF. If None, reads the first paper found.
            task: Optional task/question to guide the analysis. If None, uses the default task.

        Returns:
            Dictionary containing all reasoning steps and final analysis.
        """
        # Update task if provided
        if task is not None:
            self.task = task

        # Find a paper to read
        if paper_path is None:
            papers_dir = Path(self.papers_dir)
            pdf_files = list(papers_dir.glob("*.pdf"))
            if not pdf_files:
                raise FileNotFoundError(f"No PDF files found in {self.papers_dir}")
            paper_path = str(pdf_files[0])
            print(f"Reading paper: {paper_path}")

        # Extract text from PDF
        print("Extracting text from PDF...")
        paper_text = self._extract_text_from_pdf(paper_path)
        if not paper_text:
            raise ValueError("Failed to extract text from PDF")

        # Get paper title (from filename or extract from text)
        paper_title = Path(paper_path).stem
        if "Title:" in paper_text[:500]:
            title_line = [
                line for line in paper_text[:500].split("\n") if "Title:" in line
            ]
            if title_line:
                paper_title = title_line[0].replace("Title:", "").strip()

        print(f"Paper: {paper_title}")
        print(f"Task/Question: {self.task}")
        print(f"Text length: {len(paper_text)} characters\n")

        # Reset reasoning steps and generated prompts
        self.reasoning_steps = []
        self.generated_prompts = {}

        # Step 0: Generate prompts for all subsequent steps
        print("Step 0: Generating prompts for all steps...")
        step0 = self._step_0_generate_prompts(paper_title, paper_text)
        time.sleep(1)  # Rate limiting

        # Step 1: Summary, Learning, Limitations, Future Work, and Plan
        print(
            "Step 1: Summary, Learning, Limitations, Future Work, and Plan Generation..."
        )
        step1 = self._step_1_summary_and_plan(paper_text, paper_title)
        time.sleep(1)  # Rate limiting

        # Step 2: Find Relevant Content
        print("Step 2: Finding Relevant Content and Evidence...")
        step2 = self._step_2_find_relevant_content(paper_text, step1)
        time.sleep(1)

        # Step 3: Find Answer
        print("Step 3: Finding Answer Using Plan and Evidence...")
        step3 = self._step_3_find_answer(paper_text, [step1, step2])
        time.sleep(1)

        # Step 4: Polish and Novelty
        print("Step 4: Polishing Answer and Identifying Novelty...")
        self._step_4_polish_and_novelty(paper_text, [step0, step1, step2, step3])

        # Compile results
        result = {
            "paper_path": paper_path,
            "paper_title": paper_title,
            "task": self.task,
            "generated_prompts": self.generated_prompts,
            "reasoning_steps": self.reasoning_steps,
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
        """Save the complete reasoning process to a file"""
        if output_path is None:
            output_path = f"reasoning_{Path(reasoning_result['paper_path']).stem}.txt"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Paper: {reasoning_result['paper_title']}\n")
            f.write(f"Path: {reasoning_result['paper_path']}\n")
            f.write(f"\n{'='*80}\n")
            f.write("COMPLETE REASONING PROCESS\n")
            f.write(f"{'='*80}\n")
            f.write(reasoning_result["complete_reasoning"])

        # Also save as JSON for structured access
        json_path = output_path.replace(".txt", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(reasoning_result, f, indent=2, ensure_ascii=False)

        print("\nReasoning saved to:")
        print(f"  - {output_path}")
        print(f"  - {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chain-of-Thought Paper Reader with 4-step reasoning process"
    )
    parser.add_argument(
        "-p",
        "--paper",
        type=str,
        default=None,
        help="Path to the paper PDF (default: first paper found in data/papers/iclr23_top5)",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        default=None,
        help="Task/question to guide the analysis (default: general analysis)",
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
        result = reader.read_paper(paper_path=args.paper, task=args.task)
        reader.save_reasoning(result)

        print("\n" + "=" * 80)
        print("REASONING PROCESS COMPLETE")
        print("=" * 80)
        print(result["complete_reasoning"])
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Scraped papers using scraper.py")
        print("2. Installed required packages: pip install -r requirements.txt")
        print("3. For GPU support, ensure CUDA is properly installed")
        print("\nExample usage with different models:")
        print("  python client.py -m deepseek-ai/DeepSeek-R1")
        print("  python client.py -m meta-llama/Llama-2-7b-chat-hf")
        print("  python client.py -m microsoft/phi-2")
