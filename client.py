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
                        repetition_penalty=1.2,
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
                        repetition_penalty=1.2,  # Penalize repetition to avoid loops
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
        """Step 2: Extract insights and learnings from the solution"""
        prompt = f"""
You are analyzing a step-by-step solution to extract key insights and learnings.

Problem:
{problem}

Step-by-Step Solution:
{solution}

Your task: Extract insights, learnings, and key reasoning patterns from the solution. Focus on:
1. **What was learned**: What concepts, techniques, or strategies were used?
2. **How it was applied**: What were the concrete steps and reasoning patterns?
3. **Why it worked**: What insights made this approach effective?
4. **Generalizable patterns**: What can be reused for similar problems?

Output format (structured analysis):

### I. Key Insights and Learnings
- List the main insights gained from solving this problem
- Identify the core reasoning patterns used
- Note any "aha moments" or critical realizations

### II. Step-by-Step Reasoning Analysis
- Break down the solution into logical reasoning steps
- For each major step, explain:
  * What technique or approach was used
  * Why this step was necessary
  * How it connects to the overall solution

### III. Extractable Skills and Techniques
- Identify reusable methods, strategies, or techniques from the solution
- For each technique:
  * Name it clearly
  * Describe when and how it was applied
  * Explain why it was effective
  * Note what types of problems it could help solve

Focus on extracting actionable knowledge that can guide future problem-solving.
"""
        return prompt

    def _get_behavior_prompt(self, problem: str, solution: str, reflection: str) -> str:
        """Step 3: Generate actionable, instruction-based skills (like Claude Agent Skills)"""
        prompt_template = """
You are creating reusable, instruction-based skills that can guide an LLM or agent to solve similar problems step-by-step.

Problem: {problem}

Step-by-Step Solution: {solution}

Insights and Learnings: {reflection}

Your task: Generate concrete, actionable skills in JSON format. Each skill should be like a mini-guide that an agent can follow.

**Skill Requirements:**
1. Each skill must be **actionable** - provide clear, step-by-step instructions
2. Each skill must include **concrete steps** - not vague descriptions
3. Each skill must include **insights** - explain why and when to use it
4. Each skill must be **reusable** - applicable to similar problems, not just this one
5. Skills should be formatted like agent skills: clear instructions that guide step-by-step reasoning

**Output Format:**
Generate a JSON object where:
- Keys are skill names (must start with `skill_`)
- Values are skill descriptions that include:
  * **When to use**: Under what conditions this skill applies
  * **Step-by-step instructions**: Concrete, actionable steps
  * **Key insights**: Why this approach works and what to watch for
  * **Example application**: Brief note on how it was used in this problem

Each skill description should be comprehensive but concise. Use newlines within the description for readability.

Example format:
{{
  "skill_polynomialFactoring": "When to use: When solving equations with polynomial expressions that can be factored.\\n\\nStep-by-step: 1) Identify common factors or patterns (difference of squares, perfect square trinomials, etc.). 2) Apply appropriate factoring technique. 3) Set each factor equal to zero. 4) Solve resulting equations.\\n\\nKey insights: Factoring reduces complex polynomials to simpler linear/quadratic equations. Look for patterns like a²-b²=(a-b)(a+b) or x²+2ax+a²=(x+a)².\\n\\nExample: Used to factor x²-9=(x-3)(x+3) in quadratic equation solving.",
  "skill_systematicSubstitution": "When to use: When dealing with systems of equations or complex expressions with multiple variables.\\n\\nStep-by-step: 1) Identify which variable to substitute. 2) Express one variable in terms of others from one equation. 3) Substitute into other equations. 4) Simplify and solve. 5) Back-substitute to find remaining variables.\\n\\nKey insights: Reduces multi-variable problems to single-variable problems. Choose substitutions that simplify the most.\\n\\nExample: Used to solve system by expressing y in terms of x, then substituting into second equation."
}}

**Important:**
- Only extract skills that are actually present in the solution
- Each skill must have concrete, actionable steps
- Include insights that explain the reasoning behind the approach
- Output ONLY the JSON object, no additional commentary
                """
        prompt = prompt_template.format(
            problem=problem, solution=solution, reflection=reflection
        )
        return prompt

    def _step_solution(self, problem: str) -> Dict:
        """Step 1: Generate solution using Solution Prompt"""
        prompt = self._get_solution_prompt(problem)

        system_prompt = None
        response = self._call_model(prompt, system_prompt, max_new_tokens=16384)
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
            "name": "Insight Extraction",
            "prompt": prompt,
            "response": response,
            "timestamp": time.time(),
        }

        self.reasoning_steps.append(step_result)
        return step_result

    def _step_behavior_extraction(
        self, problem: str, solution: str, reflection: str
    ) -> Dict:
        """Step 3: Extract actionable skills using enhanced Behavior Prompt"""
        prompt = self._get_behavior_prompt(problem, solution, reflection)

        system_prompt = None
        # Step 3: Use 32768 tokens for skill extraction (needs more tokens for comprehensive, detailed skills)
        response = self._call_model(prompt, system_prompt, max_new_tokens=32768)
        print(f"Skill Extraction Response: {response}")  # Print first 500 chars

        # Parse skills from response with validation
        skills = {}
        formatted_json_array = []
        validation_errors = []
        
        try:
            # First, try to extract JSON from markdown code blocks (```json ... ```)
            json_code_block = re.search(
                r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", response, re.DOTALL
            )
            if json_code_block:
                json_str = json_code_block.group(1)
            else:
                # Try to find JSON object {...}
                json_object_match = re.search(r"\{[\s\S]*?\}", response, re.DOTALL)
                if json_object_match:
                    json_str = json_object_match.group(0)
                else:
                    # Try to find JSON array [...]
                    json_array_match = re.search(r"\[[\s\S]*?\]", response, re.DOTALL)
                    if json_array_match:
                        json_str = json_array_match.group(0)
                    else:
                        json_str = None

            if json_str:
                try:
                    # Clean up common JSON issues
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*]', ']', json_str)
                    json_data = json.loads(json_str)
                    
                    # Handle both dict and list formats
                    if isinstance(json_data, list):
                        formatted_json_array = json_data
                        for item in json_data:
                            if isinstance(item, dict):
                                skill_name = item.get("skill_name", item.get("skill", ""))
                                skill_desc = item.get("description", "")
                                if skill_name:
                                    if not skill_name.startswith("skill_"):
                                        skill_name = f"skill_{skill_name}"
                                    skills[skill_name] = skill_desc
                    elif isinstance(json_data, dict):
                        skills = json_data
                        formatted_json_array = [
                            {"skill_name": k, "description": v}
                            for k, v in skills.items()
                        ]
                    
                    # Validate skills
                    validated_skills = {}
                    for skill_name, skill_desc in skills.items():
                        # Check if skill name starts with skill_
                        if not skill_name.startswith("skill_"):
                            validation_errors.append(f"Skill '{skill_name}' does not start with 'skill_' prefix")
                            continue
                        
                        # Check if description is empty or too short
                        if not skill_desc or len(skill_desc.strip()) < 20:
                            validation_errors.append(f"Skill '{skill_name}' has empty or too short description")
                            continue
                        
                        # Check if description contains actionable content
                        skill_desc_lower = skill_desc.lower()
                        has_steps = any(keyword in skill_desc_lower for keyword in ["step", "1)", "2)", "when to use", "how to", "instructions"])
                        if not has_steps:
                            validation_errors.append(f"Skill '{skill_name}' may lack concrete steps or instructions")
                        
                        validated_skills[skill_name] = skill_desc
                    
                    skills = validated_skills
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error: {e}")
                    validation_errors.append(f"JSON parsing error: {e}")

            # If still no skills, try manual extraction
            if not skills:
                print("Warning: Could not parse JSON from skill extraction. Attempting manual extraction.")
                # Try to extract skills manually (look for skill_* patterns)
                skill_pattern = r'["\']?(skill_\w+)["\']?\s*[:=]\s*["\']?([^"\']+)["\']?'
                matches = re.findall(skill_pattern, response)
                for name, desc in matches:
                    if len(desc.strip()) >= 20:
                        skills[name] = desc.strip()
                if not skills:
                    validation_errors.append("Could not extract any skills from response")
        except Exception as e:
            print(f"Warning: Error parsing skills: {e}. Storing raw response.")
            validation_errors.append(f"Exception during parsing: {e}")
            if not skills:
                skills = {"raw_response": response}
        
        # Sanity check: Ensure we have at least one valid skill
        valid_skills = {k: v for k, v in skills.items() if k.startswith("skill_") and v and len(v.strip()) >= 20}
        if not valid_skills:
            print("WARNING: No valid skills extracted! Adding fallback skill.")
            skills["skill_fallback"] = "When to use: For any problem-solving task.\\n\\nStep-by-step: 1) Carefully read and understand the problem. 2) Break down the problem into smaller sub-problems. 3) Apply relevant mathematical/analytical techniques systematically. 4) Verify each step for correctness. 5) Check the final answer.\\n\\nKey insights: Systematic approach reduces errors. Break complex problems into manageable parts.\\n\\nExample: Used as a general problem-solving framework."
            valid_skills = {"skill_fallback": skills["skill_fallback"]}
            formatted_json_array = [{"skill_name": "skill_fallback", "description": skills["skill_fallback"]}]
        
        # Report validation results
        if validation_errors:
            print(f"Validation warnings ({len(validation_errors)}):")
            for error in validation_errors[:5]:  # Show first 5 errors
                print(f"  - {error}")
        
        print(f"Extracted {len(valid_skills)} valid skills: {list(valid_skills.keys())}")

        step_result = {
            "step": 3,
            "name": "Skill Extraction",
            "prompt": prompt,
            "response": response,
            "skills": skills,
            "valid_skills": valid_skills,
            "formatted_json_array": formatted_json_array,
            "validation_errors": validation_errors,
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

        # Step 2: Extract Insights and Learnings
        print("Step 2: Extracting insights and learnings from solution...")
        step2 = self._step_reflection(problem, solution)
        reflection = step2["response"]
        time.sleep(1)  # Rate limiting

        # Step 3: Skill Extraction
        print("Step 3: Extracting actionable skills...")
        step3 = self._step_behavior_extraction(problem, solution, reflection)
        time.sleep(1)
        
        # Update behavior_book with extracted skills
        extracted_skills = step3.get("valid_skills", step3.get("skills", {}))
        if extracted_skills:
            self.behavior_book.update(extracted_skills)
            print(f"Added {len(extracted_skills)} skills to skill book")
        else:
            print("WARNING: No skills extracted from this problem!")

        # Compile results
        result = {
            "problem": problem,
            "task": self.task,
            "solution": solution,
            "reflection": reflection,
            "skills_extracted": step3.get("valid_skills", step3.get("skills", {})),
            "skills_used": list(step3.get("valid_skills", step3.get("skills", {})).keys()),
            "validation_errors": step3.get("validation_errors", []),
            "behavior_book": self.behavior_book,
            "total_steps": len(self.reasoning_steps),
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
        result = None
        if args.single or (args.papers_dir is None and args.num_papers is None):
            result = reader.read_paper(task=args.task)
            reader.save_reasoning(result)
            
            if result:
                print("\n" + "=" * 80)
                print("SKILL CURATION PIPELINE COMPLETE")
                print("=" * 80)
                print(f"Solution: {result.get('solution', 'N/A')[:200]}...")
                print(f"\nSkills Extracted: {len(result.get('skills_extracted', {}))}")
                print(f"Skills Used: {result.get('skills_used', [])}")
                if result.get('validation_errors'):
                    print(f"Validation Warnings: {len(result.get('validation_errors', []))}")
                print("\n" + "=" * 80)
                print("EXTRACTED SKILLS")
                print("=" * 80)
                for skill_name, skill_desc in result.get("behavior_book", {}).items():
                    print(f"\n{skill_name}: {skill_desc[:200]}...")
                print("\n" + "=" * 80)
        else:
            # Process multiple papers
            papers_dir = args.papers_dir or reader.papers_dir
            if not papers_dir:
                print("Error: Please provide a papers directory using -p or --papers-dir")
                print("Example: python client.py -t 'Question' -p data/papers/iclr23_top5 -n 10")
                exit(1)
            
            print(f"Processing papers from: {papers_dir}")
            print(f"Number of papers: {args.num_papers or 'all'}")
            print("=" * 80)
            
            results = reader.process_multiple_papers(
                question=args.task,
                papers_dir=papers_dir,
                num_papers=args.num_papers
            )
            
            if results:
                print("\n" + "=" * 80)
                print("ALL PAPERS PROCESSED")
                print("=" * 80)
                print(f"Total papers processed: {len(results)}")
                total_skills = sum(len(r.get('behavior_book', {})) for r in results)
                print(f"Total skills extracted: {total_skills}")
                print("=" * 80)
            else:
                print("\nNo papers were processed. Check that:")
                print(f"1. Directory exists: {papers_dir}")
                print("2. Directory contains PDF files")
                print("3. Files are readable")

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
