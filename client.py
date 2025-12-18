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

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


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
        use_gemini: bool = False,
        gemini_api_key: Optional[str] = None,
        output_dir: str = "output",
    ):
        self.model_name = model_name
        self.papers_dir = papers_dir or "data/papers/iclr23_top5"
        self.output_dir = output_dir
        self.reasoning_steps = []
        self.task = (
            task
            or "Analyze this paper and identify key contributions, limitations, and potential future research directions."
        )
        self.behavior_book = {}  # Store extracted behaviors
        
        # Gemini API support
        self.use_gemini = use_gemini
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if self.use_gemini:
            if not HAS_GEMINI:
                raise ImportError(
                    "google-generativeai is required for Gemini API. Install with: pip install google-generativeai"
                )
            if not self.gemini_api_key:
                raise ValueError("Gemini API key is required when use_gemini=True. Set GEMINI_API_KEY env var or pass gemini_api_key parameter.")
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-3-pro-preview')
        
        # Model and tokenizer will be loaded lazily on first use (only for HuggingFace models)
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

    def _call_model(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Call the language model (HuggingFace or Gemini API).
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt (will be prepended if provided)
                          NOTE: For DeepSeek-R1 models, avoid system prompts - put all in user prompt
            max_new_tokens: Maximum number of new tokens to generate. If None, uses default 32768.
        
        Returns:
            Generated text response
        """
        # Use Gemini API if configured
        if self.use_gemini:
            return self._call_gemini(prompt, system_prompt, max_new_tokens)
        
        # Otherwise use HuggingFace model
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
                # Use standard settings for reliable generation
                # Check if model name contains "DeepSeek-R1" to use recommended settings
                is_deepseek_r1 = "DeepSeek-R1" in self.model_name

                if is_deepseek_r1:
                    # DeepSeek-R1 settings - more conservative for reliable output
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
                            repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"Error calling model: {e}")
    
    def _call_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Call Gemini API"""
        try:
            # Combine system prompt and user prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            # Configure generation parameters
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.9,
            }
            if max_new_tokens:
                generation_config["max_output_tokens"] = min(max_new_tokens, 8192)  # Gemini limit
            
            # Generate response
            response = self.gemini_model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            return response.text.strip()
            
        except Exception as e:
            raise RuntimeError(f"Error calling Gemini API: {e}")
            return f"[Error] Model generation failed: {str(e)}"

    def _get_solution_prompt(self, problem: str) -> str:
        """Get the Solution Prompt as defined in the paper"""
        prompt = f"""Please reason step by step and put the final answer in <answer_box>.

Problem: {problem}"""
        return prompt

    def _get_reflection_prompt(self, problem: str, solution: str) -> str:
        """Step 2: Extract procedural knowledge and reusable patterns for skill creation"""
        prompt = f"""
Analyze the solution below to extract procedural knowledge that can be turned into reusable skills. 
Skills are instructions that teach how to complete specific tasks in a repeatable way.

Problem:
{problem}

Step-by-Step Solution:
{solution}

Your task: Extract procedural knowledge and reusable patterns that can be packaged as skills. Focus on:

1. **Procedural Patterns**: What step-by-step procedures were used? How can these be repeated?
2. **Decision Points**: What conditions determined which approach to use? When should each technique apply?
3. **Reusable Techniques**: What methods, strategies, or workflows can be applied to similar problems?
4. **Key Insights**: What made this approach effective? What should someone know to use it correctly?
5. **Applicability**: What types of problems would benefit from these techniques?

Output your analysis covering:

### I. Procedural Knowledge
- Break down the solution into clear, repeatable procedures
- Identify the sequence of steps that led to success
- Note any decision-making criteria or conditions

### II. Reusable Techniques and Methods
- List specific techniques, strategies, or workflows used
- For each technique, identify:
  * When it should be used (conditions/triggers)
  * How it was applied (concrete steps)
  * Why it was effective (insights)
  * What problems it could solve (applicability)

### III. Critical Insights and Guidelines
- What key insights made this solution work?
- What common pitfalls should be avoided?
- What variations or edge cases should be considered?

Focus on extracting actionable, procedural knowledge that can be packaged as reusable skills for similar problems.
"""
        return prompt

    def _get_behavior_prompt(self, problem: str, solution: str, reflection: str) -> str:
        """Step 3: Generate skills following Anthropic Skills format - procedural knowledge with instructions"""
        prompt_template = """
Extract reusable skills from the solution below. Analyze the solution and reflection to identify concrete, actionable skills that can be applied to similar problems.

Problem: {problem}

Solution: {solution}

Reflection: {reflection}

**Your Task:**
Identify and extract all reusable skills, techniques, and methods used in the solution. Each skill should be a concrete procedure that can guide someone to solve similar problems.

**What Makes a Good Skill:**
- A specific technique or method that was used in the solution
- Something that can be applied to similar problems, not just this one
- Has clear steps that can be followed
- Includes guidance on when to use it

**Skill Description Must Include:**

1. **When to use**: Explain when this skill should be applied. What types of problems? What conditions must be met? What situations trigger this skill?

2. **Step-by-step**: Provide detailed, numbered steps (1) 2) 3) ...) that explain exactly how to apply this skill. Include specific techniques, formulas, methods, or approaches. Be concrete and actionable.

3. **Key insights** (optional): Important considerations, common pitfalls, tips, or why this approach works.

**Output Format (Simple JSON):**
Output a simple JSON object with skill names as keys and descriptions as string values:

{{"skill_name": "description"}}

Format Rules:
- Use valid JSON format
- Each skill name must start with "skill_"
- Each description is a single string containing: "When to use:" and "Step-by-step:" sections
- Steps must be numbered: 1) 2) 3)
- Keep JSON simple - no nested objects, just key-value pairs
- Escape quotes in descriptions with backslash: \\"

**Example:**
{{
  "skill_polynomialFactoring": "When to use: When solving equations with polynomial expressions that can be factored, especially when the polynomial has recognizable patterns like difference of squares (a²-b²), perfect square trinomials (a²±2ab+b²), or common factors. This skill is particularly useful for quadratic and higher-degree polynomial equations where factoring can simplify the problem. Step-by-step: 1) Examine the polynomial structure carefully to identify common patterns such as difference of squares where a²-b²=(a-b)(a+b), perfect square trinomials where a²+2ab+b²=(a+b)² or a²-2ab+b²=(a-b)², and common factors that can be factored out using the distributive property 2) Apply the appropriate factoring technique based on the identified pattern - for difference of squares use (a-b)(a+b), for perfect squares use (a±b)², and for common factors factor out the greatest common divisor 3) Set each factor equal to zero to create simpler equations that are easier to solve, using the zero product property which states that if ab=0 then either a=0 or b=0 4) Solve the resulting linear or quadratic equations using standard algebraic methods such as isolating the variable or applying the quadratic formula if needed 5) Verify all solutions by substituting them back into the original equation to ensure they satisfy the equation and check for extraneous solutions that may have been introduced. Key insights: Factoring reduces complex polynomials to simpler equations. Always look for patterns first before attempting brute force methods.",
  "skill_systematicSubstitution": "When to use: When dealing with systems of equations or complex expressions with multiple variables where one variable can be expressed in terms of others, making the problem more manageable. This approach is especially effective when one equation is already solved for a variable or can be easily rearranged. Step-by-step: 1) Identify which variable to substitute by finding the simplest relationship - look for equations where one variable is already isolated or can be easily isolated, prefer variables with coefficient 1 or -1 2) Express one variable in terms of others from one equation - solve for the chosen variable explicitly, ensuring the expression is valid for all values in the domain 3) Substitute this expression into other equations in the system, replacing all instances of the substituted variable with the expression, being careful to maintain parentheses when the expression contains multiple terms 4) Simplify the resulting equation(s) to reduce the number of variables and create a more solvable form, combining like terms and applying algebraic operations 5) Solve for the remaining variable(s) using appropriate algebraic techniques such as linear equation solving, quadratic formula, or other methods depending on the equation type 6) Back-substitute to find the values of all variables by plugging the solved values back into the substitution expression, working backwards through the system 7) Verify the solution satisfies all original equations by checking each equation with the found values, ensuring no arithmetic errors were made. Key insights: Substitution reduces multi-variable problems to single-variable problems. Choose the substitution that simplifies the most complex parts first."
}}

**CRITICAL Requirements:**
1. You MUST extract at least one skill from the solution - analyze the solution carefully
2. Extract ALL distinct skills, techniques, and methods used in the solution
3. Each skill must have a "When to use:" section and "Step-by-step:" section in the description
4. Output ONLY valid JSON - no markdown, no code blocks, no extra text
5. Escape all double quotes inside descriptions with backslash (\\")
6. Each skill name must start with "skill_"
7. Be thorough - if a technique was used, extract it as a skill

**Output your response as a valid JSON object only:**
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
        response = self._call_model(prompt, system_prompt, max_new_tokens=32768)
        print(f"Skill Extraction Response: {response}")

        # Simple JSON extraction: parse skills from JSON format
        skills = {}
        validation_errors = []
        
        try:
            # Method 1: Extract JSON from markdown code blocks
            json_code_block = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL
            )
            if json_code_block:
                json_str = json_code_block.group(1)
            else:
                # Method 2: Find JSON object in response
                start_idx = response.find('{')
                if start_idx != -1:
                    # Find matching closing brace
                    brace_count = 0
                    in_string = False
                    escape_next = False
                    for i in range(start_idx, len(response)):
                        char = response[i]
                        if escape_next:
                            escape_next = False
                            continue
                        if char == '\\':
                            escape_next = True
                            continue
                        if char == '"' and not escape_next:
                            in_string = not in_string
                            continue
                        if not in_string:
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_str = response[start_idx:i+1]
                                    break
                    else:
                        # If no complete match, try last brace
                        last_brace = response.rfind('}', start_idx)
                        if last_brace != -1:
                            json_str = response[start_idx:last_brace+1]
                        else:
                            json_str = None
                else:
                    json_str = None
            
            if json_str:
                try:
                    # Simple cleanup
                    json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                    json_str = re.sub(r',\s*]', ']', json_str)
                    
                    # Parse JSON
                    json_data = json.loads(json_str)
                    
                    if isinstance(json_data, dict):
                        for skill_name, skill_desc in json_data.items():
                            # Ensure skill name starts with skill_
                            if not skill_name.startswith("skill_"):
                                skill_name = f"skill_{skill_name}"
                            
                            # Convert to string and normalize
                            if isinstance(skill_desc, dict):
                                skill_desc = str(skill_desc)
                            elif isinstance(skill_desc, list):
                                skill_desc = " ".join(str(item) for item in skill_desc)
                            elif not isinstance(skill_desc, str):
                                skill_desc = str(skill_desc)
                            
                            # Normalize whitespace
                            skill_desc = re.sub(r'\s+', ' ', skill_desc).strip()
                            
                            # Validate
                            if len(skill_desc) >= 20:
                                skills[skill_name] = skill_desc
                            else:
                                validation_errors.append(f"Skill '{skill_name}' has too short description")
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error: {e}")
                    validation_errors.append(f"JSON parsing error: {e}")
            
            # Method 3: Fallback - extract using regex if JSON parsing failed
            if not skills:
                print("Warning: JSON parsing failed. Attempting regex extraction.")
                # Extract skill_name: "description" patterns
                skill_pattern = r'"skill_\w+"\s*:\s*"((?:[^"\\]|\\.)*)"'
                name_pattern = r'"skill_\w+"'
                names = re.findall(name_pattern, response)
                descriptions = re.findall(skill_pattern, response)
                
                for i, name in enumerate(names):
                    if i < len(descriptions):
                        skill_name = name.strip('"')
                        skill_desc = descriptions[i].replace('\\"', '"').replace('\\n', ' ')
                        skill_desc = re.sub(r'\s+', ' ', skill_desc).strip()
                        if len(skill_desc) >= 20:
                            skills[skill_name] = skill_desc
            
        except Exception as e:
            print(f"Warning: Error parsing skills: {e}")
            validation_errors.append(f"Exception during parsing: {e}")
        
        if not skills:
            validation_errors.append("Could not extract any skills from response")

        # Filter valid skills
        valid_skills = {}
        for k, v in skills.items():
            if not k.startswith("skill_"):
                continue
            if isinstance(v, str) and len(v.strip()) >= 20:
                valid_skills[k] = v
        if not valid_skills:
            print("WARNING: No valid skills extracted from this problem!")
            # Do not add fallback skill - just report the warning
            # formatted_json_array will remain empty or as is

        # Report validation results
        if validation_errors:
            print(f"Validation warnings ({len(validation_errors)}):")
            for error in validation_errors[:5]:  # Show first 5 errors
                print(f"  - {error}")

        print(
            f"Extracted {len(valid_skills)} valid skills: {list(valid_skills.keys())}"
        )

        # Create formatted JSON array for backward compatibility
        formatted_json_array = [
            {"skill_name": name, "description": desc}
            for name, desc in valid_skills.items()
        ]

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
            "skills_used": list(
                step3.get("valid_skills", step3.get("skills", {})).keys()
            ),
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
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

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

                # Save only skill book as simple JSON: {"skill_name": "description"}
                skill_book = result.get("behavior_book", {})
                if skill_book:
                    output_path = output_dir / f"paper_{idx:02d}.json"
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(skill_book, f, indent=2, ensure_ascii=False)
                    print(f"Saved skill book to: {output_path}")
                    print(f"Skills extracted: {len(skill_book)}")
                else:
                    print(f"No skills extracted from {paper_name}")
                print("-" * 80)

                all_results.append(result)

            except Exception as e:
                print(f"Error processing {paper_name}: {e}")
                import traceback

                traceback.print_exc()
                continue

        print(f"\n{'='*80}")
        print(f"Completed processing {len(all_results)}/{len(pdf_files)} papers")
        total_skills = sum(len(r.get("behavior_book", {})) for r in all_results)
        print(f"Total skills extracted: {total_skills}")
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
        """Save only skill book as simple JSON: {"skill_name": "description"}"""
        # Ensure output directory exists
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        skill_book = reasoning_result.get("behavior_book", {})
        if not skill_book:
            print("No skills to save")
            return
        
        if output_path is None:
            # Create a safe filename from the problem/question
            safe_name = re.sub(
                r"[^\w\s-]", "", reasoning_result.get("problem", "reasoning")[:50]
            )
            safe_name = re.sub(r"[-\s]+", "_", safe_name)
            output_path = str(output_dir / f"{safe_name}.json")
        else:
            # If relative path, make it relative to output_dir
            if not os.path.isabs(output_path):
                output_path = str(output_dir / output_path)
            # Ensure .json extension
            if not output_path.endswith(".json"):
                output_path += ".json"

        # Save only skill book as simple JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(skill_book, f, indent=2, ensure_ascii=False)

        print(f"Saved skill book to: {output_path}")


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
    parser.add_argument(
        "--use-gemini",
        action="store_true",
        help="Use Google Gemini API instead of HuggingFace model",
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Google Gemini API key (or set GEMINI_API_KEY environment variable)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output",
        help="Output directory for saving results (default: output)",
    )

    args = parser.parse_args()

    # Example usage
    reader = ChainOfThoughtReader(
        model_name=args.model, 
        task=args.task,
        device=args.device,
        papers_dir=args.papers_dir,
        use_gemini=args.use_gemini,
        gemini_api_key=args.gemini_api_key,
        output_dir=args.output,
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
                if result.get("validation_errors"):
                    print(
                        f"Validation Warnings: {len(result.get('validation_errors', []))}"
                    )
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
                print(
                    "Error: Please provide a papers directory using -p or --papers-dir"
                )
                print(
                    "Example: python client.py -t 'Question' -p data/papers/iclr23_top5 -n 10"
                )
                exit(1)

            print(f"Processing papers from: {papers_dir}")
            print(f"Number of papers: {args.num_papers or 'all'}")
            print("=" * 80)

            results = reader.process_multiple_papers(
                question=args.task, papers_dir=papers_dir, num_papers=args.num_papers
            )

            if results:
                print("\n" + "=" * 80)
                print("ALL PAPERS PROCESSED")
                print("=" * 80)
                print(f"Total papers processed: {len(results)}")
                total_skills = sum(len(r.get("behavior_book", {})) for r in results)
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
