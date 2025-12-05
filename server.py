"""
Skill Aggregation Server
Implements the aggregation phase for building an Encyclopedia from multiple skill books.
Follows the pipeline: Collect Skill Books → Aggregate Skill Store → Reflection → Encyclopedia Chapter → Encyclopedia
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class SkillAggregationServer:
    """
    Server that aggregates skill books from multiple client results
    and constructs an Encyclopedia through reflection and synthesis.
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1",
        device: Optional[str] = None,
        input_dir: str = "build/log",
    ):
        self.model_name = model_name
        self.input_dir = input_dir
        self.skill_store = (
            {}
        )  # Aggregated skill store (compatible with behavior_book from client)
        self.encyclopedia = ""  # Final encyclopedia
        self.aggregation_steps = []

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
            # Use torch_dtype for from_pretrained
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
            # Tokenize input - large limit for encyclopedia, reflections, skill stores
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=65536,  # 64k tokens - enough for large skill stores and encyclopedia content
            ).to(self.device)

            # Calculate input token count for dynamic output sizing
            input_token_count = inputs["input_ids"].shape[1]

            # Generate response - dynamically size based on input, with larger limits for encyclopedia
            # Ensure max_new_tokens is larger than input tokens for comprehensive outputs
            max_new_tokens = 32768  # Cap at 32k tokens for encyclopedia chapters

            print(
                f"Input tokens: {input_token_count}, Max new tokens: {max_new_tokens}"
            )

            with torch.no_grad():
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

    def collect_skill_books(self, json_files: Optional[List[str]] = None) -> Dict:
        """
        Step 1: Collect skill books from multiple client result JSON files sequentially.

        Args:
            json_files: List of JSON file paths. If None, scans input_dir for JSON files.

        Returns:
            Dictionary containing collected skill books and metadata.
        """
        if json_files is None:
            # Scan input directory for JSON files
            input_path = Path(self.input_dir)
            json_files = list(input_path.glob("*.json"))
            # Filter out metadata.json and behavior_book.json files if needed
            json_files = [
                str(f) for f in json_files if "metadata" not in f.name.lower()
            ]

        print(f"Collecting skill books from {len(json_files)} files...")

        collected_books = {}
        all_skills = {}
        skill_counts = {}  # Track how many times each skill appears
        total_skills_count = 0  # Total count including duplicates
        problems = []

        # Read JSON files sequentially
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Extract skill book (client.py uses "behavior_book" key but contains skills)
                # Handle both dict and list formats
                skill_book_raw = data.get("behavior_book") or data.get("behaviors") or data.get("skills")
                
                # Convert list format to dict if needed
                skill_book = {}
                if isinstance(skill_book_raw, dict):
                    skill_book = skill_book_raw
                elif isinstance(skill_book_raw, list):
                    # Convert list of dicts to dict format
                    # Expected format: [{"behavior": "name", "description": "desc"}, ...]
                    for item in skill_book_raw:
                        if isinstance(item, dict):
                            # Try different key names
                            skill_name = item.get("behavior") or item.get("skill") or item.get("name")
                            skill_desc = item.get("description") or item.get("desc")
                            if skill_name and skill_desc:
                                # Ensure skill name starts with "skill_" prefix
                                if not skill_name.startswith("skill_"):
                                    skill_name = f"skill_{skill_name}"
                                skill_book[skill_name] = skill_desc
                
                if skill_book:
                    filename = Path(json_file).stem
                    collected_books[filename] = {
                        "problem": data.get("problem", "Unknown"),
                        "skill_book": skill_book,
                        "behavior_book": skill_book,  # Keep for compatibility
                        "solution": data.get("solution", ""),
                        "reflection": data.get("reflection", ""),
                    }
                    
                    # Count and aggregate all skills
                    total_skills_count += len(skill_book)
                    for skill_name, skill_desc in skill_book.items():
                        # Update skill store (keep latest description if duplicate)
                        all_skills[skill_name] = skill_desc
                        # Count occurrences
                        skill_counts[skill_name] = skill_counts.get(skill_name, 0) + 1
                    
                    problems.append(data.get("problem", "Unknown"))
                    
                    print(f"  Collected {len(skill_book)} skills from {filename}")
            except Exception as e:
                print(f"  Warning: Failed to read {json_file}: {e}")
                continue

        # Create aggregated skill store
        self.skill_store = all_skills

        step_result = {
            "step": 1,
            "name": "Collect Skill Books",
            "files_processed": len(collected_books),
            "total_skills_collected": total_skills_count,  # Total including duplicates
            "unique_skills": len(all_skills),  # Unique skill count
            "skill_counts": skill_counts,  # Count of each skill
            "collected_books": collected_books,
            "skill_store": self.skill_store,
            "behavior_bookstore": self.skill_store,  # Keep for compatibility
            "problems": problems,
            "timestamp": time.time(),
        }

        self.aggregation_steps.append(step_result)
        print(
            f"\nCollected {total_skills_count} total skills ({len(all_skills)} unique) from {len(collected_books)} files"
        )

        return step_result

    def _get_reflection_prompt(self, skill_store: Dict) -> str:
        """Get the Reflection Prompt for analyzing strengths and weaknesses"""
        # Format skills for the prompt
        skills_text = "\n".join(
            [f"- {name}: {description}" for name, description in skill_store.items()]
        )

        prompt = f"""
### Input  
Skill Collection:
{skills_text}

### Task  
Analyze each skill and identify:

1. **Strengths**: What problems does this skill address effectively? What are its key advantages and when does it work best?

2. **Weaknesses**: What are the limitations or failure modes of this skill? Under what circumstances does it fail or become inappropriate?

3. **Use Cases**: When should this skill be applied? What types of problems benefit most from this skill?

### Output Format  
Provide a clear analysis for each skill, focusing on strengths and weaknesses. Be concise and practical.
"""

        return prompt

    def _step_reflection(self, skill_store: Dict) -> Dict:
        """Step 2: Generate reflection on skill strengths and weaknesses"""
        prompt = self._get_reflection_prompt(skill_store)

        system_prompt = None
        response = self._call_model(prompt, system_prompt)
        print(f"Reflection generated ({len(response)} characters)")

        step_result = {
            "step": 2,
            "name": "Reflection on Skills",
            "prompt": prompt,
            "response": response,
            "timestamp": time.time(),
        }

        self.aggregation_steps.append(step_result)
        return step_result

    def _get_encyclopedia_aggregation_prompt(
        self, existing_encyclopedia: str, new_skills: Dict, reflection: str
    ) -> str:
        """Get the Encyclopedia Prompt for aggregating new skills with existing encyclopedia"""
        # Format new skills for the prompt
        new_skills_text = "\n".join(
            [f"- {name}: {description}" for name, description in new_skills.items()]
        )

        prompt = f"""
You are maintaining a comprehensive Encyclopedia of problem-solving skills.

Input:
- Existing Encyclopedia (may be empty if this is the first time):  
{existing_encyclopedia if existing_encyclopedia else "[No existing encyclopedia]"}
  
- New Skills to integrate:
{new_skills_text}

- Reflection on Strengths and Weaknesses:
{reflection}

------------------------------------------------------------
TASK: Build the updated Skill Encyclopedia
------------------------------------------------------------

Your goal is to merge, refine, and reorganize all skills into a clean, logically structured JSON encyclopedia.  
Follow all instructions exactly:

1. MERGE SKILLS (Deep reasoning required)
- Combine new skills with the existing encyclopedia.  
- If a skill already exists, merge intelligently by unifying strengths and fixing weaknesses described in the reflection.  
- Merge highly similar skills into a single generalized, powerful skill.  
- Abstract underlying principles so the final skill set is compact, non-redundant, and easy to reference.

2. ORGANIZE INTO CATEGORIES
- Group skills into meaningful, clearly separable categories (e.g., Reasoning, Planning, Creativity, Decision-Making, Reflection, Communication).  
- ONLY create categories that improve clarity.  
- Each category must contain non-overlapping skills.

3. UPDATE SKILLS USING REFLECTION
- Improve each skill's description using the reflection.  
- Add use cases, failure modes, corrections, and clearer heuristics.  
- Strengthen skills to be more actionable, robust, and generalizable.

4. STYLE REQUIREMENTS
- Be concise but information-dense.  

```json
{{
  "title": "Problem-Solving Skills Encyclopedia",
  "categories": [
    {{
      "category_name": "...",
      "skills": [
        {{
          "skill_name": "...",
          "description": "...",
          "use_cases": ["..."]
        }}
      ]
    }}
  ]
}}
```

Output ONLY the JSON object, nothing else.
"""

        return prompt

    def _extract_json_only(self, text: str) -> str:
        """Extract only JSON content from response, removing any explanatory text"""
        try:
            # Strategy 1: Look for JSON in code blocks
            json_code_block = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL
            )
            if json_code_block:
                return json_code_block.group(1).strip()

            # Strategy 2: Look for JSON object (find the first { and matching })
            # Count braces to find the complete JSON object
            start_idx = text.find("{")
            if start_idx != -1:
                brace_count = 0
                for i in range(start_idx, len(text)):
                    if text[i] == "{":
                        brace_count += 1
                    elif text[i] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = text[start_idx : i + 1]
                            # Validate it's valid JSON
                            json.loads(json_str)
                            return json_str.strip()

            # Strategy 3: Try to find any JSON object
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Clean up and validate
                json_str = json_str.strip()
                json_str = re.sub(r",\s*}", "}", json_str)
                json_str = re.sub(r",\s*]", "]", json_str)
                json.loads(json_str)  # Validate
                return json_str

            # If no JSON found, return original (shouldn't happen)
            return text
        except (json.JSONDecodeError, AttributeError):
            # If extraction fails, return original text
            return text

    def _step_encyclopedia_aggregation(self, new_skills: Dict, reflection: str) -> Dict:
        """Step 3: Aggregate new skills with existing encyclopedia"""
        prompt = self._get_encyclopedia_aggregation_prompt(
            self.encyclopedia, new_skills, reflection
        )

        system_prompt = None
        response = self._call_model(prompt, system_prompt)
        print(f"Encyclopedia aggregated ({len(response)} characters)")

        # Extract only JSON content, removing any explanatory text
        json_content = self._extract_json_only(response)

        # Update the encyclopedia with only JSON content
        self.encyclopedia = json_content

        step_result = {
            "step": 3,
            "name": "Encyclopedia Aggregation",
            "prompt": prompt,
            "response": response,
            "encyclopedia": json_content,  # Store only JSON content
            "timestamp": time.time(),
        }

        self.aggregation_steps.append(step_result)
        return step_result

    def aggregate_and_build_encyclopedia(
        self,
        json_files: Optional[List[str]] = None,
        incremental: bool = False,
    ) -> Dict:
        """
        Main method to aggregate skill books and build the Encyclopedia.

        Args:
            json_files: List of JSON file paths. If None, scans input_dir.
            incremental: If True, processes files incrementally. If False, processes all at once.

        Returns:
            Dictionary containing all aggregation steps and final encyclopedia.
        """
        # Step 1: Collect Skill Books (append all skills together)
        print("\n" + "=" * 80)
        print("STEP 1: Collecting Skill Books")
        print("=" * 80)
        collection_result = self.collect_skill_books(json_files)
        time.sleep(1)

        if not self.skill_store:
            files_processed = collection_result.get("files_processed", 0)
            print(f"Warning: No skills found in {files_processed} collected files!")
            return {
                "error": "No skills found",
                "files_processed": files_processed,
                "collection_result": collection_result,
                "aggregation_steps": self.aggregation_steps,
            }

        # Step 2: Reflection on Strengths and Weaknesses
        print("\n" + "=" * 80)
        print("STEP 2: Reflecting on Strengths and Weaknesses")
        print("=" * 80)
        reflection_result = self._step_reflection(self.skill_store)
        reflection = reflection_result["response"]
        time.sleep(1)

        # Step 3: Aggregate with Existing Encyclopedia
        print("\n" + "=" * 80)
        print("STEP 3: Aggregating with Existing Encyclopedia")
        print("=" * 80)
        encyclopedia_result = self._step_encyclopedia_aggregation(
            self.skill_store, reflection
        )

        # Compile results
        result = {
            "skill_store": self.skill_store,
            "behavior_bookstore": self.skill_store,  # Keep for compatibility
            "collection_metadata": {
                "files_processed": collection_result.get("files_processed", 0),
                "total_skills_collected": collection_result.get("total_skills", 0),
                "problems": collection_result.get("problems", []),
                "collected_books": collection_result.get("collected_books", {}),
            },
            "reflection": reflection,
            "encyclopedia": self.encyclopedia,  # Final encyclopedia containing aggregated skills
            "aggregation_steps": self.aggregation_steps,
            "total_skills": len(self.skill_store),
            "total_behaviors": len(self.skill_store),  # Keep for compatibility
            "total_steps": len(self.aggregation_steps),
        }

        return result

    def save_results(self, result: Dict, output_dir: str = "build/log"):
        """Save aggregation results - only the encyclopedia"""
        os.makedirs(output_dir, exist_ok=True)

        # Save only the encyclopedia (main output)
        encyclopedia_path = os.path.join(output_dir, "encyclopedia.txt")
        with open(encyclopedia_path, "w", encoding="utf-8") as f:
            f.write(result.get("encyclopedia", "No encyclopedia generated."))

        print("\nEncyclopedia saved to:")
        print(f"  - {encyclopedia_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Skill Aggregation Server - Build Encyclopedia from Skill Books"
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default="build/log",
        help="Directory containing client result JSON files (default: build/log)",
    )
    parser.add_argument(
        "-f",
        "--files",
        type=str,
        nargs="+",
        default=None,
        help="Specific JSON files to process (default: all JSON files in input-dir)",
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
        "-o",
        "--output-dir",
        type=str,
        default="build/log",
        help="Output directory for results (default: build/log)",
    )
    parser.add_argument(
        "-w",
        type=int,
        default=None,
        help="Number of parallel workers for reading JSON files (default: min(8, num_files))",
    )

    args = parser.parse_args()

    # Create server instance
    server = SkillAggregationServer(
        model_name=args.model,
        device=args.device,
        input_dir=args.input_dir,
    )

    try:
        # Run aggregation pipeline
        result = server.aggregate_and_build_encyclopedia(json_files=args.files)

        # Save results
        server.save_results(result, output_dir=args.output_dir)

        print("\n" + "=" * 80)
        print("AGGREGATION COMPLETE")
        print("=" * 80)
        print(
            f"Total skills: {result.get('total_skills', result.get('total_behaviors', 0))}"
        )
        print(f"Encyclopedia length: {len(result.get('encyclopedia', ''))} characters")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        print("\nMake sure you have:")
        print("1. Run client.py to generate skill book JSON files")
        print("2. Installed required packages: pip install -r requirements.txt")
        print("3. For GPU support, ensure CUDA is properly installed")
