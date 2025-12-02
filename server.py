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
        self.skill_store = {}  # Aggregated skill store (compatible with behavior_book from client)
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
                self.model_name,
                trust_remote_code=True
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
                self.model_name,
                **model_kwargs
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
            max_new_tokens = max(int(input_token_count * 1.5), 4096)
            max_new_tokens = min(max_new_tokens, 32768)  # Cap at 32k tokens for encyclopedia chapters
            
            print(f"Input tokens: {input_token_count}, Max new tokens: {max_new_tokens}")
            
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
                outputs[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"Error calling model: {e}")
            return f"[Error] Model generation failed: {str(e)}"

    def collect_skill_books(self, json_files: Optional[List[str]] = None) -> Dict:
        """
        Step 1: Collect skill books from multiple client result JSON files.
        
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
            json_files = [str(f) for f in json_files if "metadata" not in f.name.lower()]
        
        collected_books = {}
        all_skills = {}
        problems = []
        
        print(f"Collecting skill books from {len(json_files)} files...")
        
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Extract skill book (client.py uses "behavior_book" key but contains skills)
                skill_book = data.get("behavior_book", {})
                if not skill_book:
                    # Try alternative keys
                    skill_book = data.get("behaviors", {}) or data.get("skills", {})
                
                if skill_book:
                    # Store with filename as key
                    filename = Path(json_file).stem
                    collected_books[filename] = {
                        "problem": data.get("problem", "Unknown"),
                        "skill_book": skill_book,  # Store as skill_book for clarity
                        "behavior_book": skill_book,  # Keep for compatibility
                        "solution": data.get("solution", ""),
                        "reflection": data.get("reflection", ""),
                    }
                    
                    # Aggregate all skills
                    all_skills.update(skill_book)
                    problems.append(data.get("problem", "Unknown"))
                    
                    print(f"  Collected {len(skill_book)} skills from {filename}")
                    
            except Exception as e:
                print(f"  Warning: Failed to read {json_file}: {e}")
        
        # Create aggregated skill store
        self.skill_store = all_skills
        
        step_result = {
            "step": 1,
            "name": "Collect Skill Books",
            "files_processed": len(collected_books),
            "total_skills": len(all_skills),
            "collected_books": collected_books,
            "skill_store": self.skill_store,
            "behavior_bookstore": self.skill_store,  # Keep for compatibility
            "problems": problems,
            "timestamp": time.time(),
        }
        
        self.aggregation_steps.append(step_result)
        print(f"\nCollected {len(all_skills)} unique skills from {len(collected_books)} files")
        
        return step_result

    def _get_skill_aggregation_prompt(self, raw_skill_store: Dict) -> str:
        """Get the prompt for intelligently aggregating similar skills"""
        # Format all skills for the prompt
        skills_text = "\n".join([
            f"- {name}: {description}"
            for name, description in raw_skill_store.items()
        ])
        
        prompt = f"""
### Input
Raw Skill Collection (from multiple sources):
{skills_text}

### Task: Intelligent Skill Aggregation
You are analyzing a collection of skills extracted from multiple problem-solving processes. Your task is to intelligently aggregate these skills by:

1. **Identifying Similar/Duplicate Skills:**
   - Find skills that are essentially the same or very similar (different names but same concept)
   - Identify skills that are variations or refinements of the same core technique
   - Detect skills that overlap significantly in their application

2. **Merging Similar Skills:**
   - For similar skills, create a unified skill with:
     * A canonical name (prefer the most descriptive or commonly used name)
     * A comprehensive description that captures the essence of all variations
     * Clear indication of when and how to apply it
   - Preserve important nuances from different variations
   - Remove true duplicates while keeping complementary aspects

3. **Organizing Distinct Skills:**
   - Keep skills that are genuinely distinct
   - Ensure each skill in the final set is unique and valuable
   - Maintain clarity about what each skill does and when to use it

### Output Format
Provide a JSON object with two keys:

1. **"aggregation_reasoning"**: A detailed explanation of:
   - Which skills you identified as similar/duplicate
   - How you merged them and why
   - What the final aggregated skill set represents

2. **"aggregated_skills"**: A JSON object where:
   - Keys are skill names (must start with `skill_`)
   - Values are comprehensive skill descriptions
   - Each skill should be a single line description
   - The set should be deduplicated and intelligently merged

Example format:
```json
{{
  "aggregation_reasoning": "I identified that skill_pattern_matching and skill_recognize_patterns are essentially the same concept... I merged skill_math_formula and skill_apply_formula into a unified skill_apply_mathematical_formula...",
  "aggregated_skills": {{
    "skill_pattern_matching": "Recognize and apply patterns in problem structures to identify solution approaches.",
    "skill_apply_mathematical_formula": "Apply relevant mathematical formulas with proper context and variable substitution.",
    ...
  }}
}}
```

Now, analyze and aggregate the skills:
"""
        return prompt

    def _step_skill_aggregation(self, raw_skill_store: Dict) -> Dict:
        """Step 1.5: Intelligently aggregate similar skills with reasoning"""
        prompt = self._get_skill_aggregation_prompt(raw_skill_store)
        
        system_prompt = None
        response = self._call_model(prompt, system_prompt)
        print(f"Skill aggregation completed ({len(response)} characters)")

        # Parse the aggregated skills
        aggregated_skills = {}
        aggregation_reasoning = ""
        
        try:
            # Try multiple strategies to extract JSON
            # Strategy 1: Look for JSON code blocks
            json_code_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_code_block:
                json_str = json_code_block.group(1)
            else:
                # Strategy 2: Look for JSON object in the response
                json_match = re.search(r'\{[\s\S]*?"aggregated_skills"[\s\S]*?\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # Strategy 3: Try to find any JSON object
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                    else:
                        json_str = None
            
            if json_str:
                # Clean up the JSON string
                json_str = json_str.strip()
                # Remove any trailing commas before closing braces
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                
                result = json.loads(json_str)
                aggregated_skills = result.get("aggregated_skills", {})
                aggregation_reasoning = result.get("aggregation_reasoning", "")
                
                # If aggregated_skills is empty, try to get skills directly
                if not aggregated_skills and isinstance(result, dict):
                    # Maybe the response structure is different
                    for key in result.keys():
                        if "skill" in key.lower() or isinstance(result[key], dict):
                            potential_skills = result[key]
                            if isinstance(potential_skills, dict):
                                # Check if it looks like a skill dictionary
                                if all(k.startswith("skill_") for k in list(potential_skills.keys())[:3]):
                                    aggregated_skills = potential_skills
                                    break
                
                if aggregated_skills:
                    # Update the skill store with aggregated skills
                    self.skill_store = aggregated_skills
                    print(f"Successfully parsed {len(aggregated_skills)} aggregated skills")
                else:
                    print("Warning: No aggregated_skills found in JSON. Using raw skills.")
                    aggregated_skills = raw_skill_store
                    self.skill_store = raw_skill_store
            else:
                print("Warning: Could not find JSON in aggregation response. Using raw skills.")
                print("Response preview:", response[:200] + "..." if len(response) > 200 else response)
                aggregated_skills = raw_skill_store
                self.skill_store = raw_skill_store
        except json.JSONDecodeError as e:
            print(f"Warning: JSON decode error: {e}")
            print("Attempting to extract skills manually...")
            # Try to extract skills manually using pattern matching
            skill_pattern = r'["\']?(skill_\w+)["\']?\s*[:=]\s*["\']([^"\']+)["\']'
            matches = re.findall(skill_pattern, response)
            if matches:
                for name, desc in matches:
                    aggregated_skills[name] = desc.strip()
                if aggregated_skills:
                    self.skill_store = aggregated_skills
                    print(f"Extracted {len(aggregated_skills)} skills using pattern matching")
                else:
                    aggregated_skills = raw_skill_store
                    self.skill_store = raw_skill_store
            else:
                print("Could not extract skills manually. Using raw skills.")
                aggregated_skills = raw_skill_store
                self.skill_store = raw_skill_store
        except Exception as e:
            print(f"Warning: Error parsing aggregated skills: {e}. Using raw skills.")
            aggregated_skills = raw_skill_store
            self.skill_store = raw_skill_store

        step_result = {
            "step": 1.5,
            "name": "Intelligent Skill Aggregation",
            "prompt": prompt,
            "response": response,
            "aggregation_reasoning": aggregation_reasoning,
            "raw_skills_count": len(raw_skill_store),
            "aggregated_skills_count": len(aggregated_skills),
            "aggregated_skills": aggregated_skills,
            "timestamp": time.time(),
        }

        self.aggregation_steps.append(step_result)
        print(f"\nAggregated {len(raw_skill_store)} raw skills into {len(aggregated_skills)} unique skills")
        
        return step_result

    def _get_reflection_prompt(self, skill_store: Dict) -> str:
        """Get the Reflection Prompt for analyzing behaviors and improving reasoning"""
        # Format skills for the prompt
        skills_text = "\n".join([
            f"- {name}: {description}"
            for name, description in skill_store.items()
        ])
        
        prompt = f"""
### Input  
Skill Store (text):
{skills_text}

### Context: Behaviors vs Skills
Following the framework of generative models as complex systems science, we distinguish:
- **Behaviors**: Observable patterns in how language models reason and solve problems (e.g., "tends to copy input patterns", "uses step-by-step decomposition", "applies domain-specific heuristics")
- **Skills**: Reusable knowledge, methods, or techniques that can be extracted and stored (what we keep in the encyclopedia)

### Task  
Analyze the reasoning behaviors that emerge from the application of these skills, and identify how behaviors can improve the reasoning process. For each skill, analyze:

#### 1. Observable Behaviors  
- What **behaviors** (observable patterns) does using this skill typically produce in reasoning?  
- How does the model behave differently when this skill is applied vs. when it's not?  
- What patterns in the reasoning process indicate this skill is being used effectively?

#### 2. Behavior-Based Strengths & Weaknesses  
- What problems does this skill address effectively, and what behaviors demonstrate this?  
- What are the limitations or failure modes, and what behaviors signal these limitations?  
- Under what circumstances do behaviors indicate the skill is failing or inappropriate?

#### 3. Behavioral Relationships & Interactions  
- How do behaviors from different skills interact or conflict?  
- What behavioral patterns emerge when skills are combined?  
- Are there complementary behaviors that suggest skills should be used together?

#### 4. Behavior-Guided Improvements  
- What **new behaviors** could improve the reasoning process for problems requiring these skills?  
- How can we modify or extend these skills to produce better reasoning behaviors?  
- What behavioral patterns are missing that would strengthen the overall reasoning capability?

### Output Format  
- Use clear section headings for each of the four analysis dimensions above.  
- For each skill, identify the behaviors it produces and how those behaviors can guide reasoning improvements.  
- Focus on observable patterns (behaviors) that can help us understand and improve reasoning, while keeping skills as the stored knowledge.
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

    def _get_thinking_prompt(self, reflection: str, skill_store: Dict) -> str:
        """Get the Thinking Prompt for generating an Encyclopedia Chapter"""
        # Format skills for the prompt
        skills_text = "\n".join([
            f"- {name}: {description}"
            for name, description in skill_store.items()
        ])
        
        prompt = f"""
You are an expert editor assembling a comprehensive **"Encyclopedia Chapter"** about problem-solving skills, informed by behavioral analysis.  

Input:  
- Skill Store (text):  
  {skills_text}  
- Reflection on Behaviors and Skills (text):  
  {reflection}  

### Framework: Behaviors Guide Skills
Following complex systems science principles, we use **behaviors** (observable reasoning patterns) to understand and improve reasoning, while storing **skills** (reusable knowledge) in the encyclopedia.

Definition:  
- **Skills**: Generalizable methods, strategies, or techniques stored in the encyclopedia  
- **Behaviors**: Observable patterns in reasoning that emerge when skills are applied (used to improve reasoning, not stored)

Task:  
Write a full Encyclopedia Chapter that:

1. Organizes the skills into logical **categories or themes** based on the behavioral patterns identified in the reflection.  
2. For each skill, explains it clearly with:
   - Context and examples of use
   - **Behaviors** that indicate the skill is being applied effectively
   - How to recognize when the skill should be used based on observable reasoning patterns
3. Highlights **relationships** among skills based on how their behaviors interact (complementary, conflicting, or sequential behaviors).  
4. Provides **guidance** on when and how to use each skill, informed by the behavioral analysis:
   - Use-cases where the skill produces effective reasoning behaviors
   - Behavioral indicators that suggest the skill is appropriate
5. Notes **best practices** and **pitfalls** based on behaviors:
   - What behaviors signal successful application
   - What behaviors indicate the skill is failing or misapplied
6. Presents a **coherent narrative** that connects skills to behaviors, helping practitioners understand both what to know (skills) and how to recognize effective reasoning (behaviors).

### Important:
- Store **skills** in the encyclopedia (reusable knowledge)
- Use **behaviors** to guide when and how to apply skills (observable patterns)
- The chapter should help readers both understand skills and recognize the behaviors that indicate effective reasoning  

Output Format:  
**IMPORTANT: Output ONLY the JSON object. Do NOT include any explanations, thinking process, or commentary before or after the JSON.**

Produce a single JSON object with the following schema:

```json
{{
  "chapter_title": "Skills Encyclopedia",
  "categories": [
    {{
      "category_name": "...",
      "skills": [
        {{
          "skill_name": "...",
          "description": "...",
          "behaviors": ["observable pattern 1", "observable pattern 2"],
          "use_cases": ["...","..."],
          "best_practices": ["...","..."],
          "pitfalls": ["...","..."],
          "related_skills": ["...","..."]
        }},
        ...
      ]
    }},
    ...
  ]
}}
```

Output ONLY the JSON object, nothing else.
"""
        
        return prompt

    def _step_encyclopedia_chapter(self, reflection: str, skill_store: Dict) -> Dict:
        """Step 3: Generate Encyclopedia Chapter from reflection and skill store"""
        prompt = self._get_thinking_prompt(reflection, skill_store)
        
        system_prompt = None
        response = self._call_model(prompt, system_prompt)
        print(f"Encyclopedia Chapter generated ({len(response)} characters)")

        # Extract only JSON content, removing any explanatory text
        json_content = self._extract_json_only(response)
        
        step_result = {
            "step": 3,
            "name": "Encyclopedia Chapter Generation",
            "prompt": prompt,
            "response": response,
            "chapter": json_content,  # Store only JSON content
            "timestamp": time.time(),
        }

        self.aggregation_steps.append(step_result)
        return step_result

    def _get_encyclopedia_prompt(self, existing_encyclopedia: str, new_chapter: str) -> str:
        """Get the Encyclopedia Prompt for synthesizing the complete Encyclopedia"""
        prompt = f"""
You are a knowledge-base curator maintaining a comprehensive Encyclopedia of problem-solving skills, organized through behavioral analysis.

Input:
- Existing Encyclopedia (may be "[No existing encyclopedia – this is the first chapter]"):  
  {existing_encyclopedia}  
- New Encyclopedia Chapter to integrate:  
  {new_chapter}

### Framework Reminder
- **Skills**: Store reusable knowledge in the encyclopedia (what we keep)
- **Behaviors**: Observable reasoning patterns that guide when/how to use skills (used to improve reasoning, not stored)

Task:
Produce the updated, complete Encyclopedia by merging the new chapter into the existing one. Your output should satisfy:

1. **Integration & Structure**  
   - Combine existing content and new chapter content into a unified structure.  
   - Preserve or adapt the existing structure when possible; if no existing encyclopedia, build a full structure from the new chapter.  
   - Ensure the final structure is hierarchical, coherent, and navigable (e.g., major sections / categories, sub-sections, skills entries).  
   - Organize based on behavioral patterns where applicable (skills with similar behaviors grouped together).

2. **Conflict & Redundancy Resolution**  
   - Detect and resolve duplicates — if the same skill appears in both old and new content, merge them thoughtfully rather than duplicating.  
   - If there are conflicting definitions or descriptions, reconcile them by merging the strengths of both or creating a consolidated, consistent version.  
   - When merging, preserve behavioral information that helps guide skill application.

3. **Cross-References & Relationships**  
   - Update cross-references: ensure that citations, internal links or references between skills, sections, or categories remain correct.  
   - Update relationships between skills based on behavioral interactions identified in the reflection.  
   - Maintain behavioral indicators that help recognize when skills should be applied.

4. **Consistency in Style & Format**  
   - Use a uniform style, naming convention, and formatting for all entries (section headers, skill naming, categories, etc.).  
   - Maintain consistent terminology distinguishing skills (stored knowledge) from behaviors (observable patterns).  
   - Ensure behavioral information is consistently formatted across all skills.

5. **Comprehensiveness & Organization**  
   - Ensure that all **skills** from both existing encyclopedia and new chapter are included (unless a duplicate is merged).  
   - Organize skills into logical categories or themes informed by behavioral patterns (e.g. "Mathematical Techniques", "Reasoning Strategies", "Verification & Error Checking", etc.).  
   - Provide a Table-of-Contents (top-level index) listing all categories and the skills they contain.  
   - Include behavioral indicators for skills where available to guide effective application.  

Output Format:
Produce the entire updated Encyclopedia in **JSON format**, using a nested structure. An example schema:

```json
{{
  "title": "Problem-Solving Skills Encyclopedia",
  "version": 2,
  "table_of_contents": [
     {{
       "category_name": "...",
       "skills": ["skill_name1", "skill_name2", ...]
     }},
     ...
  ],
  "categories": [
     {{
       "category_name": "...",
       "skills": [
         {{
           "skill_name": "...",
           "description": "...",
           "behaviors": ["observable pattern 1", "observable pattern 2"],
           "use_cases": ["...","..."],
           "best_practices": ["...","..."],
           "pitfalls": ["...","..."],
           "related_skills": ["...","..."]
         }},
         ...
       ]
     }},
     ...
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
            json_code_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if json_code_block:
                return json_code_block.group(1).strip()
            
            # Strategy 2: Look for JSON object (find the first { and matching })
            # Count braces to find the complete JSON object
            start_idx = text.find('{')
            if start_idx != -1:
                brace_count = 0
                for i in range(start_idx, len(text)):
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = text[start_idx:i+1]
                            # Validate it's valid JSON
                            json.loads(json_str)
                            return json_str.strip()
            
            # Strategy 3: Try to find any JSON object
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Clean up and validate
                json_str = json_str.strip()
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                json.loads(json_str)  # Validate
                return json_str
            
            # If no JSON found, return original (shouldn't happen)
            return text
        except (json.JSONDecodeError, AttributeError):
            # If extraction fails, return original text
            return text

    def _step_encyclopedia_synthesis(self, new_chapter: str) -> Dict:
        """Step 4: Synthesize complete Encyclopedia from existing and new chapter"""
        prompt = self._get_encyclopedia_prompt(self.encyclopedia, new_chapter)
        
        system_prompt = None
        response = self._call_model(prompt, system_prompt)
        print(f"Encyclopedia synthesized ({len(response)} characters)")

        # Extract only JSON content, removing any explanatory text
        json_content = self._extract_json_only(response)
        
        # Update the encyclopedia with only JSON content
        self.encyclopedia = json_content

        step_result = {
            "step": 4,
            "name": "Encyclopedia Synthesis",
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
        incremental: bool = False
    ) -> Dict:
        """
        Main method to aggregate skill books and build the Encyclopedia.
        
        Args:
            json_files: List of JSON file paths. If None, scans input_dir.
            incremental: If True, processes files incrementally. If False, processes all at once.
        
        Returns:
            Dictionary containing all aggregation steps and final encyclopedia.
        """
        # Step 1: Collect Skill Books
        print("\n" + "="*80)
        print("STEP 1: Collecting Skill Books")
        print("="*80)
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

        # Step 1.5: Intelligently Aggregate Similar Skills
        print("\n" + "="*80)
        print("STEP 1.5: Intelligently Aggregating Similar Skills")
        print("="*80)
        raw_skill_store = self.skill_store.copy()  # Keep raw skills for reference
        aggregation_result = self._step_skill_aggregation(raw_skill_store)
        time.sleep(1)

        if not self.skill_store:
            print("Warning: No skills after aggregation!")
            return {
                "error": "No skills after aggregation",
                "aggregation_steps": self.aggregation_steps,
            }

        # Step 2: Reflection on Skills
        print("\n" + "="*80)
        print("STEP 2: Generating Reflection on Skills")
        print("="*80)
        reflection_result = self._step_reflection(self.skill_store)
        reflection = reflection_result["response"]
        time.sleep(1)

        # Step 3: Generate Encyclopedia Chapter
        print("\n" + "="*80)
        print("STEP 3: Generating Encyclopedia Chapter")
        print("="*80)
        chapter_result = self._step_encyclopedia_chapter(reflection, self.skill_store)
        new_chapter = chapter_result["chapter"]
        time.sleep(1)

        # Step 4: Synthesize Complete Encyclopedia
        print("\n" + "="*80)
        print("STEP 4: Synthesizing Complete Encyclopedia")
        print("="*80)
        encyclopedia_result = self._step_encyclopedia_synthesis(new_chapter)

        # Compile results
        result = {
            "raw_skill_store": raw_skill_store,  # Original collected skills (before aggregation)
            "skill_store": self.skill_store,  # Aggregated skills (after intelligent merging)
            "behavior_bookstore": self.skill_store,  # Keep for compatibility
            "aggregation_reasoning": aggregation_result.get("aggregation_reasoning", ""),
            "collection_metadata": {
                "files_processed": collection_result.get("files_processed", 0),
                "total_skills_collected": collection_result.get("total_skills", 0),
                "total_skills_after_aggregation": len(self.skill_store),
                "problems": collection_result.get("problems", []),
                "collected_books": collection_result.get("collected_books", {}),
            },
            "reflection": reflection,
            "encyclopedia_chapter": new_chapter,
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
        
        print("\n" + "="*80)
        print("AGGREGATION COMPLETE")
        print("="*80)
        print(f"Total skills: {result.get('total_skills', result.get('total_behaviors', 0))}")
        print(f"Encyclopedia length: {len(result.get('encyclopedia', ''))} characters")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you have:")
        print("1. Run client.py to generate skill book JSON files")
        print("2. Installed required packages: pip install -r requirements.txt")
        print("3. For GPU support, ensure CUDA is properly installed")

