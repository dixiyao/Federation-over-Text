"""
Skill Aggregation Server
Implements the aggregation phase for building an Encyclopedia from multiple skill books.
Follows the pipeline: Collect Skill Books → Aggregate Skill Store → Reflection → Encyclopedia Chapter → Encyclopedia
"""

import argparse
import json
import os
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
Produce a single JSON object with the following schema:

```json
{
  "chapter_title": "Skills Encyclopedia",
  "categories": [
    {
      "category_name": "...",
      "skills": [
        {
          "skill_name": "...",
          "description": "...",
          "use_cases": ["...","..."],
          "best_practices": ["...","..."],
          "pitfalls": ["...","..."],
          "related_skills": ["...","..."]
        },
        ...
      ]
    },
    ...
  ]
}
"""
        
        return prompt

    def _step_encyclopedia_chapter(self, reflection: str, skill_store: Dict) -> Dict:
        """Step 3: Generate Encyclopedia Chapter from reflection and skill store"""
        prompt = self._get_thinking_prompt(reflection, skill_store)
        
        system_prompt = None
        response = self._call_model(prompt, system_prompt)
        print(f"Encyclopedia Chapter generated ({len(response)} characters)")

        step_result = {
            "step": 3,
            "name": "Encyclopedia Chapter Generation",
            "prompt": prompt,
            "response": response,
            "chapter": response,
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
{
  "title": "Problem-Solving Skills Encyclopedia",
  "version": 2,
  "table_of_contents": [
     {
       "category_name": "...",
       "skills": ["skill_name1", "skill_name2", ...]
     },
     ...
  ],
  "categories": [
     {
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
     },
     ...
  ]
}
```
"""
        
        return prompt

    def _step_encyclopedia_synthesis(self, new_chapter: str) -> Dict:
        """Step 4: Synthesize complete Encyclopedia from existing and new chapter"""
        prompt = self._get_encyclopedia_prompt(self.encyclopedia, new_chapter)
        
        system_prompt = None
        response = self._call_model(prompt, system_prompt)
        print(f"Encyclopedia synthesized ({len(response)} characters)")

        # Update the encyclopedia
        self.encyclopedia = response

        step_result = {
            "step": 4,
            "name": "Encyclopedia Synthesis",
            "prompt": prompt,
            "response": response,
            "encyclopedia": response,
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
            "skill_store": self.skill_store,
            "behavior_bookstore": self.skill_store,  # Keep for compatibility
            "collection_metadata": {
                "files_processed": collection_result.get("files_processed", 0),
                "total_skills_collected": collection_result.get("total_skills", 0),
                "problems": collection_result.get("problems", []),
                "collected_books": collection_result.get("collected_books", {}),
            },
            "reflection": reflection,
            "encyclopedia_chapter": new_chapter,
            "encyclopedia": self.encyclopedia,
            "aggregation_steps": self.aggregation_steps,
            "total_skills": len(self.skill_store),
            "total_behaviors": len(self.skill_store),  # Keep for compatibility
            "total_steps": len(self.aggregation_steps),
        }

        return result

    def save_results(self, result: Dict, output_dir: str = "build/log"):
        """Save aggregation results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save complete results as JSON
        json_path = os.path.join(output_dir, "encyclopedia_result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Save encyclopedia as text
        txt_path = os.path.join(output_dir, "encyclopedia.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("SKILL ENCYCLOPEDIA\n")
            f.write("="*80 + "\n\n")
            f.write(result.get("encyclopedia", "No encyclopedia generated."))
            f.write("\n\n" + "="*80 + "\n")
            f.write("SKILL STORE\n")
            f.write("="*80 + "\n\n")
            skill_store = result.get("skill_store", result.get("behavior_bookstore", {}))
            for name, desc in skill_store.items():
                f.write(f"{name}: {desc}\n")
        
        # Save skill store separately
        skill_store_path = os.path.join(output_dir, "skill_store.json")
        skill_store = result.get("skill_store", result.get("behavior_bookstore", {}))
        with open(skill_store_path, "w", encoding="utf-8") as f:
            json.dump(skill_store, f, indent=2, ensure_ascii=False)
        
        # Also save as behavior_bookstore.json for compatibility
        bookstore_path = os.path.join(output_dir, "behavior_bookstore.json")
        with open(bookstore_path, "w", encoding="utf-8") as f:
            json.dump(skill_store, f, indent=2, ensure_ascii=False)
        
        print("\nResults saved to:")
        print(f"  - {json_path}")
        print(f"  - {txt_path}")
        print(f"  - {skill_store_path}")
        print(f"  - {bookstore_path} (compatibility)")


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

