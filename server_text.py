"""
Text-Based Skill Aggregation Server
Implements a purely text-based approach for building an Encyclopedia from multiple skill books.
Uses LLM prompts to analyze relationships, merge skills, and extract general knowledge.

Pipeline:
1. Collect Skill Books → Aggregate Skill Store
2. Text-Based Profiling → Analyze relationships, merge same skills, cluster related skills
3. Knowledge Extraction → Extract general, fundamental knowledge that can derive collected skills
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


class TextBasedSkillAggregationServer:
    """
    Text-based server that aggregates skill books using LLM prompts
    to analyze relationships and extract general knowledge.
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        device: Optional[str] = None,
        input_dir: str = "math_output",
        use_gemini: bool = False,
        gemini_api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.input_dir = input_dir
        self.skill_store = {}  # Aggregated skill store
        self.encyclopedia = ""  # Final encyclopedia
        self.aggregation_steps = []
        self.skill_relationships = {}  # Text-based profiling of relationships

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

            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print("Model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")

    def _call_model(
        self, prompt: str, system_prompt: Optional[str] = None, max_new_tokens: int = 78632
    ) -> str:
        """Call the model with a prompt (HuggingFace or Gemini API)"""
        # Use Gemini API if configured
        if self.use_gemini:
            return self._call_gemini(prompt, system_prompt, max_new_tokens)
        
        # Otherwise use HuggingFace model
        self._load_model()

        try:
            # Prepare input
            if system_prompt:
                # For models that support system prompts
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                input_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                input_text = prompt

            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=65536,
            ).to(self.device)

            # Check if model name contains "DeepSeek-R1"
            is_deepseek_r1 = "DeepSeek-R1" in self.model_name

            # Generate response
            if is_deepseek_r1:
                # DeepSeek-R1 settings
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
            raise RuntimeError(f"Error calling model: {e}")

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
            # Filter out non-skill files (metadata, summary, results, etc.)
            # Process problem_*.json (from math_pipeline), paper_*.json (from client.py), or other skill files
            json_files = [
                str(f)
                for f in json_files
                if "metadata" not in f.name.lower()
                and "summary" not in f.name.lower()
                and "results" not in f.name.lower()
                and "encyclopedia" not in f.name.lower()
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
                skill_book_raw = (
                    data.get("behavior_book")
                    or data.get("behaviors")
                    or data.get("skills")
                )

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
                            skill_name = (
                                item.get("behavior")
                                or item.get("skill")
                                or item.get("name")
                            )
                            skill_desc = item.get("description") or item.get("desc")
                            if skill_name and skill_desc:
                                # Ensure skill name starts with "skill_" prefix
                                if not skill_name.startswith("skill_"):
                                    skill_name = f"skill_{skill_name}"
                                skill_book[skill_name] = skill_desc
                elif isinstance(data, dict) and not skill_book_raw:
                    # Check if data itself is a skill book (direct format from client.py)
                    # If all keys start with "skill_" or most keys are strings with descriptions
                    skill_keys = [k for k in data.keys() if isinstance(k, str) and k.startswith("skill_")]
                    if len(skill_keys) > 0 and len(skill_keys) >= len(data) * 0.8:
                        # This is likely a direct skill book dictionary
                        skill_book = data
                    elif all(isinstance(v, str) and len(v) > 20 for v in data.values() if isinstance(v, str)):
                        # All values are strings (likely descriptions), treat as skill book
                        skill_book = data

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


    def _get_text_profiling_prompt(self, skill_store: Dict) -> str:
        """Step 2: Prompt for text-based profiling of skill relationships
        
        Based on research in hierarchical skill learning, skill composition, and knowledge graphs.
        References:
        - Generalizable Hierarchical Skill Learning (GSL): Object-centric skill primitives
        - Skill Chaining: Composing skills for complex tasks
        - Knowledge Graph approaches: Relationship mapping and ontology construction
        """
        skills_text = "\n".join(
            [
                f"- {name}: {desc}"
                for name, desc in skill_store.items()
            ]
        )

        prompt = f"""
You are analyzing a collection of problem-solving skills to understand their relationships and structure, following principles from hierarchical skill learning and knowledge graph construction.

**Collected Skills ({len(skill_store)} total):**
{skills_text}

**Your Task:**
Analyze these skills and build a profiling of their relationships:

1. **Identify Skill Clusters**:
   Group related skills that share:
   - Common themes or problem-solving patterns
   - Same domain or application area
   - Similar approaches or techniques
   Skills in the same cluster should be related. Individual skills can be in separate clusters if they don't fit with others.

2. **Map Skill Relationships**:
   Build a relationship graph that records:
   - **Prerequisite relationships**: Skills that must be learned/used before others
   - **Composition relationships**: Skills that can be chained/composed together
   - **Alternative relationships**: Different approaches to the same problem
   - **Complementary relationships**: Skills that work well together
   - **Derivation relationships**: Skills derived from or based on others
   - **Similar relationships**: Skills that are similar but not identical

**Output Format (JSON) - Simplified:**
{{
  "clusters": [
    {{
      "cluster_id": 0,
      "cluster_name": "Domain/Theme Name",
      "skills": ["skill_name1", "skill_name2", "skill_name3"],
      "common_theme": "What these skills share in common",
      "domain": "mathematics/algebra/geometry/etc"
    }}
  ],
  "relationships": [
    {{
      "skill_a": "skill_name1",
      "skill_b": "skill_name2",
      "relationship_type": "prerequisite/complementary/alternative/similar/derived_from/composes_with",
      "description": "How these skills relate to each other"
    }}
  ]
}}

**Important Guidelines:**
- **Clusters**: Create meaningful clusters based on shared themes. Individual skills can be in separate clusters.
- **Relationships**: Record all important relationships - skills don't exist in isolation
- **Be thorough**: Map relationships between skills within clusters and across clusters
- **Preserve Original Skills**: Keep original skill names in clusters and relationships for traceability

**Output your analysis as JSON only:**
"""
        return prompt

    def _step_text_profiling(self, skill_store: Dict) -> Dict:
        """Step 2: Text-based profiling of skill relationships"""
        print("Building text-based profiling of skill relationships...")
        print(f"Analyzing {len(skill_store)} skills...")

        prompt = self._get_text_profiling_prompt(skill_store)
        system_prompt = None

        response = self._call_model(prompt, system_prompt, max_new_tokens=92768)
        print(f"Profiling response generated ({len(response)} characters)")

        # Extract JSON from response
        json_content = self._extract_json_only(response)

        try:
            profiling_data = json.loads(json_content)
            self.skill_relationships = profiling_data
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse profiling JSON: {e}")
            print("Using raw response as profiling data")
            self.skill_relationships = {"raw_response": response}

        step_result = {
            "step": 2,
            "name": "Text-Based Profiling",
            "prompt": prompt,
            "response": response,
            "profiling": self.skill_relationships,
            "timestamp": time.time(),
        }

        self.aggregation_steps.append(step_result)
        return step_result

    def _get_knowledge_extraction_prompt(
        self, skill_store: Dict, profiling: Dict, existing_encyclopedia: str = ""
    ) -> str:
        """Step 3: Prompt for extracting general, fundamental knowledge
        
        Based on research in:
        - Knowledge Distillation: Extracting higher-level abstractions
        - Hierarchical Skill Learning: Multi-level skill organization
        - Cross-domain Transfer: Identifying universal patterns
        - Skill Composition: Creating composable, reusable skills
        - Anthropic Skills: Composable, portable skill structure
        """
        
        # Format clusters
        clusters_text = ""
        if isinstance(profiling, dict) and "clusters" in profiling:
            clusters_text = "\n".join([
                f"- Cluster {cluster.get('cluster_id', '?')} ({cluster.get('cluster_name', 'unnamed')}): "
                f"{', '.join(cluster.get('skills', []))}"
                for cluster in profiling["clusters"]
            ])

        # Sample of original skills (for reference) - showing all skills
        sample_skills = "\n".join([
            f"- {name}: {desc}"
            for name, desc in skill_store.items()
        ])

        # Format all skills from skill_store (for reference)
        all_skills_text = "\n".join([
            f"{name}: {desc}"
            for name, desc in skill_store.items()
        ])
        
        # Include full previous encyclopedia if available
        previous_encyclopedia_text = ""
        if existing_encyclopedia:
            try:
                # Try to parse as JSON to extract skills
                enc_data = json.loads(existing_encyclopedia)
                if isinstance(enc_data, dict):
                    # Extract skills if they're in a dictionary format
                    enc_skills = []
                    if "general_skills" in enc_data:
                        for skill in enc_data["general_skills"]:
                            if isinstance(skill, dict):
                                skill_name = skill.get("skill_name", "")
                                skill_desc = skill.get("description", "")
                                if skill_name and skill_desc:
                                    enc_skills.append(f"{skill_name}: {skill_desc}")
                    elif isinstance(enc_data, dict):
                        # If it's a flat dictionary of skills
                        for name, desc in enc_data.items():
                            if name.startswith("skill_"):
                                enc_skills.append(f"{name}: {desc}")
                    if enc_skills:
                        previous_encyclopedia_text = "\n".join(enc_skills)
                    else:
                        previous_encyclopedia_text = existing_encyclopedia
                else:
                    previous_encyclopedia_text = existing_encyclopedia
            except:
                # If not JSON, treat as plain text
                previous_encyclopedia_text = existing_encyclopedia

        prompt = f"""
You are extracting fundamental knowledge from a collection of problem-solving skills.

**Context:**
- Total skills collected from clients: {len(skill_store)}
- These skills were derived from solving specific problems (bottom-up approach)
- Your task is to extract multi-disciplinary, fundamental knowledge (top-down approach) which can be generalized to multi-domain problem-solving.
- The extracted knowledge should be able to DERIVE and GUIDE the use of the collected skills
- Original skills should NOT be forgotten - they remain available for direct use

**Previous Encyclopedia (Existing Skills):**
{previous_encyclopedia_text if previous_encyclopedia_text else "None - this is the first encyclopedia"}

**All Client Skills (New Skills to Integrate):**
{all_skills_text}

**Skill Clusters (from Step 2):**
{clusters_text if clusters_text else "None identified"}

**Relationships (from Step 2):**
{json.dumps(profiling.get("relationships", []), indent=2) if isinstance(profiling, dict) and "relationships" in profiling else "None identified"}

**Your Task:**
You have:
1. Previous encyclopedia (existing general skills)
2. All client skills (new specific skills from problem-solving)

1. **Knowledge Distillation - Extract Reusable Skill Primitives**:
   Following the principle of creating object-centric, transferable skill primitives:
   - Identify the core essence of each skill cluster
   - Extract reusable components that can be applied across contexts
   - Create skill primitives that serve as building blocks for composition
   - General enough to transfer, specific enough to be actionable

2. **Hierarchical Skill Learning - Multi-Level Organization**:
   Following GSL framework principles:
   - **Fundamental Level**: Core principles and theories that underlie multiple domains
   - **General Level**: Broad techniques applicable across related problem types
   - **Specific Level**: Domain-specific applications (preserve original skills here)
   - Each level should guide the level below it
   - Skills should be organized in a hierarchy where higher levels subsume lower levels

2. **Preserve Original Skills**:
   - Original skills (both from encyclopedia and clients) remain available
   - General skills should reference and work WITH original skills
   - General skills provide a framework for selecting and using original skills

3. **Better Skills**:
   - More fundamental and general than original skills
   - Can guide the use of original skills
   - Include cross-domain insights and patterns
   - Still actionable with clear step-by-step instructions

**Output Format (Simple - Same as client.py):**
Output a simple JSON object with skill names as keys and descriptions as string values, exactly like client.py format:

{{"skill_name": "description"}}

**Description Format (Must match client.py exactly):**
Each description must be a single string containing all the following sections, separated naturally in the text:

1. **When to use**: Detailed explanation of when to apply this skill, what problem types trigger it, and what conditions must be met. Be comprehensive and specific.

2. **Step-by-step**: Detailed, numbered steps (1) 2) 3) ...) that explain exactly how to apply this skill. Include specific techniques, formulas, methods, or approaches. Be concrete and actionable.

4. **Cross-Domain Transfer - Identify Universal Patterns**:
   Following research on transfer learning and domain adaptation:
   - **Cross-domain insights**: Patterns that appear across different fields (universal principles)
   - **In-domain patterns**: Patterns specific to a domain but general within it
   - **Shared insights**: Common principles that appear in multiple domains
   - Document which skills transfer well across domains

5. **Anthropic Skills Format - Composable and Portable**:
   Following Anthropic's skill structure:
   - Each general skill should have clear metadata (when to use, domains applicable)
   - Detailed instructions (step-by-step guidance)
   - Resources (which lower-level skills it guides/derives)
   - Composable: Can work with other skills
   - Portable: Applicable across different contexts

**Format Rules:**
- Use valid JSON format
- Each skill name must start with "skill_"
- Each description is a single string containing: "When to use:" and "Step-by-step:" sections (and optionally "Key insights:" and "Example:")
- Steps must be numbered: 1) 2) 3)
- Keep JSON simple - no nested objects, just key-value pairs
- Escape quotes in descriptions with backslash: \\"

**Example:**
{{
  "skill_transformerArchitecture": "When to use: When you need to model relationships and dependencies in sequential or structured data, especially when long-range dependencies are important. This fundamental neural network architecture applies across natural language processing, computer vision, time series analysis, graph neural networks, and multi-modal learning. Use transformer architecture when: you need to capture relationships between all elements simultaneously (self-attention), you're working with sequences of variable length, you need parallel processing of sequences, or when the problem involves understanding context and relationships. This skill is essential for modern AI applications including language models, image processing, code generation, and scientific computing. Step-by-step: 1) Understand the core components - identify the key elements: multi-head self-attention mechanism (allows the model to focus on different parts of the input simultaneously), position encoding (adds information about sequence order), feed-forward networks (processes attended information), and layer normalization with residual connections (enables deep networks) 2) Design the input representation - convert your data into embeddings (token embeddings for text, patch embeddings for images, node embeddings for graphs), add positional encodings to preserve sequence information, and prepare the input for the transformer blocks 3) Apply self-attention mechanism - compute attention scores between all pairs of elements in the sequence using query, key, and value matrices, allowing each element to attend to all others and capture long-range dependencies 4) Use multi-head attention - apply multiple attention heads in parallel, each learning different types of relationships (syntactic, semantic, positional), then concatenate and project the results 5) Process through feed-forward networks - apply position-wise feed-forward networks to each attended representation, typically using two linear transformations with an activation function in between 6) Stack transformer blocks - layer multiple transformer blocks (attention + feed-forward) with residual connections and layer normalization, allowing the model to learn hierarchical representations 7) Apply to your specific problem - adapt the architecture for your task: encoder-only for understanding (BERT), decoder-only for generation (GPT), encoder-decoder for translation/summarization, or custom architectures for specialized tasks. Key insights: The transformer architecture's power comes from its ability to model all pairwise relationships simultaneously through attention, enabling parallel processing and capturing long-range dependencies. This architecture has revolutionized NLP and is being applied to vision, audio, and other domains. The attention mechanism allows the model to learn which parts of the input are most relevant for each output, making it interpretable and powerful. This principle underlies modern language models, vision transformers, and many state-of-the-art AI systems.",
  "skill_taylorExpansionApproximation": "When to use: When you need to approximate complex functions, analyze local behavior, or simplify nonlinear problems. This fundamental technique applies across calculus, physics, engineering, numerical analysis, machine learning, and optimization. Use Taylor expansion when: functions are too complex to work with directly, you need to understand behavior near a point, you want to linearize nonlinear systems, or you need numerical approximations. This skill is essential for understanding derivatives, optimization, differential equations, and function approximation. Step-by-step: 1) Identify the function to expand and the point of expansion - choose the expansion point (often x=0 for Maclaurin series, or a point where the function is well-behaved) based on where you need the approximation to be accurate 2) Calculate derivatives at the expansion point - compute the function value and its derivatives (first, second, third, etc.) at the chosen point, either analytically or numerically 3) Construct the Taylor series - use the formula f(x) = f(a) + f'(a)(x-a) + f''(a)(x-a)²/2! + f'''(a)(x-a)³/3! + ... where a is the expansion point, continuing to the desired order of approximation 4) Determine the order needed - assess how many terms are required based on desired accuracy, convergence properties, and the range of x values you need to approximate 5) Apply the approximation - use the truncated series to approximate the function, solve equations, or analyze behavior, being mindful of the approximation error 6) Estimate error bounds - use the remainder term (Lagrange or integral form) to bound the approximation error and ensure it meets accuracy requirements 7) Verify and refine - check the approximation against known values or higher-order expansions, and refine if necessary. Key insights: Taylor expansion reveals local structure of functions and enables linearization of nonlinear problems. Lower-order terms often capture the most important behavior. This principle underlies calculus, numerical methods, physics (small angle approximations, perturbation theory), machine learning (gradient descent uses first-order Taylor expansion), and engineering (linearization of nonlinear systems). The expansion point choice is critical - expanding around a point where the function is smooth and well-behaved yields better approximations."
}}

**CRITICAL: Avoid Generic/Fallback Skills**
DO NOT create skills that are:
- Overly generic or universal (e.g., "general problem-solving", "systematic approach", "careful analysis")
- Too abstract without specific techniques or methods
- Applicable to "any problem" or "all challenges"
- Lacking concrete, domain-specific techniques
- Fallback skills that don't provide specific value

**What Makes a Good Skill:**
- Specific enough to have concrete techniques, formulas, or methods
- General enough to apply across related domains, but NOT to all domains
- Has clear, actionable steps with specific techniques
- Addresses a particular class of problems with identifiable characteristics
- Provides domain-specific insights or methods that can be transferred

**Examples of BAD skills (DO NOT CREATE):**
- "skill_fallback": "When to use: For any problem-solving task..." (TOO GENERIC)
- "skill_generalProblemSolving": "When to use: For all problems..." (TOO GENERIC)
- "skill_systematicApproach": "When to use: Always..." (TOO GENERIC)

**Examples of GOOD skills:**
- "skill_transformerArchitecture": Specific neural network architecture with concrete components
- "skill_taylorExpansionApproximation": Specific mathematical technique with clear formulas
- "skill_polynomialFactoring": Specific algebraic technique with concrete methods

**Critical Requirements (Based on Research):**
1. **Output Format**: Must be simple JSON with skill_name: description (exactly like client.py format)
2. **Description Format**: Must include "When to use:" and "Step-by-step:" sections in the description string (same as client.py)
3. **Knowledge Distillation**: Extract higher-level abstractions that preserve essential information while generalizing
4. **Hierarchical Organization**: Create skills at different levels (fundamental → general → specific) where higher levels guide lower levels
5. **Skill Composition**: Skills should be composable - can be chained together for complex problems (following Generative Skill Chaining)
6. **Cross-Domain Transfer**: Include cross-domain insights (universal patterns) and in-domain patterns in descriptions
7. **Reusable Primitives**: Extract reusable skill primitives that can be applied across contexts (object-centric representation)
8. **Better Skills**: Skills should be more general and fundamental than original skills, but still actionable and SPECIFIC
9. **Merge Skills**: Merge similar skills from previous encyclopedia and client skills into better, unified versions
10. **Preserve Original**: Original skills remain available - general skills should be able to guide/derive them
11. **Actionable**: Each skill must have detailed, numbered step-by-step instructions with SPECIFIC techniques
12. **Complete**: Include all necessary information in the description string (when to use, steps, insights, example)
13. **Portable**: Skills should be portable - applicable across different domains and contexts (Anthropic Skills principle)
14. **Specificity**: Skills must be specific enough to provide concrete value - avoid generic fallback skills
15. **Domain-Bounded**: Skills should apply to a class of related problems, not "all problems"

**Existing Encyclopedia (if any):**
{existing_encyclopedia if existing_encyclopedia else "None"}

**Output your extracted knowledge as JSON only:**
"""
        return prompt

    def _step_knowledge_extraction(
        self, skill_store: Dict, profiling: Dict, existing_encyclopedia: str = ""
    ) -> Dict:
        """Step 3: Extract general, fundamental knowledge"""
        print("Extracting general, fundamental knowledge...")

        prompt = self._get_knowledge_extraction_prompt(
            skill_store, profiling, existing_encyclopedia
        )
        system_prompt = None

        response = self._call_model(prompt, system_prompt, max_new_tokens=78632)
        print(f"Knowledge extraction response generated ({len(response)} characters)")

        # Extract JSON from response
        json_content = self._extract_json_only(response)

        # Update encyclopedia
        self.encyclopedia = json_content

        step_result = {
            "step": 3,
            "name": "Knowledge Extraction",
            "prompt": prompt,
            "response": response,
            "encyclopedia": json_content,
            "timestamp": time.time(),
        }

        self.aggregation_steps.append(step_result)
        return step_result

    def _extract_json_only(self, text: str) -> str:
        """Extract JSON content from response, removing any explanatory text"""
        # Try to find JSON in code blocks first
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()

        # Try to find JSON object
        start_idx = text.find("{")
        if start_idx != -1:
            # Find matching closing brace
            brace_count = 0
            in_string = False
            escape_next = False
            for i in range(start_idx, len(text)):
                char = text[i]
                if escape_next:
                    escape_next = False
                    continue
                if char == "\\":
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if not in_string:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            return text[start_idx : i + 1]
            # If no complete match, try last brace
            last_brace = text.rfind("}", start_idx)
            if last_brace != -1:
                return text[start_idx : last_brace + 1]

        return text

    def _load_existing_encyclopedia(self, output_dir: str) -> str:
        """Load existing encyclopedia from output directory if it exists"""
        encyclopedia_path = os.path.join(output_dir, "encyclopedia.txt")
        if os.path.exists(encyclopedia_path):
            try:
                with open(encyclopedia_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    print(
                        f"Loaded existing encyclopedia from {encyclopedia_path} ({len(content)} characters)"
                    )
                    return content
            except Exception as e:
                print(f"Warning: Could not load existing encyclopedia: {e}")
        return ""

    def aggregate_and_build_encyclopedia(
        self,
        json_files: Optional[List[str]] = None,
        output_dir: str = "math_output",
    ) -> Dict:
        """
        Main method to aggregate skill books and build the Encyclopedia using text-based approach.

        Args:
            json_files: List of JSON file paths. If None, scans input_dir.
            output_dir: Output directory to check for existing encyclopedia.

        Returns:
            Dictionary containing all aggregation steps and final encyclopedia.
        """
        # Step 1: Collect Skill Books
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

        # Step 2: Text-Based Profiling
        print("\n" + "=" * 80)
        print("STEP 2: Text-Based Profiling of Skill Relationships")
        print("=" * 80)
        profiling_result = self._step_text_profiling(self.skill_store)
        time.sleep(1)

        # Step 3: Knowledge Extraction
        print("\n" + "=" * 80)
        print("STEP 3: Extracting General, Fundamental Knowledge")
        print("=" * 80)
        
        # Load existing encyclopedia if available
        existing_encyclopedia = self._load_existing_encyclopedia(output_dir)
        
        extraction_result = self._step_knowledge_extraction(
            self.skill_store,
            self.skill_relationships,
            existing_encyclopedia,
        )

        # Combine all results
        result = {
            "collection": collection_result,
            "profiling": profiling_result,
            "extraction": extraction_result,
            "aggregation_steps": self.aggregation_steps,
            "encyclopedia": self.encyclopedia,
            "skill_store": self.skill_store,  # Preserve original skills
            "skill_relationships": self.skill_relationships,
        }

        return result

    def save_results(self, result: Dict, output_dir: str = "math_output"):
        """Save only encyclopedia.json with format {"skill_name": "description"}"""
        os.makedirs(output_dir, exist_ok=True)
        encyclopedia_path = os.path.join(output_dir, "encyclopedia.json")

        # Parse encyclopedia JSON string and save as formatted JSON
        try:
            encyclopedia_dict = json.loads(self.encyclopedia)
            with open(encyclopedia_path, "w", encoding="utf-8") as f:
                json.dump(encyclopedia_dict, f, indent=2, ensure_ascii=False)
            print(f"Encyclopedia saved to: {encyclopedia_path}")
        except json.JSONDecodeError:
            # If not valid JSON, try to extract JSON from the string
            json_content = self._extract_json_only(self.encyclopedia)
            if json_content:
                encyclopedia_dict = json.loads(json_content)
                with open(encyclopedia_path, "w", encoding="utf-8") as f:
                    json.dump(encyclopedia_dict, f, indent=2, ensure_ascii=False)
                print(f"Encyclopedia saved to: {encyclopedia_path}")
            else:
                print(f"Warning: Could not parse encyclopedia as JSON. Saving raw content.")
                with open(encyclopedia_path, "w", encoding="utf-8") as f:
                    f.write(self.encyclopedia)
                print(f"Encyclopedia saved to: {encyclopedia_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Text-Based Skill Aggregation Server - Build Encyclopedia using LLM prompts"
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default="math_output",
        help="Input directory containing skill JSON files (default: math_output)",
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
        help="Output directory for encyclopedia (default: math_output)",
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

    args = parser.parse_args()

    # Create server
    server = TextBasedSkillAggregationServer(
        model_name=args.model, device=args.device, input_dir=args.input_dir,
        use_gemini=args.use_gemini, gemini_api_key=args.gemini_api_key
    )

    try:
        # Aggregate and build encyclopedia
        result = server.aggregate_and_build_encyclopedia(output_dir=args.output_dir)

        # Save results
        server.save_results(result, output_dir=args.output_dir)

        print("\n" + "=" * 80)
        print("AGGREGATION COMPLETE")
        print("=" * 80)
        print(f"Total skills collected: {len(server.skill_store)}")
        print(f"Encyclopedia length: {len(server.encyclopedia)} characters")
        print(f"Output directory: {args.output_dir}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        print("\nMake sure you have:")
        print("1. Installed required packages: pip install -r requirements.txt")
        print("2. Skill JSON files in the input directory")
        print("3. For GPU support, ensure CUDA is properly installed")

