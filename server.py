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
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
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
        
        # Embedding model for knowledge graph (loaded lazily)
        self.embedding_model = None
        self.embedding_model_name = "BAAI/bge-base-en-v1.5"

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
            max_new_tokens = 78632  # 78k tokens for encyclopedia chapters

            print(
                f"Input tokens: {input_token_count}, Max new tokens: {max_new_tokens}"
            )

            with torch.no_grad():
                # DeepSeek-R1 recommendations: temperature 0.5-0.7 (0.6 recommended)
                # Check if model name contains "DeepSeek-R1" to use recommended settings
                is_deepseek_r1 = "DeepSeek-R1" in self.model_name
                
                if is_deepseek_r1:
                    # DeepSeek-R1 recommended settings
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,  # Recommended for DeepSeek-R1
                        do_sample=True,
                        top_p=0.9,  # Recommended for DeepSeek-R1
                        repetition_penalty=1.2,
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
                        repetition_penalty=1.2,
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

    def _load_embedding_model(self):
        """Load the embedding model for knowledge graph clustering"""
        if self.embedding_model is not None:
            return
        
        try:
            print(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)
            print("Embedding model loaded successfully!")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model {self.embedding_model_name}: {e}")
    
    def _step_knowledge_graph_clustering(self, skill_store: Dict, r1: float = 0.9, r2: float = 0.4) -> Dict:
        """Step 2: Build knowledge graph and cluster skills using embeddings"""
        print("Building knowledge graph with embeddings...")
        
        # Load embedding model
        self._load_embedding_model()
        
        # Prepare skill texts for embedding
        skill_names = list(skill_store.keys())
        skill_texts = [f"{name}: {desc}" for name, desc in skill_store.items()]
        
        # Compute embeddings
        print(f"Computing embeddings for {len(skill_texts)} skills...")
        embeddings = self.embedding_model.encode(skill_texts, convert_to_numpy=True, show_progress_bar=True)
        
        # Normalize embeddings for cosine similarity
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Compute pairwise cosine similarity
        print("Computing pairwise cosine similarities...")
        similarity_matrix = cosine_similarity(embeddings_norm)
        
        # Normalize similarity to [0, 1] range (cosine similarity is already in [-1, 1])
        # Map from [-1, 1] to [0, 1]
        similarity_matrix = (similarity_matrix + 1) / 2
        
        # Build graph: same skills (r1 threshold) and linked skills (r2 threshold)
        same_skills = defaultdict(set)  # Groups of identical skills
        graph_edges = []  # Edges for clustering
        skill_to_group = {}  # Map skill to its same_skills group
        
        print(f"Building graph with thresholds: r1={r1} (same skill), r2={r2} (linked)")
        for i in range(len(skill_names)):
            for j in range(i + 1, len(skill_names)):
                sim = similarity_matrix[i][j]
                
                if sim >= r1:
                    # Same skill - merge into group
                    if i not in skill_to_group and j not in skill_to_group:
                        # Create new group
                        group_id = len(same_skills)
                        same_skills[group_id].add(i)
                        same_skills[group_id].add(j)
                        skill_to_group[i] = group_id
                        skill_to_group[j] = group_id
                    elif i in skill_to_group:
                        # Add j to i's group
                        group_id = skill_to_group[i]
                        same_skills[group_id].add(j)
                        skill_to_group[j] = group_id
                    elif j in skill_to_group:
                        # Add i to j's group
                        group_id = skill_to_group[j]
                        same_skills[group_id].add(i)
                        skill_to_group[i] = group_id
                elif sim >= r2:
                    # Linked skills - add edge
                    graph_edges.append((i, j, sim))
        
        # Build adjacency list for clustering
        adjacency = defaultdict(set)
        for i, j, sim in graph_edges:
            adjacency[i].add(j)
            adjacency[j].add(i)
        
        # Cluster skills within 2 hops
        print("Clustering skills within 2 hops...")
        clusters = []
        visited = set()
        
        def get_2hop_neighbors(node: int) -> Set[int]:
            """Get all nodes within 2 hops of a given node"""
            neighbors = {node}
            # 1-hop neighbors
            if node in adjacency:
                neighbors.update(adjacency[node])
            # 2-hop neighbors
            for neighbor in list(neighbors):
                if neighbor in adjacency:
                    neighbors.update(adjacency[neighbor])
            return neighbors
        
        for i in range(len(skill_names)):
            if i in visited:
                continue
            
            # Get 2-hop cluster
            cluster_nodes = get_2hop_neighbors(i)
            cluster_nodes = {n for n in cluster_nodes if n not in visited}
            
            if len(cluster_nodes) > 1:  # Only create cluster if more than 1 node
                clusters.append(cluster_nodes)
                visited.update(cluster_nodes)
        
        # Convert to skill names
        same_skills_groups = {
            group_id: [skill_names[i] for i in group]
            for group_id, group in same_skills.items()
        }
        
        clusters_named = [
            [skill_names[i] for i in cluster]
            for cluster in clusters
        ]
        
        print(f"Found {len(same_skills_groups)} groups of identical skills")
        print(f"Found {len(clusters_named)} clusters")
        
        step_result = {
            "step": 2,
            "name": "Knowledge Graph Clustering",
            "same_skills_groups": same_skills_groups,
            "clusters": clusters_named,
            "similarity_matrix": similarity_matrix.tolist(),
            "graph_edges": [(skill_names[i], skill_names[j], float(sim)) for i, j, sim in graph_edges],
            "timestamp": time.time(),
        }
        
        self.aggregation_steps.append(step_result)
        return step_result

    def _get_cluster_summary_prompt(
        self, same_skills_groups: Dict, clusters: List[List[str]], skill_store: Dict, existing_encyclopedia: str = "", r1: float = 0.9, r2: float = 0.4
    ) -> str:
        """Get the prompt for merging same skills and summarizing clusters"""
        # Format same skills groups
        same_skills_text = ""
        for group_id, skill_names in same_skills_groups.items():
            same_skills_text += f"\nGroup {group_id} (merge these into one skill):\n"
            for skill_name in skill_names:
                same_skills_text += f"  - {skill_name}: {skill_store.get(skill_name, 'N/A')}\n"
        
        # Format clusters
        clusters_text = ""
        for cluster_id, cluster_skills in enumerate(clusters):
            clusters_text += f"\nCluster {cluster_id}:\n"
            clusters_text += "  Skills in this cluster:\n"
            for skill_name in cluster_skills:
                clusters_text += f"    - {skill_name}: {skill_store.get(skill_name, 'N/A')}\n"
        
        # Get all unique skills (not in same_skills_groups)
        all_skill_names = set(skill_store.keys())
        merged_skill_names = set()
        for group in same_skills_groups.values():
            merged_skill_names.update(group)
        standalone_skills = all_skill_names - merged_skill_names
        
        standalone_text = ""
        if standalone_skills:
            standalone_text = "\nStandalone Skills (keep as-is):\n"
            for skill_name in standalone_skills:
                standalone_text += f"  - {skill_name}: {skill_store.get(skill_name, 'N/A')}\n"
        
        # Format existing encyclopedia section
        existing_encyclopedia_section = ""
        if existing_encyclopedia and existing_encyclopedia.strip():
            existing_encyclopedia_section = f"""
### Existing Encyclopedia (merge with this):
{existing_encyclopedia}

Note: You should merge new skills with the existing encyclopedia, combining similar skills and adding new ones.
"""
        
        prompt = f"""
You are building a comprehensive Skills Encyclopedia from clustered skills.
{existing_encyclopedia_section}
------------------------------------------------------------
TASK: Merge Same Skills and Summarize Clusters
------------------------------------------------------------

### Instructions:

1. MERGE WITH EXISTING ENCYCLOPEDIA (if provided)
   - If an existing encyclopedia is provided above, merge the new skills with it
   - If a skill in the new set already exists in the encyclopedia, merge them intelligently
   - Combine descriptions, keeping all important technical details from both
   - Add new skills that don't exist in the encyclopedia
   - Maintain the structure and organization of the existing encyclopedia when possible

2. MERGE SAME SKILLS (similarity >= {r1})
   - For each group of same skills below, merge them into ONE unified skill
   - These skills have cosine similarity >= {r1}, indicating they are essentially the same
   - Combine their descriptions, keeping all important technical details
   - Create a single, comprehensive skill that captures all variations
   - Skills should be very detailed and specific, going into deep technical details
   - DO NOT abstract to general principles - keep technical specificity

3. SUMMARIZE CLUSTERS (similarity {r2} to {r1})
   - For each cluster below, create a cluster summary with:
     * Cluster topic/theme (what connects these skills)
     * Potential use cases (when to use skills from this cluster)
     * **REQUIRED: List ALL skills in the cluster in the "skills" array** (DO NOT merge them - keep them separate)
   - Skills in a cluster have cosine similarity between {r2} and {r1}, meaning they are related but distinct
   - **CRITICAL: Each cluster MUST include a "skills" array containing all skills from that cluster with their full descriptions**
   - Keep them as separate entries within the cluster

4. KEEP STANDALONE SKILLS
   - Standalone skills (not in any group or cluster) should be kept as-is

### Same Skills Groups (MERGE these):
{same_skills_text}

### Clusters (SUMMARIZE these, but keep skills separate):
{clusters_text}

**IMPORTANT**: For each cluster above, you MUST include ALL skills from that cluster in the "skills" array. Each skill should have "skill_name" and "description" fields. Do NOT omit the skills array - it is REQUIRED.

### Standalone Skills:
{standalone_text}

### Output Format:
Output ONLY a JSON object with this structure:

```json
{{
  "title": "Problem-Solving Skills Encyclopedia",
  "merged_skills": [
    {{
      "skill_name": "...",
      "description": "...",
      "merged_from": ["skill1", "skill2", ...]
    }}
  ],
  "clusters": [
    {{
      "cluster_id": 0,
      "topic": "...",
      "use_cases": ["...", "..."],
      "skills": [
        {{
          "skill_name": "...",
          "description": "..."
        }},
        {{
          "skill_name": "...",
          "description": "..."
        }}
      ],
      "standalone": false
    }}
  ],
  "standalone_skills": [
    {{
      "skill_name": "...",
      "description": "..."
    }}
  ]
}}
```

**CRITICAL REQUIREMENTS**:
1. Every cluster MUST have a "skills" array containing ALL skills from that cluster
2. The "skills" array cannot be empty - it must contain at least one skill object
3. Each skill in the "skills" array must have both "skill_name" and "description" fields
4. Do NOT omit the "skills" array from any cluster - it is REQUIRED

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

    def _step_cluster_summary(self, same_skills_groups: Dict, clusters: List[List[str]], skill_store: Dict, existing_encyclopedia: str = "", r1: float = 0.9, r2: float = 0.4) -> Dict:
        """Step 3: Merge same skills and summarize clusters"""
        prompt = self._get_cluster_summary_prompt(same_skills_groups, clusters, skill_store, existing_encyclopedia, r1, r2)

        system_prompt = None
        response = self._call_model(prompt, system_prompt)
        print(f"Cluster summary generated ({len(response)} characters)")

        # Extract only JSON content, removing any explanatory text
        json_content = self._extract_json_only(response)

        # Update the encyclopedia with only JSON content
        self.encyclopedia = json_content

        step_result = {
            "step": 3,
            "name": "Cluster Summary and Skill Merging",
            "prompt": prompt,
            "response": response,
            "encyclopedia": json_content,  # Store only JSON content
            "timestamp": time.time(),
        }

        self.aggregation_steps.append(step_result)
        return step_result

    def _load_existing_encyclopedia(self, output_dir: str) -> str:
        """Load existing encyclopedia from output directory if it exists"""
        encyclopedia_path = os.path.join(output_dir, "encyclopedia.txt")
        if os.path.exists(encyclopedia_path):
            try:
                with open(encyclopedia_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    print(f"Loaded existing encyclopedia from {encyclopedia_path} ({len(content)} characters)")
                    return content
            except Exception as e:
                print(f"Warning: Could not load existing encyclopedia: {e}")
        return ""

    def aggregate_and_build_encyclopedia(
        self,
        json_files: Optional[List[str]] = None,
        incremental: bool = False,
        r1: float = 0.9,
        r2: float = 0.4,
        output_dir: str = "build/log",
    ) -> Dict:
        """
        Main method to aggregate skill books and build the Encyclopedia.

        Args:
            json_files: List of JSON file paths. If None, scans input_dir.
            incremental: If True, processes files incrementally. If False, processes all at once.
            r1: Threshold for same skills (cosine similarity >= r1 means same skill).
            r2: Threshold for linked skills (r2 <= similarity < r1 means linked).
            output_dir: Output directory to check for existing encyclopedia.

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

        # Step 2: Knowledge Graph Clustering
        print("\n" + "=" * 80)
        print("STEP 2: Knowledge Graph Clustering")
        print(f"Using thresholds: r1={r1} (same skill), r2={r2} (linked)")
        print("=" * 80)
        clustering_result = self._step_knowledge_graph_clustering(self.skill_store, r1=r1, r2=r2)
        same_skills_groups = clustering_result["same_skills_groups"]
        clusters = clustering_result["clusters"]
        time.sleep(1)

        # Step 3: Merge Same Skills and Summarize Clusters
        print("\n" + "=" * 80)
        print("STEP 3: Merging Same Skills and Summarizing Clusters")
        print("=" * 80)
        # Load existing encyclopedia if it exists
        existing_encyclopedia = self._load_existing_encyclopedia(output_dir)
        if existing_encyclopedia:
            print("Found existing encyclopedia - will merge with new skills")
        summary_result = self._step_cluster_summary(same_skills_groups, clusters, self.skill_store, existing_encyclopedia, r1=r1, r2=r2)

        # Compile results
        result = {
            "skill_store": self.skill_store,
            "behavior_bookstore": self.skill_store,  # Keep for compatibility
            "collection_metadata": {
                "files_processed": collection_result.get("files_processed", 0),
                "total_skills_collected": collection_result.get("total_skills_collected", 0),
                "unique_skills": collection_result.get("unique_skills", 0),
                "problems": collection_result.get("problems", []),
                "collected_books": collection_result.get("collected_books", {}),
            },
            "clustering": {
                "same_skills_groups": same_skills_groups,
                "clusters": clusters,
            },
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
        "--r1",
        type=float,
        default=0.9,
        help="Threshold r1 for same skills (cosine similarity >= r1 means same skill, default: 0.9)",
    )
    parser.add_argument(
        "--r2",
        type=float,
        default=0.4,
        help="Threshold r2 for linked skills (r2 <= similarity < r1 means linked, default: 0.4)",
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
        result = server.aggregate_and_build_encyclopedia(
            json_files=args.files,
            r1=args.r1,
            r2=args.r2,
            output_dir=args.output_dir
        )

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
