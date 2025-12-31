"""
Generate Server - Inference using Encyclopedia

NOTE: This is a standalone CLI tool for querying the encyclopedia directly.
For use in pipelines, use client.py's solve_with_encyclopedia() method instead:
- math_pipeline.py uses client.solve_with_encyclopedia()
- math_domain.py uses client.solve_with_encyclopedia()

This file is kept for:
1. Command-line querying of the encyclopedia (see example_command.sh)
2. Standalone testing and demos
3. Direct API usage without the full pipeline
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import google.generativeai as genai

    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


class GenerateServer:
    """
    Simple inference server that uses the encyclopedia to answer queries.
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        device: Optional[str] = None,
        max_new_tokens: int = 98304,
        use_gemini: bool = False,
        gemini_api_key: Optional[str] = None,
        mode: str = "normal",
    ):
        self.model_name = model_name
        self.encyclopedia = ""
        self.encyclopedia_dict = {}  # For text mode (JSON format)
        self.max_new_tokens = max_new_tokens
        self.mode = mode  # "normal" for GraphRAG, "text" for encyclopedia.json

        # Gemini API support
        self.use_gemini = use_gemini
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if self.use_gemini:
            if not HAS_GEMINI:
                raise ImportError(
                    "google-generativeai is required for Gemini API. Install with: pip install google-generativeai"
                )
            if not self.gemini_api_key:
                raise ValueError(
                    "Gemini API key is required when use_gemini=True. Set GEMINI_API_KEY env var or pass gemini_api_key parameter."
                )
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel("gemini-3-pro-preview")

        # Model and tokenizer will be loaded lazily on first use (only for HuggingFace models)
        self.model = None
        self.tokenizer = None
        self.device = device or ("cuda" if self._check_cuda() else "cpu")

        # GraphRAG database (only for normal mode)
        self.graphrag_db = None
        self.graphrag_embeddings = None
        self.graphrag_graph = None
        self.embedding_model = None
        self.embedding_model_name = "BAAI/bge-base-en-v1.5"

        # System prompt for text mode
        self.system_prompt = "Using the insight set as the help, when necessary please refer to the insights and guide you resolve question."

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

    def load_encyclopedia(self, encyclopedia_path: str):
        """Load the encyclopedia from a file. Mode determines how it's loaded:
        - normal mode: Load encyclopedia.txt and GraphRAG database from server.py
        - text mode: Load encyclopedia.json from server_text.py
        """
        try:
            if self.mode == "text":
                # Text mode: Load encyclopedia.json
                with open(encyclopedia_path, "r", encoding="utf-8") as f:
                    self.encyclopedia_dict = json.load(f)
                # Convert to string format for compatibility
                self.encyclopedia = json.dumps(self.encyclopedia_dict, indent=2)
                print(
                    f"Loaded encyclopedia.json from {encyclopedia_path} ({len(self.encyclopedia_dict)} insights)"
                )
            else:
                # Normal mode: Load encyclopedia.txt and GraphRAG database
                with open(encyclopedia_path, "r", encoding="utf-8") as f:
                    self.encyclopedia = f.read().strip()
                print(
                    f"Loaded encyclopedia from {encyclopedia_path} ({len(self.encyclopedia)} characters)"
                )

                # Try to load GraphRAG database from the same directory
                encyclopedia_dir = os.path.dirname(encyclopedia_path)
                graphrag_path = os.path.join(encyclopedia_dir, "graphrag_db.json")
                if os.path.exists(graphrag_path):
                    self._load_graphrag_database(graphrag_path, encyclopedia_dir)
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to load encyclopedia from {encyclopedia_path}: {e}"
            )

    def _load_embedding_model(self):
        """Load embedding model for GraphRAG retrieval"""
        if self.embedding_model is not None:
            return

        try:
            print(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print("Embedding model loaded successfully!")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for GraphRAG. Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model {self.embedding_model_name}: {e}"
            )

    def _load_graphrag_database(self, graphrag_path: str, data_dir: str):
        """Load GraphRAG database from disk"""
        try:
            print(f"Loading GraphRAG database from {graphrag_path}...")
            with open(graphrag_path, "r", encoding="utf-8") as db_file:
                self.graphrag_db = json.load(db_file)

            # Load embeddings
            embeddings_path = os.path.join(data_dir, "graphrag_embeddings.npy")
            if os.path.exists(embeddings_path):
                self.graphrag_embeddings = np.load(embeddings_path)
                print(f"Loaded {len(self.graphrag_embeddings)} skill embeddings")

            # Build graph if networkx is available
            if HAS_NETWORKX and self.graphrag_db:
                self.graphrag_graph = nx.Graph()
                # Add nodes
                for skill_name, node_data in self.graphrag_db["nodes"].items():
                    self.graphrag_graph.add_node(skill_name, **node_data)
                # Add edges
                for edge in self.graphrag_db["edges"]:
                    self.graphrag_graph.add_edge(
                        edge["source"],
                        edge["target"],
                        similarity=edge["similarity"],
                        type=edge["type"],
                    )
                print(
                    f"GraphRAG graph built: {self.graphrag_graph.number_of_nodes()} nodes, {self.graphrag_graph.number_of_edges()} edges"
                )

            print("GraphRAG database loaded successfully!")
        except Exception as e:
            print(f"Warning: Failed to load GraphRAG database: {e}")
            self.graphrag_db = None

    def _retrieve_insights_rag(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve relevant insights using GraphRAG (graph traversal + similarity search)"""
        if not self.graphrag_db:
            print("Warning: GraphRAG database not available. Cannot retrieve skills.")
            return []

        # Sanity check: Ensure graphrag_db has nodes
        if not self.graphrag_db.get("nodes"):
            print("Warning: GraphRAG database has no nodes. Cannot retrieve skills.")
            return []

        # Load embedding model
        self._load_embedding_model()

        # Get query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
        query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)

        # Normalize embeddings if needed
        if self.graphrag_embeddings is not None:
            embeddings_norm = self.graphrag_embeddings / np.linalg.norm(
                self.graphrag_embeddings, axis=1, keepdims=True
            )
        else:
            # Extract from graphrag_db
            embeddings_list = [
                node["embedding"] for node in self.graphrag_db["nodes"].values()
            ]
            if not embeddings_list:
                print("Warning: No embeddings found in GraphRAG database.")
                return []
            embeddings_norm = np.array(embeddings_list)
            embeddings_norm = embeddings_norm / np.linalg.norm(
                embeddings_norm, axis=1, keepdims=True
            )

        # Compute similarity scores
        similarities = cosine_similarity([query_embedding_norm], embeddings_norm)[0]
        # Normalize to [0, 1]
        similarities = (similarities + 1) / 2

        # Get top-k by similarity
        skill_names = list(self.graphrag_db["nodes"].keys())
        top_indices = np.argsort(similarities)[::-1][:top_k]

        retrieved_skills = []
        retrieved_names = set()

        # Add top-k similar skills with validation
        for idx in top_indices:
            skill_name = skill_names[idx]
            if skill_name not in retrieved_names:
                node_data = self.graphrag_db["nodes"][skill_name]
                skill_desc = node_data.get("description", "")

                # Validate skill before adding
                if not skill_desc or len(skill_desc.strip()) < 10:
                    print(
                        f"Warning: Skipping skill '{skill_name}' - empty or invalid description"
                    )
                    continue

                retrieved_skills.append(
                    {
                        "skill_name": skill_name,
                        "description": skill_desc,
                        "similarity": float(similarities[idx]),
                        "retrieval_method": "similarity_search",
                    }
                )
                retrieved_names.add(skill_name)

        # Graph traversal: expand from top skills using graph edges
        if self.graphrag_graph and HAS_NETWORKX:
            for skill_name in list(retrieved_names)[:3]:  # Expand from top 3
                if skill_name in self.graphrag_graph:
                    # Get neighbors (1-hop)
                    neighbors = list(self.graphrag_graph.neighbors(skill_name))
                    for neighbor in neighbors[:5]:  # Limit to 5 neighbors
                        if neighbor not in retrieved_names:
                            node_data = self.graphrag_db["nodes"].get(neighbor)
                            if not node_data:
                                continue
                            edge_data = self.graphrag_graph.get_edge_data(
                                skill_name, neighbor, {}
                            )
                            skill_desc = node_data.get("description", "")

                            # Validate skill before adding
                            if not skill_desc or len(skill_desc.strip()) < 10:
                                continue

                            retrieved_skills.append(
                                {
                                    "skill_name": neighbor,
                                    "description": skill_desc,
                                    "similarity": edge_data.get("similarity", 0.0),
                                    "retrieval_method": "graph_traversal",
                                    "connected_to": skill_name,
                                }
                            )
                            retrieved_names.add(neighbor)
                            if len(retrieved_skills) >= top_k * 2:  # Limit total
                                break
                    if len(retrieved_skills) >= top_k * 2:
                        break

        # Sort by similarity and return top_k
        retrieved_skills.sort(key=lambda x: x["similarity"], reverse=True)
        final_skills = retrieved_skills[:top_k]

        # Report which skills were retrieved
        if final_skills:
            print(
                f"Retrieved insights: {[s['skill_name'] for s in final_skills]}"
            )
        else:
            print("Warning: No valid skills retrieved after validation")

        return final_skills

    def _get_generation_prompt(self, query: str, is_math: bool = True) -> tuple:
        """
        Get the prompt for generating an answer using the encyclopedia.
        Returns (system_prompt, user_prompt) tuple.

        For text mode: Uses system prompt with skills.
        For normal mode: Uses GraphRAG retrieval (no system prompt for DeepSeek-R1).
        """
        system_prompt = None
        user_prompt = ""

        if self.mode == "text":
            # Text mode: Use system prompt and full encyclopedia.json
            system_prompt = self.system_prompt

            # Format skills from encyclopedia.json
            skills_list = []
            for skill_name, skill_desc in self.encyclopedia_dict.items():
                skills_list.append(f"**{skill_name}**:\n{skill_desc}")

            skills_text = "\n\n".join(skills_list)
            skills_section = f"""Insights Encyclopedia:

{skills_text}

---
"""

            if is_math:
                user_prompt = f"""{skills_section}Problem: {query}

Please reason step by step using the relevant skills above. Follow the step-by-step instructions in each skill. Put your final answer within \\boxed{{}}.

<think>
"""
            else:
                user_prompt = f"""{skills_section}Query: {query}

Based on the relevant skills above, provide a clear and comprehensive answer to the query. Follow the step-by-step instructions in each skill when applicable. Reference specific skills, categories, or techniques when relevant.

<think>
"""
        else:
            # Normal mode: Use GraphRAG retrieval (no system prompt for DeepSeek-R1)
            retrieved_skills = []
            skills_text = ""

            if self.graphrag_db:
                retrieved_skills = self._retrieve_insights_rag(query, top_k=10)
                if retrieved_skills:
                    # Format skills with clear structure
                    skills_list = []
                    for skill in retrieved_skills:
                        skill_name = skill["skill_name"]
                        skill_desc = skill["description"]
                        # Check if skill description is valid
                        if skill_desc and len(skill_desc.strip()) >= 10:
                            skills_list.append(f"**{skill_name}**:\n{skill_desc}")
                        else:
                            print(
                                f"Warning: Skipping skill '{skill_name}' - invalid description"
                            )

                    if skills_list:
                        skills_text = "\n\n".join(skills_list)
                        skills_section = f"""Relevant Insights to Guide Your Solution:

{skills_text}

---
"""
                    else:
                        print("Warning: No valid skills retrieved after validation")
                        skills_section = ""
                else:
                    print("Warning: GraphRAG retrieval returned no skills")
                    skills_section = ""
            else:
                # Fallback to full encyclopedia if GraphRAG not available
                if self.encyclopedia:
                    skills_section = f"""Insights Encyclopedia:
{self.encyclopedia}

"""
                else:
                    skills_section = ""
                    print("Warning: No encyclopedia or GraphRAG database available")

            # Report which skills will be used
            if retrieved_skills:
                skill_names_used = [s["skill_name"] for s in retrieved_skills]
                print(f"Using {len(skill_names_used)} skills: {skill_names_used}")

            # For DeepSeek-R1: all instructions in user prompt, no system prompt
            if is_math:
                user_prompt = f"""{skills_section}Problem: {query}

Please reason step by step using the relevant skills above. Follow the step-by-step instructions in each skill. Put your final answer within \\boxed{{}}.

<think>
"""
            else:
                user_prompt = f"""{skills_section}Query: {query}

Based on the relevant skills above, provide a clear and comprehensive answer to the query. Follow the step-by-step instructions in each skill when applicable. Reference specific skills, categories, or techniques when relevant.

<think>
"""

        return (system_prompt, user_prompt)

    def generate(
        self, query: str, max_new_tokens: Optional[int] = None, is_math: bool = True
    ) -> str:
        """
        Generate an answer to a query using the encyclopedia.

        Args:
            query: The question or query to answer
            max_new_tokens: Maximum number of new tokens to generate.
                           If None, uses the value from __init__ (default: None)
            is_math: Whether this is a math problem (affects prompt format for DeepSeek-R1)

        Returns:
            Generated answer text
        """
        if not self.encyclopedia and not self.encyclopedia_dict:
            raise ValueError("Encyclopedia not loaded. Call load_encyclopedia() first.")

        # Get the prompt (returns (system_prompt, user_prompt) tuple)
        system_prompt, user_prompt = self._get_generation_prompt(query, is_math=is_math)

        # Use Gemini API if configured
        if self.use_gemini:
            return self._call_gemini(user_prompt, system_prompt, max_new_tokens)

        # Otherwise use HuggingFace model
        # Load model if not already loaded
        self._load_model()

        try:
            # For text mode with system prompt, combine into user prompt for DeepSeek-R1
            # For normal mode, system_prompt is None
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
            else:
                full_prompt = user_prompt

            # Tokenize input
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=65536,  # Large limit for encyclopedia content
            ).to(self.device)

            # Calculate input token count for dynamic output sizing
            input_token_count = inputs["input_ids"].shape[1]

            # Use provided max_new_tokens or fall back to instance default
            max_tokens = (
                max_new_tokens if max_new_tokens is not None else self.max_new_tokens
            )

            print(f"Input tokens: {input_token_count}, Max new tokens: {max_tokens}")

            with torch.no_grad():
                # Use standard settings for reliable generation
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,  # Standard setting for reliable output
                    do_sample=True,
                    top_p=0.9,  # Standard setting for reliable output
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            return generated_text.strip()

        except Exception as e:
            print(f"Error generating response: {e}")
            return f"[Error] Generation failed: {str(e)}"

    def _call_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Call Gemini API"""
        try:
            # Use provided max_new_tokens or fall back to instance default
            max_tokens = (
                max_new_tokens if max_new_tokens is not None else self.max_new_tokens
            )

            # Combine system prompt and user prompt (Gemini API doesn't support system_instruction parameter)
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt

            # Configure generation parameters - use Gemini defaults, only set max_output_tokens
            generation_config = {}
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens

            # Generate response
            if generation_config:
                response = self.gemini_model.generate_content(
                    full_prompt, generation_config=generation_config
                )
            else:
                response = self.gemini_model.generate_content(full_prompt)

            # Handle response safely - check for blocked/filtered content
            if not response.candidates:
                raise RuntimeError(
                    "Gemini API returned no candidates. Response may have been blocked."
                )

            candidate = response.candidates[0]
            if candidate.finish_reason == 2:  # MAX_TOKENS
                # Hit token limit, but try to get partial text
                if candidate.content and candidate.content.parts:
                    text_parts = [
                        part.text
                        for part in candidate.content.parts
                        if hasattr(part, "text") and part.text
                    ]
                    if text_parts:
                        return "\n".join(text_parts).strip()
                raise RuntimeError(
                    "Gemini API hit token limit and no text was returned."
                )
            elif candidate.finish_reason == 3:  # SAFETY
                raise RuntimeError(
                    "Gemini API blocked the response due to safety filters."
                )
            elif candidate.finish_reason == 4:  # RECITATION
                raise RuntimeError("Gemini API blocked the response due to recitation.")

            # Try to get text from response
            try:
                return response.text.strip()
            except ValueError as e:
                # If response.text fails, try to extract from parts manually
                if candidate.content and candidate.content.parts:
                    text_parts = [
                        part.text
                        for part in candidate.content.parts
                        if hasattr(part, "text") and part.text
                    ]
                    if text_parts:
                        return "\n".join(text_parts).strip()
                raise RuntimeError(f"Failed to extract text from Gemini response: {e}")

        except Exception as e:
            raise RuntimeError(f"Error calling Gemini API: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Server - Answer queries using the Insights Encyclopedia"
    )
    parser.add_argument(
        "-e",
        "--encyclopedia",
        type=str,
        required=True,
        help="Path to encyclopedia file (encyclopedia.txt for normal mode, encyclopedia.json for text mode)",
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        required=True,
        help="Query/question to answer",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Hugging Face model name to use (default: deepseek-ai/DeepSeek-R1-Distill-Llama-8B)",
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
        "--output",
        type=str,
        default=None,
        help="Output file to save the answer (optional)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=98304,
        help="Maximum number of new tokens to generate (default: 98304)",
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
        "--mode",
        type=str,
        choices=["normal", "text"],
        default="normal",
        help="Mode: 'normal' for GraphRAG (server.py output) or 'text' for encyclopedia.json (server_text.py output) (default: normal)",
    )

    args = parser.parse_args()

    # Create server instance
    server = GenerateServer(
        model_name=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        use_gemini=args.use_gemini,
        gemini_api_key=args.gemini_api_key,
        mode=args.mode,
    )

    try:
        # Load encyclopedia
        server.load_encyclopedia(args.encyclopedia)

        # Generate answer
        print(f"\nQuery: {args.query}\n")
        print("Generating answer...")
        answer = server.generate(args.query)

        # Print answer
        print("\n" + "=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(answer)

        # Save to file if requested
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(f"Query: {args.query}\n\n")
                f.write("=" * 80 + "\n")
                f.write("ANSWER\n")
                f.write("=" * 80 + "\n\n")
                f.write(answer)
            print(f"\nAnswer saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        print("\nMake sure you have:")
        print("1. Generated the encyclopedia using server.py")
        print("2. Installed required packages: pip install -r requirements.txt")
        print("3. For GPU support, ensure CUDA is properly installed")
        print("\nExample usage:")
        print(
            "  python generate_server.py -e build/log/encyclopedia.txt -q 'How do I solve quadratic equations?'"
        )
        print(
            "  python generate_server.py -e build/log/encyclopedia.txt -q 'What skills are available for pattern matching?' -o answer.txt"
        )
