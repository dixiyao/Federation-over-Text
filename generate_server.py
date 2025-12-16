"""
Generate Server - Inference using Encyclopedia
Simple inference server that uses the aggregated encyclopedia to answer queries.
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


class GenerateServer:
    """
    Simple inference server that uses the encyclopedia to answer queries.
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        device: Optional[str] = None,
        max_new_tokens: int = 98304,
    ):
        self.model_name = model_name
        self.encyclopedia = ""
        self.max_new_tokens = max_new_tokens

        # Model and tokenizer will be loaded lazily on first use
        self.model = None
        self.tokenizer = None
        self.device = device or ("cuda" if self._check_cuda() else "cpu")
        
        # GraphRAG database
        self.graphrag_db = None
        self.graphrag_embeddings = None
        self.graphrag_graph = None
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
        """Load the encyclopedia from a file and GraphRAG database if available"""
        try:
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
            raise RuntimeError(f"Failed to load embedding model {self.embedding_model_name}: {e}")
    
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
                        edge["source"], edge["target"],
                        similarity=edge["similarity"], type=edge["type"]
                    )
                print(f"GraphRAG graph built: {self.graphrag_graph.number_of_nodes()} nodes, {self.graphrag_graph.number_of_edges()} edges")
            
            print("GraphRAG database loaded successfully!")
        except Exception as e:
            print(f"Warning: Failed to load GraphRAG database: {e}")
            self.graphrag_db = None
    
    def _retrieve_skills_rag(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve relevant skills using GraphRAG (graph traversal + similarity search)"""
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
            embeddings_norm = self.graphrag_embeddings / np.linalg.norm(self.graphrag_embeddings, axis=1, keepdims=True)
        else:
            # Extract from graphrag_db
            embeddings_list = [node["embedding"] for node in self.graphrag_db["nodes"].values()]
            if not embeddings_list:
                print("Warning: No embeddings found in GraphRAG database.")
                return []
            embeddings_norm = np.array(embeddings_list)
            embeddings_norm = embeddings_norm / np.linalg.norm(embeddings_norm, axis=1, keepdims=True)
        
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
                    print(f"Warning: Skipping skill '{skill_name}' - empty or invalid description")
                    continue
                
                retrieved_skills.append({
                    "skill_name": skill_name,
                    "description": skill_desc,
                    "similarity": float(similarities[idx]),
                    "retrieval_method": "similarity_search"
                })
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
                            edge_data = self.graphrag_graph.get_edge_data(skill_name, neighbor, {})
                            skill_desc = node_data.get("description", "")
                            
                            # Validate skill before adding
                            if not skill_desc or len(skill_desc.strip()) < 10:
                                continue
                            
                            retrieved_skills.append({
                                "skill_name": neighbor,
                                "description": skill_desc,
                                "similarity": edge_data.get("similarity", 0.0),
                                "retrieval_method": "graph_traversal",
                                "connected_to": skill_name
                            })
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
            print(f"Retrieved {len(final_skills)} skills: {[s['skill_name'] for s in final_skills]}")
        else:
            print("Warning: No valid skills retrieved after validation")
        
        return final_skills

    def _get_generation_prompt(self, query: str, is_math: bool = True) -> str:
        """
        Get the prompt for generating an answer using the encyclopedia with GraphRAG retrieval.
        
        For DeepSeek-R1 models: All instructions must be in the user prompt (no system prompt).
        For math problems: Include directive to reason step by step and put answer in \\boxed{}.
        """
        # Use GraphRAG to retrieve relevant skills
        retrieved_skills = []
        skills_text = ""
        
        if self.graphrag_db:
            retrieved_skills = self._retrieve_skills_rag(query, top_k=10)
            if retrieved_skills:
                # Format skills with clear structure
                skills_list = []
                for skill in retrieved_skills:
                    skill_name = skill['skill_name']
                    skill_desc = skill['description']
                    # Check if skill description is valid
                    if skill_desc and len(skill_desc.strip()) >= 10:
                        skills_list.append(f"**{skill_name}**:\n{skill_desc}")
                    else:
                        print(f"Warning: Skipping skill '{skill_name}' - invalid description")
                
                if skills_list:
                    skills_text = "\n\n".join(skills_list)
                    skills_section = f"""Relevant Skills to Guide Your Solution:

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
                skills_section = f"""Skills Encyclopedia:
{self.encyclopedia}

"""
            else:
                skills_section = ""
                print("Warning: No encyclopedia or GraphRAG database available")
        
        # Report which skills will be used
        if retrieved_skills:
            skill_names_used = [s['skill_name'] for s in retrieved_skills]
            print(f"Using {len(skill_names_used)} skills: {skill_names_used}")
        
        # For DeepSeek-R1: all instructions in user prompt, no system prompt
        if is_math:
            prompt = f"""{skills_section}Problem: {query}

Please reason step by step using the relevant skills above. Follow the step-by-step instructions in each skill. Put your final answer within \\boxed{{}}.

<think>
"""
        else:
            prompt = f"""{skills_section}Query: {query}

Based on the relevant skills above, provide a clear and comprehensive answer to the query. Follow the step-by-step instructions in each skill when applicable. Reference specific skills, categories, or techniques when relevant.

<think>
"""
        return prompt

    def generate(self, query: str, max_new_tokens: Optional[int] = None, is_math: bool = True) -> str:
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
        if not self.encyclopedia:
            raise ValueError("Encyclopedia not loaded. Call load_encyclopedia() first.")

        # Load model if not already loaded
        self._load_model()

        # Get the prompt (DeepSeek-R1: all instructions in user prompt, no system prompt)
        prompt = self._get_generation_prompt(query, is_math=is_math)

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=65536,  # Large limit for encyclopedia content
            ).to(self.device)

            # Calculate input token count for dynamic output sizing
            input_token_count = inputs["input_ids"].shape[1]

            # Use provided max_new_tokens or fall back to instance default
            max_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens

            print(
                f"Input tokens: {input_token_count}, Max new tokens: {max_tokens}"
            )

            with torch.no_grad():
                # DeepSeek-R1 recommendations: temperature 0.5-0.7 (0.6 recommended)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0,  # Recommended for DeepSeek-R1
                    do_sample=True,
                    top_p=0.95,  # Recommended for DeepSeek-R1
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Server - Answer queries using the Skills Encyclopedia"
    )
    parser.add_argument(
        "-e",
        "--encyclopedia",
        type=str,
        required=True,
        help="Path to encyclopedia.txt file",
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

    args = parser.parse_args()

    # Create server instance
    server = GenerateServer(
        model_name=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
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
