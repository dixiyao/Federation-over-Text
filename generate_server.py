"""
Generate Server - Inference using Encyclopedia
Simple inference server that uses the aggregated encyclopedia to answer queries.
"""

import argparse
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class GenerateServer:
    """
    Simple inference server that uses the encyclopedia to answer queries.
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.encyclopedia = ""
        
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

    def load_encyclopedia(self, encyclopedia_path: str):
        """Load the encyclopedia from a file"""
        try:
            with open(encyclopedia_path, "r", encoding="utf-8") as f:
                self.encyclopedia = f.read().strip()
            print(f"Loaded encyclopedia from {encyclopedia_path} ({len(self.encyclopedia)} characters)")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load encyclopedia from {encyclopedia_path}: {e}")

    def _get_generation_prompt(self, query: str) -> str:
        """Get the prompt for generating an answer using the encyclopedia"""
        prompt = f"""
You are an expert assistant with access to a comprehensive Skills Encyclopedia. Use the encyclopedia to provide accurate, helpful answers to queries.

### Encyclopedia:
{self.encyclopedia}

### Query:
{query}

### Task:
Based on the Skills Encyclopedia above, provide a clear and comprehensive answer to the query. Reference specific skills, categories, or techniques from the encyclopedia when relevant. If the query requires skills or knowledge not in the encyclopedia, acknowledge this and provide the best answer you can based on the available information.

Provide your answer:
"""
        return prompt

    def generate(self, query: str) -> str:
        """
        Generate an answer to a query using the encyclopedia.
        
        Args:
            query: The question or query to answer
        
        Returns:
            Generated answer text
        """
        if not self.encyclopedia:
            raise ValueError("Encyclopedia not loaded. Call load_encyclopedia() first.")
        
        # Load model if not already loaded
        self._load_model()
        
        # Get the prompt
        prompt = self._get_generation_prompt(query)
        
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
            
            # Generate response - dynamically size based on input
            max_new_tokens = max(int(input_token_count * 1.5), 4096)
            max_new_tokens = min(max_new_tokens, 32768)  # Cap at 32k tokens
            
            print(f"Input tokens: {input_token_count}, Max new tokens: {max_new_tokens}")
            
            with torch.no_grad():
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
                outputs[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
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

    args = parser.parse_args()

    # Create server instance
    server = GenerateServer(
        model_name=args.model,
        device=args.device,
    )

    try:
        # Load encyclopedia
        server.load_encyclopedia(args.encyclopedia)
        
        # Generate answer
        print(f"\nQuery: {args.query}\n")
        print("Generating answer...")
        answer = server.generate(args.query)
        
        # Print answer
        print("\n" + "="*80)
        print("ANSWER")
        print("="*80)
        print(answer)
        
        # Save to file if requested
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(f"Query: {args.query}\n\n")
                f.write("="*80 + "\n")
                f.write("ANSWER\n")
                f.write("="*80 + "\n\n")
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
        print("  python generate_server.py -e build/log/encyclopedia.txt -q 'How do I solve quadratic equations?'")
        print("  python generate_server.py -e build/log/encyclopedia.txt -q 'What skills are available for pattern matching?' -o answer.txt")

