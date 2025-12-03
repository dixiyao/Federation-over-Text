
# Task 1
# Step 1: Scrape the papers
# python scraper.py -n 10 -o data/papers/iclr23_top5

# Step 2: Read the papers
python client.py -t "Find out the solutions to the proposed research questions in the paper." -m deepseek-ai/DeepSeek-R1-Distill-Llama-8B -p data/papers/iclr23_top5 -n 10 -d cuda

# Step 3: Aggregate the skill books
python server.py -i output -f output/paper_*.json -m deepseek-ai/DeepSeek-R1-Distill-Llama-8B -d cuda -o encyclopedia 

# Step 4: Generate the answer with new encyclopedia
python generate_server.py -e encyclopedia/encyclopedia.txt -q "Based on existing skills learned, provide a novel research question around large diffusion language modeland potential solution" -m deepseek-ai/DeepSeek-R1-Distill-Llama-8B -d cuda -o answer.txt

# Task 2 Diffusion class
python client.py -t "Resolve a critical question in diffusion model and related problems according to the paper" -m deepseek-ai/DeepSeek-R1-Distill-Llama-8B -p data/papers/iclr23_diffusion -d cuda

python generate_server.py -e encyclopedia/encyclopedia.txt -q "Based on existing skills learned, find out a new research question in the field of diffusion model and develop a research proposal with potential solutions" -m deepseek-ai/DeepSeek-R1-Distill-Llama-8B -d cuda -o answer.txt