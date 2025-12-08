
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
python scraper.py -f diffusion -o data/papers/iclr23_diffusion --top5 --top25 --poster

python client.py -t "Resolve a critical question in diffusion model and related problems according to the paper" -m deepseek-ai/DeepSeek-R1-Distill-Llama-8B -p data/papers/iclr23_diffusion -d cuda

python server.py -i output -f math_output/problem*.json -m deepseek-ai/DeepSeek-R1-Distill-Llama-8B -d cuda -o encyclopedia

python generate_server.py -e encyclopedia/encyclopedia.txt -q "Based on existing skills learned, write 5 new research papers in the field of diffusion model which should be qualified for conference at the level of ICML, NeurIPS, ICLR, etc. Write out the proper academic paper title, abstract, and introduction for each research paper." -m deepseek-ai/DeepSeek-R1-Distill-Llama-8B -d cuda -o answer.txt

# Task 3 Math
python math_pipeline.py --start-from-step3  --encyclopedia encyclopedia/encyclopedia.txt --dataset2 math500 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --device cuda