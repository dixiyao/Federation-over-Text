# Step 1: Scrape the papers
# python scraper.py -n 10 -o data/papers/iclr23_top5

# Step 2: Read the papers
python client.py -t "Find out the solutions to the proposed research questions in the paper." -m deepseek-ai/DeepSeek-R1-Distill-Llama-8B -p data/papers/iclr23_top5 -n 10 -d cuda

# Step 3: Aggregate the skill books
python server.py -i output -f output/paper_*.json -m deepseek-ai/DeepSeek-R1-Distill-Llama-8B -d cuda -o encyclopedia 