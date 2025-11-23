# Step 1: Scrape the papers
# python scraper.py -n 10 -o data/papers/iclr23_top5

# Step 2: Read the papers
python client.py -t "Summarize the paper of the contribution. Find out the main contributions and limitations of the paper. Discuss the potential future research directions and what novel research questions it raises. Also proposed solutions for this novel research questions." -m deepseek-ai/DeepSeek-R1-Distill-Llama-8B -p data/papers/iclr23_top5 -n 10 -d cuda