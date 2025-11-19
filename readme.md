# Federation over Text - Project Tasks

## Step 1: Paper Scraper ✅
Created `scraper.py` that can scrape the notable top 5% papers from ICLR 2023 from the OpenReview website:
- Source: https://openreview.net/group?id=ICLR.cc%2F2023%2FConference#notable-top-5-
- Target folder: `data/papers/iclr23_top5/`

### Usage:
```bash
python scraper.py
```

## Step 2: Chain-of-Thought Paper Reader ✅
Created `client.py` that:
- Uses a model (e.g., Deepseek-r1) to read one paper from the scraped folder
- Generates a chain-of-thought reasoning process with 5 distinct steps:
  1. Initial Analysis - Overview and structure
  2. Methodology Deep Dive - Technical approach analysis
  3. Experiments Analysis - Experimental validation
  4. Critical Evaluation - Comprehensive assessment
  5. Final Summary - Key takeaways
- Uses multiple-step generation (not single inference)
- Adopts similar approaches to current AI assistants (like Cursor) with multi-step thinking and reasoning
- Captures and outputs the complete reasoning process

### Usage:
```bash
# Set your API key (for Deepseek or other models)
export DEEPSEEK_API_KEY="your-api-key-here"

# Run the client
python client.py
```

### Setup:
```bash
# Install dependencies
pip install -r requirements.txt
```

### Features:
- **Multi-step reasoning**: Breaks down paper analysis into 5 distinct reasoning steps
- **Chain-of-thought**: Each step builds on previous analysis
- **Complete reasoning capture**: Saves both formatted text and JSON output
- **PDF extraction**: Automatically extracts text from PDF papers
- **Flexible API**: Can be adapted to work with different language model APIs