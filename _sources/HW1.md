# Homework 1: Lopez-Lira Replication

**Replicating Sentiment-Based Return Prediction**

In this assignment, you will replicate the core methodology from Lopez-Lira (2023), "Can ChatGPT Forecast Stock Price Movements?" This paper demonstrated that GPT-4's sentiment classifications of financial news headlines correlate with next-day stock returns—a finding with significant implications for quantitative trading.

---

## Learning Objectives

By completing this assignment, you will:

1. **Understand the Lopez-Lira methodology** and its significance
2. **Build a complete sentiment classification pipeline** using the OpenAI API
3. **Extract structured data** using Pydantic schemas
4. **Process financial news data** at scale
5. **Evaluate classifier performance** against baseline methods

---

## Background Reading

Before starting, read the following paper:

> **Lopez-Lira, A. & Tang, Y. (2023). "Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models."** [arXiv:2304.07619](https://arxiv.org/abs/2304.07619)

Key findings to understand:
- GPT-4 sentiment scores predict next-day returns
- Smaller models (BERT, GPT-3.5) showed no predictive power
- The effect is strongest for smaller, less-covered stocks

---

## Assignment Components

### Part 1: Paper Analysis (10 points)

Write a brief (1-2 page) summary addressing:

1. **Methodology:** How did Lopez-Lira collect headlines and classify sentiment?
2. **Key Finding:** What is the main result, and why does model scale matter?
3. **Limitations:** What are the paper's main limitations?
4. **Replication Scope:** What aspects will you replicate, and what will you simplify?

### Part 2: Sentiment Classification Pipeline (40 points)

Build a Python pipeline that:

1. **Loads headlines** from a CSV file
2. **Classifies each headline** as "good news," "bad news," or "uncertain"
3. **Returns structured output** with confidence scores
4. **Handles errors gracefully** (rate limits, API failures)

**Required structure:**

```python
from pydantic import BaseModel, Field
from typing import Literal

class LopezLiraSentiment(BaseModel):
    """Sentiment classification following Lopez-Lira (2023)."""

    classification: Literal["good news", "bad news", "uncertain"] = Field(
        description="Sentiment classification"
    )
    confidence: float = Field(
        description="Confidence score from 0.0 to 1.0",
        ge=0.0, le=1.0
    )
```

**Starter code:** See [11_lopez_lira](https://github.com/finm-33200/ai_inclass_examples/tree/main/basic_llm_api/11_lopez_lira) in the course examples repository.

### Part 3: Data Collection (20 points)

Collect or curate a dataset of at least **50 financial news headlines**. Options:

1. **Manual collection:** Gather headlines from financial news sites (Reuters, Bloomberg, WSJ)
2. **Public datasets:** Use existing headline datasets (e.g., Kaggle financial news)
3. **API access:** If you have access, use a news API

For each headline, record:
- Headline text
- Source
- Date
- Associated ticker (if mentioned)

**Important:** Document your data collection process. Include a README explaining your sources and any preprocessing.

### Part 4: Classification Results (20 points)

Run your classifier on the collected headlines and report:

1. **Distribution of classifications**
   - How many headlines were classified as good/bad/uncertain?
   - What is the average confidence score by category?

2. **Example outputs**
   - Show 5 example headlines with their classifications and reasoning
   - Identify any surprising or incorrect classifications

3. **Cost analysis**
   - Total tokens used
   - Estimated API cost

### Part 5: Baseline Comparison (10 points, Optional but Recommended)

Compare GPT-4o-mini's classifications against a baseline:

**Option A: VADER Sentiment**
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
scores = analyzer.polarity_scores(headline)
```

**Option B: FinBERT**
```python
from transformers import pipeline
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
result = finbert(headline)
```

Report:
- Agreement rate between GPT and baseline
- Cases where they disagree
- Which seems more accurate on your data?

---

## Deliverables

Submit the following:

1. **`report.pdf`** - Written analysis (Parts 1, 4, and 5)
2. **`classify_headlines.py`** - Your classification pipeline
3. **`data/headlines.csv`** - Your collected headlines
4. **`results/classifications.csv`** - Classification outputs
5. **`README.md`** - Setup instructions and data documentation

---

## Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| Paper Analysis | 10 | Clear summary, identifies key insights and limitations |
| Pipeline Code | 40 | Working code, proper error handling, structured outputs |
| Data Collection | 20 | Sufficient quantity, proper documentation, diverse sources |
| Results Analysis | 20 | Accurate reporting, insightful examples, cost awareness |
| Baseline Comparison | 10 | Meaningful comparison, correct interpretation |

**Total: 100 points** (90 without optional baseline comparison)

---

## Tips for Success

### Prompt Engineering

The system prompt significantly affects classification quality. Experiment with:

```python
# Basic prompt
"Classify this headline as good news, bad news, or uncertain."

# More detailed prompt (Lopez-Lira style)
"""You are a financial expert. For each headline, determine whether
it represents 'good news', 'bad news', or 'uncertain' for the
mentioned company's stock price. Consider market expectations
and typical investor reactions."""
```

### Rate Limit Handling

The OpenAI API has rate limits. Add retries with exponential backoff:

```python
import time
from openai import RateLimitError

def classify_with_retry(headline, max_retries=3):
    for attempt in range(max_retries):
        try:
            return classify_headline(headline)
        except RateLimitError:
            wait_time = 2 ** attempt
            time.sleep(wait_time)
    raise Exception("Max retries exceeded")
```

### Cost Management

GPT-4o-mini is cost-effective, but monitor usage:
- ~25 tokens per headline (input)
- ~50 tokens per classification (output)
- 50 headlines ≈ 3,750 tokens ≈ $0.001

### Model Comparison

If you want to replicate Lopez-Lira's finding that scale matters, try classifying the same headlines with different models:
- `gpt-4o-mini` (our default)
- `gpt-4o` (more expensive, potentially better)
- `meta-llama/llama-3.1-8b-instruct` via OpenRouter (cheaper, smaller)

---

## Resources

- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [In-Class Examples: 11_lopez_lira](https://github.com/finm-33200/ai_inclass_examples/tree/main/basic_llm_api/11_lopez_lira)
- [Lopez-Lira Paper](https://arxiv.org/abs/2304.07619)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)
- [FinBERT on HuggingFace](https://huggingface.co/ProsusAI/finbert)

---

## Submission

Submit via the course's designated submission method (Canvas/GitHub Classroom) by the deadline posted in the syllabus.

**Questions?** Ask on the course discussion forum or during office hours.
