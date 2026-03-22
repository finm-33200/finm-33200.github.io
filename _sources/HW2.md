# Homework 2: Embedding-Based Return Prediction

**Replicating Chen, Kelly & Xiu (2022) with RavenPack Headlines**

In this assignment, you will replicate a portion of the analysis from Chen, Kelly & Xiu (2022), "Expected Returns and Large Language Models." This paper demonstrated that LLM embeddings—dense vector representations of news text—significantly outperform simpler NLP methods for predicting stock returns. You will use RavenPack news headlines (via WRDS) instead of full article text, and compare two embedding models: BERT and OpenAI's `text-embedding-3-large`.

---

## Learning Objectives

By completing this assignment, you will:

1. **Understand the evolution of text representations** from bag-of-words to contextual embeddings
2. **Build embedding pipelines** using both open-source (BERT) and commercial (OpenAI) models
3. **Work with WRDS financial databases** (RavenPack for news, CRSP for returns)
4. **Link news to stock returns** using entity mapping
5. **Replicate empirical asset pricing tests** (portfolio sorts, Fama-MacBeth regressions)

---

## Background Reading

Before starting, read the following paper:

> **Chen, Y., Kelly, B., & Xiu, D. (2022). "Expected Returns and Large Language Models."** [SSRN](https://ssrn.com/abstract=4416687)

This paper investigates the predictive power of Large Language Models (LLMs) on stock returns. Unlike traditional "bag-of-words" approaches that ignore word order, LLMs capture context and nuance (such as negation) to extract richer information from news text. The authors find that embeddings from models like BERT and ChatGPT significantly improve return prediction accuracy compared to simpler NLP methods.

From the abstract:

> "We leverage state-of-the-art large language models (LLMs) such as ChatGPT and LLaMA to extract contextualized representations of news text for predicting stock returns. Our results show that prices respond slowly to news reports indicative of market inefficiencies and limits-to-arbitrage. Predictions from LLM embeddings significantly improve over leading technical signals (such as past returns) or simpler NLP methods by understanding news text in light of the broader article context."

Also relevant for context on simpler text representations:

> **Bybee, L., Kelly, B., Manela, A., & Xiu, D. (2024). "Business News and Business Cycles." *Journal of Finance*, 79(5), 3105-3147.** [Published version](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13377)

Bybee et al. use a bag-of-words topic model on 800,000 Wall Street Journal articles to measure the state of the economy. Their news attention time series closely track macroeconomic indicators and forecast stock returns. This paper represents the "old way" of extracting information from text—Chen et al. show how LLM embeddings improve on it.

---

## Key Simplifications

You will not replicate the full paper. Key differences from the original:

| Original Paper | Your Replication |
|----------------|-----------------|
| Full article text (Thomson Reuters) | Headlines only (RavenPack via WRDS) |
| 16 global equity markets | US equities only |
| Many models (BERT, GPT, LLaMA, Word2Vec, BoW) | Two models: BERT and `text-embedding-3-large` |
| Tables 1–8 | Tables 1, 2, and 3 only |

---

## Assignment Components

### Part 1: Paper Analysis (10 points)

Write a brief (1-2 page) summary addressing:

1. **Text representations:** How do bag-of-words, Word2Vec, BERT, and GPT embeddings differ? Why do contextual embeddings capture more information?
2. **Methodology:** How do Chen et al. construct their prediction pipeline? What is the role of embeddings in the prediction?
3. **Key finding:** What is the main result comparing LLM embeddings to simpler methods?
4. **Replication scope:** What aspects will you replicate, and what will you simplify?

### Part 2: Data Collection from WRDS (25 points)

Build a data pipeline that:

1. **Connects to WRDS** via the `wrds` Python package
2. **Queries RavenPack headlines** with entity mapping to stock tickers
3. **Pulls CRSP daily stock returns** for matched firms
4. **Documents the data collection process** with a clear README

**Starter code for WRDS connection:**

```python
import wrds

db = wrds.Connection()

# Query RavenPack headlines with entity mapping
headlines = db.raw_sql("""
    SELECT rp_story_id, headline, entity_name,
           rp_entity_id, relevance, event_sentiment_score,
           timestamp_utc
    FROM ravenpack.rpa_dj_equities
    WHERE relevance >= 75
    AND timestamp_utc BETWEEN '2018-01-01' AND '2022-12-31'
    LIMIT 1000
""")
```

**Requirements:**
- Filter by relevance score >= 75 (standard RavenPack convention)
- Use RavenPack entity mapping to link headlines to CRSP PERMNOs
- Pull daily returns from CRSP for matched firms
- Choose a reasonable time window (e.g., 2018–2022)
- Start with a small sample (1,000 headlines) to debug, then scale up

### Part 3: Embedding Pipeline (30 points)

Build two embedding pipelines and embed all collected headlines:

**1. BERT embeddings** using HuggingFace Transformers:

```python
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def get_bert_embedding(text: str) -> np.ndarray:
    """Extract BERT [CLS] token embedding for a headline."""
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use [CLS] token embedding
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()
```

**2. OpenAI embeddings** using `text-embedding-3-large`:

```python
from openai import OpenAI

client = OpenAI()

def get_openai_embedding(text: str) -> list[float]:
    """Get OpenAI embedding for a headline."""
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding
```

**Requirements:**
- Embed all headlines with both models
- Store embeddings efficiently (parquet or numpy `.npy` files)
- Report embedding dimensions for each model
- Report cost analysis for OpenAI embeddings
- Handle batching for OpenAI API calls (the API accepts lists of strings)

### Part 4: Return Prediction — Tables 1, 2, 3 (25 points)

Replicate the following tables from Chen et al., adapted for your headline dataset:

**Table 1: Summary Statistics**
- Number of headlines per month
- Average number of headlines per firm
- Cross-sectional distribution of stock returns (mean, std, percentiles)
- Embedding dimensionality for each model
- Sample period coverage

**Table 2: Portfolio Sorts**

Sort stocks into quintile portfolios based on an embedding-derived signal:

1. Use a rolling out-of-sample approach: at each date, estimate a linear projection of embeddings onto future returns using only past data
2. Form quintile portfolios based on the predicted return signal
3. Report for each quintile: average excess return, CAPM alpha, and Sharpe ratio
4. Report the long-short spread (Q5 - Q1)

Compare results for BERT vs. OpenAI embeddings.

**Table 3: Fama-MacBeth Regressions**

Run cross-sectional Fama-MacBeth regressions:

1. Regress monthly stock returns on the embedding-derived signal
2. Add controls: size (log market cap), book-to-market, and momentum (past 12-month return)
3. Report time-series averages of cross-sectional coefficients with Newey-West t-statistics

```python
# Example using linearmodels
from linearmodels.panel import FamaMacBeth

# Set up panel data with entity and time indices
panel = data.set_index(['permno', 'date'])

# Run Fama-MacBeth regression
fm = FamaMacBeth(panel['ret'], panel[['embedding_signal', 'log_mcap', 'bm', 'mom']])
result = fm.fit(cov_type='kernel')
print(result.summary)
```

### Part 5: Discussion (10 points)

Write a discussion (1-2 pages) addressing:

1. **BERT vs. OpenAI:** How do the two embedding models compare in predictive power? Which produces stronger portfolio sorts?
2. **Headlines vs. articles:** How might using headlines (rather than full articles) affect the results relative to the original paper?
3. **Connection to Bybee et al.:** How do your embedding-based results relate to the bag-of-words approach used in Bybee et al. (2024)? What additional information might embeddings capture?

---

## Deliverables

Submit the following:

1. **`report.pdf`** - Written analysis (Parts 1, 4, and 5)
2. **`collect_data.py`** - WRDS data collection pipeline
3. **`embed_headlines.py`** - Embedding pipeline for both models
4. **`predict_returns.py`** - Portfolio sorts and Fama-MacBeth regressions
5. **`data/`** - Collected headlines and merged dataset
6. **`results/`** - Tables and figures
7. **`README.md`** - Setup instructions and data documentation

---

## Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| Paper Analysis | 10 | Clear summary, understands representation differences |
| Data Collection | 25 | Working WRDS pipeline, proper entity mapping, documented |
| Embedding Pipeline | 30 | Both models working, proper storage, cost analysis |
| Return Prediction | 25 | Correct portfolio sorts, Fama-MacBeth regressions |
| Discussion | 10 | Meaningful comparison, connects to course themes |

**Total: 100 points**

---

## Tips for Success

### Cost Management

OpenAI's `text-embedding-3-large` is very affordable:
- ~$0.13 per 1M tokens
- A typical headline is ~15 tokens
- 10,000 headlines = ~150K tokens = **~$0.02**
- Even 100,000 headlines costs less than $0.20

BERT runs locally and is free (but slower without a GPU).

### WRDS Access

All students should have WRDS access through the university. If you haven't set up your account:
1. Go to [wrds.wharton.upenn.edu](https://wrds.wharton.upenn.edu)
2. Register with your UChicago email
3. Install the Python package: `pip install wrds`

### Fama-MacBeth Regressions

Use the `linearmodels` package:
```bash
pip install linearmodels
```

The `FamaMacBeth` class handles the two-step procedure automatically.

### Dimensionality Reduction

BERT produces 768-dimensional embeddings; `text-embedding-3-large` produces 3,072-dimensional embeddings. For the linear projection onto returns, you may need to reduce dimensionality first (e.g., PCA) to avoid overfitting, especially with limited training data.

### Start Small

Debug with 1,000 headlines before scaling up. The full pipeline (data collection → embedding → prediction) has many steps where things can go wrong. Get each step working on a small sample first.

---

## Resources

- [Chen, Kelly & Xiu (2022) on SSRN](https://ssrn.com/abstract=4416687)
- [Bybee et al. (2024) - Business News and Business Cycles](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13377)
- [Structure of News Data](https://structureofnews.com/) — Pre-constructed topic attention time series from Bybee et al.
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [WRDS Python Package](https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/)
- [linearmodels Documentation](https://bashtage.github.io/linearmodels/)

---

## Submission

Submit via the course's designated submission method (Canvas/GitHub Classroom) by the deadline posted in the syllabus.

**Questions?** Ask on the course discussion forum or during office hours.
