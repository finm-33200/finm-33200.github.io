# ChartBook: Cataloging Research for Humans and AI

**Learning Objectives:**
- Understand why coding agents create a research management problem
- Install ChartBook and explore its CLI and Python API
- Distinguish between Pipeline (single project) and Catalog (multi-pipeline) project types
- Use ChartBook to programmatically discover and load datasets across repositories

---

## The Research Repository Problem

With coding agents like Claude Code, you will create far more repositories than you ever did before. Each project produces cleaned datasets, derived data, charts, and documentation. Within a few months, you'll have dozens of repositories — and finding what you built last quarter becomes a real problem.

This isn't just a human problem. When you ask an AI agent to start a new analysis, it needs to know what data you've already cleaned, what transformations you've already built, and what datasets are available across your projects. Without a catalog, the agent starts from scratch every time.

**ChartBook** solves this by making every project's outputs — datasets, charts, and documentation — discoverable through a CLI and Python API, by both humans and AI agents.

### Why This Matters for AI Workflows

The key insight is that ChartBook is designed not only to help *you* catalog your prior research, but to make it easy for *AI agents* to discover and use your datasets. An AI agent can:

- Run `chartbook ls` to see every pipeline and dataset you've ever created
- Call `chartbook data get-docs` to pull the full data dictionary for any dataset
- Use the Python API to load cleaned data directly from another repository
- Browse your chart catalog to understand what analyses already exist

This is the infrastructure you need when coding agents multiply your repository count by 10x. You need tools that let you — and your AI — programmatically manage all of your research.

---

## What is ChartBook?

[ChartBook](https://github.com/backofficedev/chartbook) is a developer platform for cataloging data science and research projects. It organizes your work into a structured, searchable catalog with automatic documentation generation.

**Links:** [PyPI](https://pypi.org/project/chartbook/) | [Documentation](https://backofficedev.github.io/chartbook/) | [GitHub](https://github.com/backofficedev/chartbook)

### Core Concepts

| Concept | Description | Example |
|---------|-------------|---------|
| **Pipeline** | A single research project with data, charts, and docs | `news_headlines` |
| **Catalog** | An aggregation of multiple pipelines | Your master research catalog |
| **Dataframe** | A dataset (parquet) with metadata and documentation | `scraped_headlines_with_rp_metadata` |
| **Chart** | A visualization exported from a pipeline | Interactive Plotly HTML charts |

### Key Features

**For Humans:**
- Export cleaned datasets from each project as documented parquet files
- Browse charts and data summaries across all your projects in a generated HTML site
- Automate data refresh with `dodo.py` task runners
- Track data governance: licenses, access permissions, provider contacts

**For AI Agents:**
- CLI commands that agents can call to discover datasets and load documentation
- Python API for programmatic data access across repositories
- Auto-generated `llms.txt` files for LLM-friendly project summaries
- Data dictionaries loadable in a single function call — no manual copy-paste

---

## Installation

```bash
# Recommended: install with pipx for CLI use
pipx install chartbook

# Or install with pip (includes data loading)
pip install "chartbook[all]"
```

---

## CLI Reference

ChartBook's CLI is designed to be used by both humans and AI agents. Here are the key commands:

### Listing What's Available

```bash
# List all pipelines in your catalog
chartbook ls

# List dataframes in a specific pipeline
chartbook ls --pipeline NEWS_HEADLINES

# List charts
chartbook ls --charts
```

### Working with Data

```bash
# Get the file path to a dataframe's parquet file
chartbook data get-path --pipeline NEWS_HEADLINES --dataframe scraped_headlines

# Print the full data documentation (data dictionary, metadata, governance)
chartbook data get-docs --pipeline NEWS_HEADLINES --dataframe scraped_headlines
```

### Building Documentation

```bash
# Generate an HTML documentation site for your project
chartbook build

# Open the built docs in your browser
chartbook browse
```

### Catalog Management

```bash
# Initialize a global catalog
chartbook catalog init

# Add a pipeline to your catalog
chartbook catalog add /path/to/my/project

# Build the catalog documentation (aggregates all pipelines)
chartbook catalog build

# Browse the catalog
chartbook catalog browse
```

---

## Python API

The Python API lets you load data and documentation from any cataloged pipeline — this is what makes ChartBook powerful for cross-project work and AI agent workflows.

### Loading Data

```python
from chartbook import data

# Load a dataframe as pandas
df = data.load(pipeline="NEWS_HEADLINES", dataframe="scraped_headlines", format="pandas")

# Load as Polars LazyFrame (default, more efficient for large datasets)
lf = data.load(pipeline="NEWS_HEADLINES", dataframe="scraped_headlines")

# Get the resolved file path
path = data.get_data_path(pipeline="NEWS_HEADLINES", dataframe="scraped_headlines")
```

### Loading Documentation

```python
# Get the full data dictionary as a string
docs = data.get_docs(pipeline="NEWS_HEADLINES", dataframe="scraped_headlines")
print(docs)

# Get the path to the documentation file
docs_path = data.get_docs_path(pipeline="NEWS_HEADLINES", dataframe="scraped_headlines")
```

This is especially powerful for AI agents: instead of manually describing your data to the LLM, the agent can call `data.get_docs()` to pull the complete data dictionary automatically.

---

## In-Class Exercise: News Headlines Dataset

We'll demonstrate ChartBook by working with a real research pipeline that you'll use throughout this course.

### Clone the Repository

```bash
git clone https://github.com/finm-33200/news_headlines
cd news_headlines
```

### Background

The [news_headlines](https://github.com/finm-33200/news_headlines) repository curates firm-level news headlines for S&P 500 companies from multiple sources. This dataset is critical for replicating two key papers in our upcoming homework assignments:

- **HW1** — [Lopez-Lira & Tang (2023)](../references/lopez-lira_tang_2023.pdf): Sentiment classification of headlines using LLMs
- **HW2** — [Chen, Kelly & Xiu (2022)](../references/chen_kelly_xiu_2022_expected_returns_and_large_language_models.pdf): Embedding-based return prediction

### The RavenPack Problem

[RavenPack](https://www.ravenpack.com/) provides the best entity-tagged financial news headlines available. Their data includes company identifiers, sentiment scores, relevance ratings, and topic classifications — exactly what you need to connect headlines to stock returns.

**The catch:** RavenPack's terms of use prohibit uploading their headline text to third-party LLMs like OpenAI's API. You can't just take RavenPack headlines and send them to GPT-4 for sentiment classification.

### The Solution: Scrape and Fuzzy-Match

The `news_headlines` pipeline implements a creative workaround:

1. **Scrape headlines independently** from free sources — GDELT (via Google BigQuery) and free newswire services (Dow Jones Newswires, PR Newswire, Business Wire, GlobeNewswire)
2. **Send the scraped headlines to OpenAI** — these are independently sourced, so there are no licensing restrictions
3. **Fuzzy-match the scraped headlines to RavenPack** — transfer RavenPack's entity metadata (company IDs, relevance scores, topic tags) to the independently-sourced headlines
4. **Use the metadata to connect headlines to stocks** — now you have headlines you can send to LLMs, tagged with the companies they're about

The fuzzy match quality is high enough that the matched headlines are essentially identical — the same news event reported by the same wire service, just sourced through different channels. This gives you the best of both worlds: RavenPack's metadata infrastructure with freely usable headline text.

### Explore with ChartBook

Once you've cloned the repository, use ChartBook to explore its contents:

```bash
# See what datasets are available
chartbook ls

# Read the documentation for the main output dataset
chartbook data get-docs --pipeline NEWS_HEADLINES --dataframe scraped_headlines_with_rp_metadata
```

```python
from chartbook import data

# Load the dataset
df = data.load(
    pipeline="NEWS_HEADLINES",
    dataframe="scraped_headlines_with_rp_metadata",
    format="pandas"
)

# See what columns are available
print(df.columns.tolist())
print(df.head())
```

---

## Key Takeaways

1. **Coding agents multiply your repositories** — you need infrastructure to manage the outputs across all of them
2. **ChartBook makes research discoverable** — both humans and AI agents can find and load datasets programmatically
3. **The CLI is designed for AI** — agents can call `chartbook ls` and `chartbook data get-docs` to discover what data exists and understand its schema
4. **Data licensing requires creative solutions** — the RavenPack workaround shows how to respect terms of use while still leveraging LLMs
5. **The news_headlines dataset** will be used throughout this course for homework assignments on sentiment analysis and embedding-based return prediction

*Later in the course, we'll wrap ChartBook's API as [LangChain tools](chartbook_integration.md) to build agents that can autonomously discover and analyze financial data.*
