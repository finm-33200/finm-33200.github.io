# ChartBook Integration: Connecting Agents to Financial Data

**Duration:** 60 minutes

**Learning Objectives:**
- Install and configure ChartBook
- Understand ChartBook's data organization (Pipelines, Dataframes, Charts)
- Create LangChain tools that wrap ChartBook API
- Experience why large data dictionaries stress context windows

---

## What is ChartBook?

[ChartBook](https://github.com/chartbook-project/chartbook) is a data science documentation platform that organizes financial data into a structured catalog:

| Concept | Description | Example |
|---------|-------------|---------|
| **Pipeline** | An analytics project | `yield_curve` |
| **Dataframe** | A dataset with metadata | `repo_public` |
| **Chart** | A visualization | `sofr_timeseries` |
| **Documentation** | Data dictionary, governance | Column definitions |

### Why ChartBook for This Course?

1. **Real-world data access pattern** - How financial firms organize data
2. **Data dictionaries are LARGE** - Perfect for testing context limits
3. **Structured catalog** - Enables programmatic discovery
4. **Finance-relevant datasets** - Treasury yields, repo rates, SOFR

---

## Part 1: Installation & Configuration (20 min)

### Install ChartBook

```bash
pip install chartbook
```

### Clone Sample Catalogs

Your instructor will provide repository URLs for sample financial data catalogs:

```bash
# Clone the catalogs (instructor provides URLs)
git clone [catalog-repo-1]
git clone [catalog-repo-2]
```

### Configure ChartBook

ChartBook needs to know where to find your catalogs:

```bash
# Interactive configuration
chartbook config

# Or set via environment variable
export CHARTBOOK_CATALOG_PATH=/path/to/catalogs
```

### Verify Installation

```bash
# List all pipelines
chartbook ls

# List all dataframes
chartbook ls dataframes

# Get documentation for a specific dataframe
chartbook data get-docs --pipeline yield_curve --dataframe repo_public
```

### Expected Output

```
$ chartbook ls dataframes
┌──────────────────┬─────────────────┬────────────┐
│ Pipeline         │ Dataframe       │ Records    │
├──────────────────┼─────────────────┼────────────┤
│ yield_curve      │ repo_public     │ 9,132      │
│ yield_curve      │ treasury_yields │ 12,450     │
│ money_markets    │ sofr_daily      │ 1,825      │
│ money_markets    │ effr_daily      │ 1,825      │
└──────────────────┴─────────────────┴────────────┘
```

---

## Part 2: Explore ChartBook API (15 min)

### Python API Basics

```python
from chartbook import data

# Load a dataframe
df = data.load(pipeline="yield_curve", dataframe="repo_public")
print(f"Shape: {df.shape}")  # (9132, 44) - 44 columns!

# Get column names
print(df.columns.tolist())

# Preview data
print(df.tail(5))
```

### The Documentation Problem

```python
# Get documentation (THIS IS LARGE!)
docs = data.get_docs(pipeline="yield_curve", dataframe="repo_public")

print(f"Documentation length: {len(docs)} characters")
# Often 5000+ characters!

print(docs[:500])  # Preview
```

**Sample documentation output:**
```
# repo_public

Federal Reserve Repo Operations Data

## Column Definitions

| Column | Type | Description |
|--------|------|-------------|
| observation_date | date | The date of the observation |
| total_amt_accepted | float | Total amount accepted in millions USD |
| total_amt_submitted | float | Total amount submitted in millions USD |
| operation_type | str | Type of repo operation (overnight, term) |
| award_rate | float | The awarded rate for the operation |
| high_rate | float | Highest rate submitted |
| low_rate | float | Lowest rate submitted |
| bid_to_cover | float | Ratio of submitted to accepted |
... [continues for 44 columns]
```

### Discussion Question

> **If each dataframe's docs are 5000+ characters, and we query 3 dataframes, how quickly do we fill a 128K context window?**
>
> Do the math: 5000 chars × 3 = 15,000 chars ≈ 4,000 tokens. After 30 dataframe queries, you've used ~120K tokens just for documentation.

---

## Part 3: Create ChartBook Tools (15 min)

Let's create LangChain tools that wrap the ChartBook API.

### Tool 1: chartbook_list

```python
from langchain_core.tools import tool
import subprocess

@tool
def chartbook_list() -> str:
    """List available ChartBook pipelines and dataframes.

    Returns a table of all available data sources with their record counts.
    Use this to discover what data is available before loading.
    """
    result = subprocess.run(
        ["chartbook", "ls", "dataframes"],
        capture_output=True,
        text=True
    )
    return result.stdout if result.stdout else "No dataframes found"
```

### Tool 2: chartbook_get_docs

```python
from chartbook import data
from typing import Annotated

@tool
def chartbook_get_docs(
    pipeline: Annotated[str, "Pipeline name (e.g., 'yield_curve')"],
    dataframe: Annotated[str, "Dataframe name (e.g., 'repo_public')"]
) -> str:
    """Get full documentation for a ChartBook dataframe.

    Returns column definitions, data types, and descriptions.
    WARNING: Returns large content - may need to save to file for reference.
    """
    try:
        docs = data.get_docs(pipeline=pipeline, dataframe=dataframe)
        return docs
    except Exception as e:
        return f"Error getting docs: {e}"
```

### Tool 3: chartbook_load_data

```python
@tool
def chartbook_load_data(
    pipeline: Annotated[str, "Pipeline name"],
    dataframe: Annotated[str, "Dataframe name"],
    columns: Annotated[str, "Comma-separated column names (optional)"] = None,
    limit: Annotated[int, "Maximum rows to return"] = 100
) -> str:
    """Load data from a ChartBook dataframe.

    Returns the most recent rows as a formatted table.
    Use 'columns' to select specific columns and 'limit' to control size.
    """
    try:
        df = data.load(pipeline=pipeline, dataframe=dataframe)

        if columns:
            col_list = [c.strip() for c in columns.split(",")]
            df = df[col_list]

        return df.tail(limit).to_string()
    except Exception as e:
        return f"Error loading data: {e}"
```

---

## Part 4: Integrate with FlawedCode (10 min)

### Add Tools to Agent

```python
from tools import (
    read_file, write_file, list_files, search_files, run_python,
    chartbook_list, chartbook_get_docs, chartbook_load_data  # NEW!
)

# All tools available to the agent
tools = [
    read_file,
    write_file,
    list_files,
    search_files,
    run_python,
    chartbook_list,       # NEW
    chartbook_get_docs,   # NEW
    chartbook_load_data,  # NEW
]

# Update system prompt
SYSTEM_PROMPT = """You are FlawedCode, a coding assistant with financial data access.

You have access to:
- File operations: read_file, write_file, list_files, search_files
- Code execution: run_python
- Financial data: chartbook_list, chartbook_get_docs, chartbook_load_data

When working with ChartBook data:
1. First use chartbook_list to see available dataframes
2. Use chartbook_get_docs to understand column definitions
3. Use chartbook_load_data to fetch actual data
4. Analyze or visualize with run_python

Be concise. Large data results may need to be saved to files.
"""
```

### Live Demo

```
flawed_code> List ChartBook dataframes
  [chartbook_list] {}
    → ┌──────────────────┬─────────────────┬────────────┐
       │ Pipeline         │ Dataframe       │ Records    │
       ...
Available dataframes: yield_curve/repo_public, money_markets/sofr_daily...

flawed_code> Get docs for yield_curve/repo_public
  [chartbook_get_docs] {"pipeline": "yield_curve", "dataframe": "repo_public"}
    → # repo_public
       Federal Reserve Repo Operations Data
       ## Column Definitions...
The repo_public dataframe contains 44 columns including observation_date,
total_amt_accepted, award_rate, SOFR rates...

flawed_code> What columns are related to repo rates?
Based on the documentation, the rate-related columns are:
- award_rate: The awarded rate for the operation
- high_rate: Highest rate submitted
- low_rate: Lowest rate submitted
- sofr_rate: Secured Overnight Financing Rate
- effr_rate: Effective Federal Funds Rate

flawed_code> Load 10 days of SOFR and EFFR data
  [chartbook_load_data] {"pipeline": "yield_curve", "dataframe": "repo_public",
                         "columns": "observation_date,sofr_rate,effr_rate",
                         "limit": 10}
    → observation_date  sofr_rate  effr_rate
       2024-01-15       5.31       5.33
       2024-01-16       5.32       5.33
       ...
Here's the last 10 days of SOFR and EFFR rates...
```

---

## Hands-On Exercise: ChartBook Exploration

### Task 1: Discover Available Data

```
"List all available ChartBook dataframes"
```

### Task 2: Understand a Dataset

```
"Get the documentation for the money_markets/sofr_daily dataframe"
```

### Task 3: Load and Analyze

```
"Load 30 days of SOFR data and calculate the average rate"
```

### Task 4: Multi-Dataset Query

```
"Compare SOFR and EFFR rates over the last week - which is higher on average?"
```

---

## The Context Problem Emerges

After several ChartBook queries, watch what happens:

### Sequence That Fills Context

1. `"List ChartBook dataframes"` → ~500 chars
2. `"Get docs for yield_curve/repo_public"` → ~5,000 chars
3. `"Get docs for money_markets/sofr_daily"` → ~3,000 chars
4. `"Load 50 rows of SOFR data"` → ~2,000 chars
5. `"Get docs for treasury_yields"` → ~4,000 chars
6. `"What was my original goal?"` → **Agent struggles!**

### What's Happening?

```
┌────────────────────────────────────────────────────────┐
│                   CONTEXT WINDOW                        │
├─────────────────────────────────────────────────────────┤
│ System prompt                           [1,000 tokens] │
│ User: "List dataframes"                 [10 tokens]    │
│ Tool result: dataframe list             [200 tokens]   │
│ User: "Get docs repo_public"            [15 tokens]    │
│ Tool result: FULL DOCUMENTATION         [1,500 tokens] │ ← BIG
│ User: "Get docs sofr_daily"             [15 tokens]    │
│ Tool result: FULL DOCUMENTATION         [1,000 tokens] │ ← BIG
│ User: "Load 50 rows"                    [20 tokens]    │
│ Tool result: data table                 [800 tokens]   │
│ User: "Get docs treasury"               [15 tokens]    │
│ Tool result: FULL DOCUMENTATION         [1,200 tokens] │ ← BIG
│ ...                                                    │
│ [EARLIER CONTEXT GETS PUSHED OUT]                      │
└────────────────────────────────────────────────────────┘
```

### Symptoms of Context Pressure

- Agent forgets earlier conversation
- Responses become less coherent
- Goal retention drops
- Token usage spikes

### Discussion Question

> **How would you handle a 10,000 character tool result?**
>
> You can't just truncate it—the details might be important. But you can't keep everything in context either.

---

## Preview: The Solution

This is the problem we solve with **context management**:

| Problem | Solution |
|---------|----------|
| Large tool results fill context | **Offload to filesystem** with preview |
| Context grows unbounded | **Summarize** at threshold |
| Lost information needed later | **Archive + search** for recovery |

We'll implement these techniques in [Context Management](context_management.md).

---

## Key Takeaways

1. **ChartBook provides structured access** to financial data catalogs
2. **Documentation size creates context pressure** - real data has real metadata
3. **Tools are easy to add** - just decorate functions with `@tool`
4. **Context limits become visible** with real-world data
5. **The problem is predictable** - we can design solutions

---

## Checkpoint Questions

- What's the largest tool result you've seen during this exercise?
- How would you decide what to keep in context vs. save to disk?
- What information is most important to preserve for goal retention?

---

## Next Steps

Now that we've experienced the context problem firsthand, let's learn how to solve it with [Context Management](context_management.md).
