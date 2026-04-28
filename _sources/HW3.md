# Homework 3: Benchmarking RAG with SEC Filings

- **GitHub Classroom Repository:** https://classroom.github.com/a/3JZeJvZ-
- **Due Date:** Friday May 9, 2026 at 11:59 pm

To get started with this homework, accept the GitHub Classroom assignment and clone the repository found above. **All detailed instructions, deliverables, and grading criteria are in the homework repository README.**

You will build a **Retrieval-Augmented Generation (RAG)** pipeline against SEC 10-K filings and measure how much retrieval is worth — comparing a bare-question baseline against three RAG variants on Compustat ground truth (revenue, net income, total assets).

## Prerequisites

- WRDS account with SEC Analytics Suite + Compustat access
- OpenAI API key
- Python 3.12+

See the homework README for setup, pipeline commands, and the autograded test details.

---

## Solution Notebooks

```{toctree}
:maxdepth: 1

notebooks/01_data_overview
notebooks/02_rag_pipeline_walkthrough
notebooks/03_rag_benchmark
```
