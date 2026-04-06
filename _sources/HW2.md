# Homework 2: Chen, Kelly, Xiu (2022)

- **GitHub Classroom Repository:** https://classroom.github.com/a/1T15YUwT
- **Due Date:** Tuesday, April 14, 2026 at 11:59 PM

**Replicating Chen, Kelly & Xiu (2022) with News Headlines**

To get started with this homework, accept the GitHub Classroom assignment and clone the repository found above. All detailed instructions, deliverables, and grading criteria are in the homework repository README.

In this assignment, you will replicate a portion of the analysis from Chen, Kelly, and Xiu (2022), "Expected Returns and Large Language Models." This paper demonstrated that LLM embeddings—dense vector representations of news text—significantly outperform simpler NLP methods for predicting stock returns. You will build a pipeline that converts financial news headlines into embeddings and uses them to predict stock returns via rolling-window supervised learning. As in Homework 1, you will scrape your own headlines and merge them with RavenPack data, sending only your own headlines to OpenAI for embedding.

> **Chen, L., Kelly, B. T., & Xiu, D. (2022). "Expected Returns and Large Language Models."** [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4416687)

## Key Simplifications

You will not replicate the full paper. Key differences from the original:

| Original Paper | Your Replication |
|----------------|-----------------|
| Full article text (Thomson Reuters) | Scraped headlines merged with RavenPack |
| 16 global equity markets | US equities only |
| Many models (BERT, GPT, LLaMA, Word2Vec, BoW) | Three representations: BERT, `text-embedding-3-small`, and TF-IDF |
| Tables 1–8 | Tables 1, 2, and 3 only |

See the solution notebooks below for a full walkthrough of the data, methodology, and results.

---

## Acknowledgments

This case study is based on a class project originally developed by
**Andrew Moukabary** and **Reece VanDeWeghe** for FINM 32900.
The current version has been adapted for use as a teaching case study.
Credit for the original pipeline design, data cleaning logic, and
analytical framework belongs to the original authors.

---

## Required "Reading"

```{note}
The material in this section will be covered on the midterm.
```

```{important}
All course notes from the corresponding discussion pages are also required reading. We may not cover every page during class — you are responsible for reading the rest on your own. See the [Exam Preparation](exam_prep.md) page for details.
```

The following videos are meant to solidify your understanding of the fundamentals of how large language models work. The goal is not to get deep into the math — it is to build a general, intuitive understanding of what an LLM is doing on the inside. Having this basic insight into the architecture will prove useful throughout the rest of the course.

### 3Blue1Brown — "But what is a GPT?"

<iframe width="560" height="315" src="https://www.youtube.com/embed/wjZofJX0v4M?si=a6JrtjzS4nOWQFvE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### 3Blue1Brown — "Attention in Transformers, Visually Explained"

<iframe width="560" height="315" src="https://www.youtube.com/embed/eMlx5fFNoYc?si=hKXu9FUpnx8vlvz0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### 3Blue1Brown — "How might LLMs store facts"

<iframe width="560" height="315" src="https://www.youtube.com/embed/9-Jl0dxWQs8?si=C0v8uP-veZbz3VPY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

---

## Solution Notebooks

```{toctree}
:maxdepth: 1

notebooks/01_explore_data_ipynb
notebooks/02_embeddings_demos_ipynb
notebooks/03_methodology_ipynb
notebooks/04_results_ipynb
```