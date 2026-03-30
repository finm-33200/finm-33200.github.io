# Homework 1: Lopez-Lira, Tang (2023)

- **GitHub Classroom Repository:** https://classroom.github.com/a/lxSMjzYe
- **Due Date:** Tuesday, April 7, 2026 at 11:59 PM

**Replicating Sentiment-Based Return Prediction**

To get started with this homework, accept the GitHub Classroom assignment and clone the repository found above.

In this assignment, you will replicate the core methodology from Lopez-Lira & Tang (2023), "Can ChatGPT Forecast Stock Price Movements?" You will build a sentiment classification pipeline using the OpenAI API to classify financial news headlines and evaluate whether LLM-based sentiment predictions correlate with stock returns.

> **Lopez-Lira, A. & Tang, Y. (2023). "Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models."** [arXiv:2304.07619](https://arxiv.org/abs/2304.07619)

All detailed instructions, deliverables, and grading criteria are in the homework repository.

---

## Required "Reading"

```{note}
The material in these videos will be covered on the midterm.
```

```{important}
All course notes from the corresponding discussion pages are also required reading. We may not cover every page during class — you are responsible for reading the rest on your own. See the [Exam Preparation](exam_prep.md) page for details.
```

The following videos will get you fully up to speed with Claude Code, Anthropic's agentic coding tool. **For HW1, you should try to complete as much of the assignment as possible using Claude Code.** Learning to work effectively with an AI coding agent is a core skill in this course, and this homework is your first real opportunity to practice it.

### "Claude Code Setup for Beginners"

A beginner-friendly guide to getting Claude Code installed and configured. Start here before watching the other videos.

<iframe width="560" height="315" src="https://www.youtube.com/embed/qYqIhX9hTQk?si=uD71CVdm8L5DrFsk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### "I was an AI skeptic. Then I tried plan mode"

A practical introduction to Claude Code's plan mode — a structured way to break down complex tasks before writing code. Plan mode is especially useful for assignments like this one where you need to think through a multi-step pipeline before implementing it.

<iframe width="560" height="315" src="https://www.youtube.com/embed/WNx-s-RxVxk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### "Don't Build Agents, Build Skills Instead" — Barry Zhang & Mahesh Murag, Anthropic

An Anthropic talk on how to think about structuring work with Claude Code using skills and slash commands rather than monolithic agent prompts. Understanding this mental model will help you use Claude Code more effectively throughout the course.

<iframe width="560" height="315" src="https://www.youtube.com/embed/CEvIs9y1uog" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### "5 Claude Code skills I use every single day"

A quick tour of the most useful Claude Code features and habits for daily use. These are the practical tips that will save you the most time on this and future assignments.

<iframe width="560" height="315" src="https://www.youtube.com/embed/EJyuu6zlQCg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### "The pipeline function"

This video introduces the Hugging Face Transformers `pipeline` function, which provides a simple interface for running pre-trained models. Watch this to get some context for how to use [FinBERT](https://huggingface.co/ProsusAI/finbert), a BERT model fine-tuned for financial sentiment analysis, in the homework.

<iframe width="560" height="315" src="https://www.youtube.com/embed/tiZFewofSLM?si=3kO8UhjuBMlexzhd" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

---

## Solution Notebooks

```{toctree}
:maxdepth: 1

notebooks/01_data_overview_ipynb
notebooks/02_methodology_ipynb
notebooks/03_replication_results_ipynb
```