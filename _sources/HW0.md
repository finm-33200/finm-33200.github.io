# Homework 0: Set Up Your Environment

- **GitHub Classroom Repository:** https://classroom.github.com/a/emovA-K3
- **Due Date:** Tuesday, March 31, 2026 at 11:59 PM

**Setting up your computing environment**

To receive credit for this homework, please complete the following steps and affirm that you have completed it by indicating it in the code in the HW repository found above.

---

## Software to Install

Install the following software on your laptop (all free):

### Core Development Tools

- **[Anaconda](https://www.anaconda.com/download)** - Python 3.10+ distribution with data science packages
- **[Visual Studio Code](https://code.visualstudio.com/)** - Code editor (NOT Visual Studio, which is different)
  - Free AI features via GitHub Copilot — sign up through [GitHub Education](https://github.com/education/students) with your .edu email. See [Copilot student access](https://docs.github.com/en/copilot/how-tos/manage-your-account/free-access-with-copilot-student).
- **[Cursor](https://cursor.com/)** - AI-native code editor
  - Free 1-year Pro plan for students at [cursor.com/students](https://cursor.com/students). Verify with your .edu email.
  - Imports VS Code settings automatically
- **[Git](https://git-scm.com/)** - Version control
- **[GitKraken](https://www.gitkraken.com/)** - Git GUI (free for students via [GitHub Student Pack](https://www.gitkraken.com/github-student-developer-pack))
- **[GitHub CLI](https://cli.github.com/)** - Command-line interface for GitHub (managing repos, PRs, issues, etc.)
- **[TeX Live](https://tug.org/texlive/)** - Only needed if you want to compile LaTeX documents

I want you to try both VS Code and Cursor. Cursor has been one of the most important tools in AI-assisted coding over the last two years and you should be familiar with it. That said, **do not spend money on Cursor** — the free student plan is more than sufficient. We will use Claude Code as our primary AI coding tool in this course.

### AI Development Tools

- **[Claude Code](https://code.claude.com/docs/en/quickstart)** - Anthropic's agentic coding tool. Follow the [quickstart guide](https://code.claude.com/docs/en/quickstart) for installation instructions.


---

## Accounts to Create

Create accounts with the following services (free tiers available):

### Required

| Service | URL | Notes |
|---------|-----|-------|
| **GitHub** | [github.com](https://github.com/) | Code hosting, assignment submission |
| **OpenAI Platform** | [platform.openai.com](https://platform.openai.com/) | API access (don't add credits yet) |
| **Claude** | [claude.ai](https://claude.ai/) | Claude Code subscription ([quickstart](https://code.claude.com/docs/en/quickstart)) |
| **OpenRouter** | [openrouter.ai](https://openrouter.ai/) | Multi-model API access (don't add credits yet) |

For HW0, just create these accounts. We will add API credits and subscriptions when needed in later assignments. See the Course Budget section below for details.

### WRDS (Financial Data)

We will use [WRDS](https://wrds-www.wharton.upenn.edu/) for financial data access. If you are a Financial Mathematics student, you should be able to apply for your own personal account through the [WRDS registration page](https://wrds-www.wharton.upenn.edu/register/). If you are not, you can use the class account code. The code for this class account will be posted on Canvas.

For WRDS access issues, contact the UChicago Mathematics department representative: John Zekos (zekos@math.uchicago.edu).

---

## Submission

Once you have completed the steps above, submit this homework via GitHub using the homework repository link at the top of this page.

---

## Course Budget

Just like other courses require textbooks, this course requires AI tools. Instead of textbooks, you are investing in AI usage, API credits, and a Claude Code plan.

### Claude Code (Anthropic)

Claude Code requires a Claude subscription. You are welcome to start with the **Pro plan (\$20/month)**, but you will quickly run out of usage. I **strongly recommend the Max plan (\$100/month)** for a smooth experience throughout the quarter.

### API Credits

Budget approximately **\$100 for API usage** over the quarter. We will try to use less than that, but plan accordingly. This covers:

- **OpenAI API** — the most feature-rich LLM API and an important one to know
- **OpenRouter** — allows us to try many different models through a single API

You are responsible for your own API keys and usage. If you exceed this budget, that is your own responsibility.

---

## Required "Reading"

```{note}
The material in these videos will be covered on the midterm.
```

```{important}
All course notes from the corresponding discussion pages are also required reading. We may not cover every page during class — you are responsible for reading the rest on your own. See the [Exam Preparation](exam_prep.md) page for details.
```

### Andrej Karpathy — "How I Use LLMs"

Andrej Karpathy is one of the most influential figures in modern AI. He earned his PhD at Stanford under Fei-Fei Li, was a founding member of OpenAI, and served as Director of AI at Tesla. He later founded Eureka Labs, an AI education startup. TIME named him one of the 100 Most Influential People in AI in 2024. He is widely known for making deep learning concepts accessible to a broad audience.

In this video, Karpathy walks through the entire LLM ecosystem with practical, real-world examples from his own workflow.

<iframe width="560" height="315" src="https://www.youtube.com/embed/EWvNQjAaOHw?si=TBRYHi-eSshhMP02" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### Cursor — AI-Native Code Editor

Cursor is a fork of VS Code that became the breakout AI coding tool by deeply integrating LLM-powered autocomplete, chat, and code generation directly into the editor. Its success came from meeting developers where they already were — in VS Code — and making AI assistance feel native rather than bolted on.

This video walks through using Cursor's AI features in practice. **For HW0, try Cursor and use its AI agents as much as possible.** Do not pay for it — you get a free year of Cursor Pro through the [Student Plan](https://cursor.com/students) by verifying with your .edu email.

<iframe width="560" height="315" src="https://www.youtube.com/embed/WVeYLlKOWc0?si=VLs6tqFTlYhvlz6j" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

---

## News Headlines Notebooks

```{toctree}
:maxdepth: 1
notebooks/01_data_sources_overview_ipynb.ipynb
notebooks/02_gdelt_sp500_filtering_ipynb.ipynb
notebooks/03_crosswalk_quality_ipynb.ipynb
```

## Additional Resources

See the [Appendix](appendix.md) for video tutorials on setting up Python and Visual Studio Code.
