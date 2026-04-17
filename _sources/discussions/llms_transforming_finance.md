# LLMs Transforming Finance: The New Quant

**Learning Objectives:**
- Survey 6 categories of LLM applications in quantitative finance
- Understand why model scale matters for financial tasks
- Connect research papers to practical tools
- Recognize the productivity gains from AI assistance

---

## The Emergence of the "New Quant"

> **📚 Literature: Fu, X. (2025). "The New Quant: A Survey of LLMs in Financial Prediction and Trading." arXiv:2510.05533**
>
> This comprehensive survey maps the landscape of LLM applications in finance. Fu organizes the field into six categories: sentiment and event extraction, numerical reasoning, multimodal understanding, retrieval-augmented generation, agentic systems, and domain-specific models. The key insight: LLMs are not replacing quants—they're augmenting them, creating a new type of practitioner who combines traditional quantitative skills with AI-powered analysis.

A growing body of research suggests the emergence of a "new quant"—practitioners who leverage large language models to amplify their analytical capabilities. Empirical evidence supports this shift:

| Study | Finding |
|-------|---------|
| Brynjolfsson (2025) | 15% average productivity increase for knowledge workers using AI assistance |
| Kwa (2025) | AI task-completion capability doubling every 7 months |

For quantitative finance practitioners who routinely process documents, analyze market data, write code, and synthesize research, these tools represent a fundamental change in how work gets done.

### The Generative AI Tech Stack

The generative AI ecosystem is organized in layers: cloud **infrastructure** at the base, **data stores** (including vector databases for embeddings), **foundation models** (GPT-4, Claude, LLaMA), **APIs** for programmatic access, **application frameworks** (LangChain, LlamaIndex) for orchestration, and **user interfaces** (ChatGPT, Cursor, Claude Code) at the top. As practitioners, you'll spend most of your time at the API and Application layers — calling models, structuring prompts, and building workflows — while infrastructure and model training remain abstracted away.

![Gen AI Stack](assets/gen_ai_stack3.jpg)

*The generative AI technology stack spans infrastructure, data, LLMs, APIs, and applications. For a deeper dive into each layer, see [The Generative AI Tech Stack](gen_ai_tech_stack.md).*

---

## Coding Agents & Copilots

The most visible LLM application you've likely encountered is the chatbot — ChatGPT, Claude.ai, Gemini. The second most common is the **coding agent**: AI tools like Claude Code, Cursor, and GitHub Copilot that write, edit, and reason about code alongside you. For quantitative finance practitioners who spend significant time writing code, these tools represent the most immediate productivity shift.

The change has been dramatic. Leading developers report that 70–90% of their code is now AI-generated, with the human role shifting from writing code to reviewing, guiding, and orchestrating AI output. This is no longer early adoption — the tipping point has passed. The question is no longer *whether* to use these tools, but *how well* you can wield them. The edge now comes from understanding what these tools can and can't do, and learning to think outside the box when they fall short.

### Stages of AI Coding Maturity

Some authors online have proposed a framework for thinking about the progression of AI-assisted coding. The stages map naturally onto an organizational metaphor: as you advance, your role shifts from individual contributor writing code to supervisor delegating tasks to director managing an organization of agents.

| Stage | Description | Role |
|-------|-------------|------|
| 1–4 | Code completions → IDE-integrated agents (Copilot, Cursor) | Individual contributor |
| 5 | CLI single agent, YOLO mode (e.g., Claude Code with auto-accept) | Senior IC |
| 6 | CLI multi-agent: 3–5 parallel instances on different tasks | Supervisor |
| 7 | 10+ hand-managed agents across worktrees | Director |
| 8 | Build your own orchestrator coordinating agents programmatically | Executive |

Most developers today are somewhere in Stages 1–5. Stages 6–7 involve running multiple coding agents in parallel — for example, several Claude Code instances in separate terminal panes, each tackling a different feature or bug:

![Multiple Claude Code instances running in parallel](assets/built-a-tmux-sidebar-that-shows-sessions-windows-panes-as-a-v0-kadzodajp6pg1.gif)

*Running multiple Claude Code instances in parallel across tmux panes — an example of Stages 6–7, where the developer supervises several agents working on different tasks simultaneously.*

At Stage 8, developers build dedicated orchestrators. Tools like OpenClaw coordinate multiple coding agents — OpenAI Codex, Claude Code, Gemini CLI — dispatching tasks automatically, running results through CI/CD pipelines, and validating output without manual intervention. The human becomes the "overseer" of an organization of agents.

![OpenClaw Architecture](assets/openclaw_org_chart.jpg)

*OpenClaw orchestrates multiple coding agents with CI/CD integration, illustrating Stage 8 — the developer as executive managing an organization of autonomous agents. Source: [@elvissun](https://x.com/elvissun)*

### AutoResearch: Agents Beyond Code

Coding agents don't just write application code — they can run entire research loops. Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) takes this further: an AI agent reads your training code, forms hypotheses for improvement, modifies the code, runs fixed-length training experiments, and evaluates whether the changes helped — all autonomously. It can run approximately 100 experiments overnight on a single GPU.

[![Karpathy's autoresearch announcement](../assets/karpathy_autoresearch_tweet.jpg)](https://x.com/karpathy/status/2030371219518931079)

*Andrej Karpathy's announcement of autoresearch. [View original tweet.](https://x.com/karpathy/status/2030371219518931079)*

For quantitative finance, this framework could be applied directly to financial time series forecasting — systematically searching over model architectures, feature engineering approaches, and hyperparameters for strategies like those in the [Financial Time Series Forecasting Repository (FTSFR)](https://jeremybejarano.com/ftsfr/).

### The New Programming Paradigm

Andrej Karpathy — founding member of OpenAI, former head of AI at Tesla, and one of the most respected AI researchers and practitioners in the field — captured the current moment in a widely shared post. If someone with Karpathy's depth of experience feels behind, it's a sign that we're all figuring this out together:

> I've never felt this much behind as a programmer. The profession is being dramatically refactored as the bits contributed by the programmer are increasingly sparse and between. I have a sense that I could be 10X more powerful if I just properly string together what has become available over the last ~year and a failure to claim the boost feels decidedly like skill issue. There's a new programmable layer of abstraction to master (in addition to the usual layers below) involving agents, subagents, their prompts, contexts, memory, modes, permissions, tools, plugins, skills, hooks, MCP, LSP, slash commands, workflows, IDE integrations, and a need to build an all-encompassing mental model for strengths and pitfalls of fundamentally stochastic, fallible, unintelligible and changing entities suddenly intermingled with what used to be good old fashioned engineering. Clearly some powerful alien tool was handed around except it comes with no manual and everyone has to figure out how to hold it and operate it, while the resulting magnitude 9 earthquake is rocking the profession. Roll up your sleeves to not fall behind.
>
> — Andrej Karpathy ([source](https://x.com/karpathy/status/2004607146781278521))

*For hands-on setup of Claude Code and Cursor — the tools that put you at Stage 5 and beyond — see [AI Copilots: Claude Code and Cursor](ai_copilots_intro.md).*

---

## 1. Sentiment & Event Extraction

The most direct application of LLMs in finance: reading text and extracting trading signals.

### The Lopez-Lira & Tang Finding

> **📚 Literature: Lopez-Lira, A. & Tang, Y. (2023). "Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models."**
>
> This paper tested whether ChatGPT could predict stock returns by classifying news headlines as "good news," "bad news," or "uncertain." The headline finding: **GPT-4's sentiment scores correlate positively with next-day returns and outperform traditional sentiment metrics.** Crucially, smaller models like BERT showed no predictive power—only the largest models captured this return predictability. This "scale matters" insight explains why the field shifted from fine-tuned small models to prompting frontier models.

**Key insight:** The same headline classified by BERT and GPT-4 produces different signals—and only GPT-4's signal predicts returns.

### LLM Embeddings for Return Prediction

> **📚 Literature: Chen, L., Kelly, B., & Xiu, D. (2022). "Return Prediction with Large Language Models."**
>
> While Lopez-Lira & Tang use sentiment labels, Chen et al. use embeddings—the dense vector representations that LLMs produce internally. They find that LLM embeddings significantly outperform simpler NLP methods (bag-of-words, Word2Vec) and technical signals across 16 global equity markets and 13 languages. Benefits are most pronounced for articles featuring negation or complex narratives—exactly where simpler methods fail.

*We explore text representations — from bag-of-words to embeddings — in [Discussion 4: Text Representation & Embeddings](../discussion_04.md), and replicate a portion of this analysis in [Homework 2](../HW2.md).*

---

## 2. Numerical & Economic Reasoning

LLMs can do more than read—they can reason about quantitative problems.

### Chain-of-Thought Prompting

> **📚 Literature: Wei, J. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models."**
>
> This paper showed that prompting models to produce explicit step-by-step reasoning unlocks performance on math and logic problems. The technique is simple: instead of asking for an answer directly, you prompt with "Let's think step by step" or provide examples that show the reasoning process. Chain-of-thought is now standard practice in commercial LLM products and foundational for finance applications.

![Chain-of-Thought Prompting](assets/cot_standard_vs_reasoning.png)

*Standard prompting asks for direct answers, while chain-of-thought prompting elicits step-by-step reasoning. This technique is foundational for financial analysis tasks requiring multi-step logic. Source: [Google Research](https://research.google/blog/language-models-perform-reasoning-via-chain-of-thought/)*

### FinCoT: Finance-Specific Reasoning

> **📚 Literature: Nitarach, N. (2025). "FinCoT: Chain-of-Thought Reasoning for Financial Analysis."**
>
> FinCoT adapts chain-of-thought for finance by injecting expert-defined reasoning blueprints into prompts. Evaluated on CFA-style questions across financial domains, **FinCoT improved accuracy from 63% to 80% on a general 8B model**—demonstrating that careful prompt engineering can substantially enhance performance without fine-tuning.

**Practical takeaway:** Before fine-tuning, try better prompts. FinCoT achieves significant gains just by structuring the reasoning process.

---

## 3. Multimodal Understanding

Modern LLMs can process images, enabling new applications.

### Vision Models for Financial Documents

Vision-language models like GPT-4V and Claude 3 can:
- Read earnings tables directly from images
- Interpret financial charts and graphs
- Extract data from PDFs that resist text extraction
- Process historical documents with complex layouts

**Use case:** An analyst receives a quarterly report as a scanned PDF. Instead of manual data entry, a vision model extracts the key figures directly, maintaining the semantic context that pure OCR would lose.

---

## 4. Retrieval-Augmented Generation (RAG)

RAG connects LLMs to external knowledge bases, extending them beyond their training data.

### NeuSym-RAG for Financial Documents

> **📚 Literature: Cao, Y. (2025). "NeuSym-RAG: A Hybrid Neural-Symbolic Framework for Financial Document Q&A."**
>
> Financial documents combine structured tables with unstructured narrative—a challenge for pure vector search. NeuSym-RAG addresses this with a hybrid approach: neural retrieval for narrative sections, symbolic SQL-based retrieval for tabular data. The framework outperforms pure neural methods on financial document question answering.

![Augmented LLM](assets/augmented_llm.png)

*The augmented LLM building block combines retrieval, tools, and memory to extend model capabilities beyond training data. RAG systems like NeuSym-RAG build on this foundation. Source: [Anthropic](https://www.anthropic.com/research/building-effective-agents)*

### Asset Embeddings

> **📚 Literature: Gabaix, X. (2025). "Asset Embeddings: Large Language Models for Financial Asset Representation."**
>
> Just as words appearing in similar contexts get similar embeddings, assets held together by investors get similar *asset embeddings*. Gabaix applies the word embedding paradigm to financial assets: firms that appear together in portfolios receive similar vector representations. The follow-up work demonstrates that these firm embeddings explain credit spreads better than ratings or distance-to-default.

**Key insight:** The embedding techniques that power LLMs can represent assets, not just text—opening new approaches to similarity analysis and clustering.

---

## 5. Agentic Systems

Agents combine LLM reasoning with real-world actions, enabling autonomous workflows.

### Trading-R1: RL for Trading Decisions

> **📚 Literature: Xiao, Y. (2025). "Trading-R1: Reasoning-Guided LLM for Financial Trading."**
>
> While reasoning LLMs excel at math and coding, their unguided reasoning often drifts away from market-relevant analysis. Trading-R1 uses reinforcement learning to train LLMs for trading decisions, achieving **Sharpe ratios of 1.6–2.7 across major equities** while generating structured, evidence-backed investment theses. The model learns to focus its reasoning on the factors that actually predict returns.

### Deep Research Agents

> **📚 Literature: Du, L. (2025). "DeepResearch Bench: Evaluating Deep Research Agents."**
>
> Deep Research Agents produce analyst-grade, citation-rich reports by autonomously searching, reading, and synthesizing information. Du introduces the first comprehensive benchmark for these systems. ChatGPT's Deep Research mode exemplifies this capability—give it a complex question, and it conducts multi-step research before producing a coherent report.

### The ReAct Pattern

> **📚 Literature: Yao, S. "ReAct: Synergizing Reasoning and Acting in Language Models."**
>
> ReAct established the paradigm that underlies every modern agent: interleaving reasoning traces ("thoughts") with actions. When an LLM explains its reasoning before acting, it makes fewer errors and recovers better from mistakes. This pattern powers Claude Code, ChatGPT's tool use, and all major agentic frameworks.

![ReAct Pattern](assets/react_overview.png)

*The ReAct pattern interleaves reasoning traces (thoughts) with actions, enabling agents to explain their logic before acting. This pattern underlies modern agent frameworks including Claude Code and ChatGPT's tool use. Source: [Google Research](https://research.google/blog/react-synergizing-reasoning-and-acting-in-language-models/)*

---

## 6. Domain-Specific Models

Some applications require models trained specifically on financial data.

### BloombergGPT

> **📚 Literature: Wu, S. (2023). "BloombergGPT: A Large Language Model for Finance."**
>
> Bloomberg trained a 50-billion parameter model on their proprietary dataset of 363 billion tokens of financial text. BloombergGPT significantly outperforms general-purpose LLMs on financial tasks while maintaining competitive performance on standard benchmarks. The model demonstrates that domain-specific pre-training can yield substantial improvements.

### Open-Source Alternative: FinGPT

While BloombergGPT remains proprietary, the AI4Finance Foundation has released FinGPT as an open-source alternative:

![FinGPT Framework](assets/fingpt_framework.png)

*FinGPT's full-stack framework demonstrates how practitioners can build domain-specific financial LLMs using open-source tools and lightweight fine-tuning techniques like LoRA. Source: [AI4Finance Foundation](https://github.com/AI4Finance-Foundation/FinGPT)*

### ChronoGPT: Eliminating Lookahead Bias

> **📚 Literature: He, X., Lv, J., Manela, A., & Wu, R. (2025a). "ChronoGPT: Chronologically Consistent Language Models."**
>
> Standard LLMs are trained on text that includes future information relative to any historical backtest date—creating data leakage. ChronoGPT trains only on text available at each point in time, eliminating lookahead bias. For backtesting applications, this is critical: using a model trained on 2024 data to analyze a 2020 headline introduces forward-looking information that wouldn't have existed at the time.

> **📚 Literature: He, X., Lv, J., Manela, A., & Wu, R. (2025). "Instruction-Tuning ChronoGPT for Chat Capabilities."**
>
> The follow-up work adds instruction tuning to ChronoGPT, creating chat-capable models that maintain chronological consistency. Each model has a clearly defined knowledge cutoff date, enabling realistic historical analysis.

**Key insight for practitioners:** When backtesting LLM-based strategies, standard models introduce forward-looking bias. ChronoGPT offers a methodologically sound alternative.

---

## What This Means for Quant Practitioners

The papers above suggest several practical implications:

1. **Scale matters.** The Lopez-Lira & Tang finding—that only frontier models capture return-predictive sentiment—means practitioners should use the best available models for signal extraction, not fine-tuned small models.

2. **Prompts before fine-tuning.** FinCoT's 17-percentage-point improvement from better prompts suggests exhausting prompt engineering before investing in fine-tuning.

3. **Hybrid approaches work.** NeuSym-RAG's combination of neural and symbolic retrieval outperforms either alone—a pattern likely to generalize across financial applications.

4. **Agents are ready.** Trading-R1's Sharpe ratios and Deep Research's analyst-grade reports demonstrate that agentic systems can perform real financial work, not just demos.

5. **Temporal consistency matters.** ChronoGPT addresses a methodological problem that most practitioners haven't considered: standard LLMs leak future information into historical analysis.

---

## Discussion Questions

1. **Scale vs. fine-tuning:** Given the Lopez-Lira & Tang finding, when would you still choose to fine-tune a smaller model rather than prompt a frontier model?

2. **Lookahead bias:** How would you design a backtesting framework that uses LLMs without introducing forward-looking bias? Would you always use ChronoGPT, or are there cases where standard models are acceptable?

3. **Agentic trading:** Trading-R1 achieves strong Sharpe ratios in research settings. What additional challenges would you face deploying such a system in production?

4. **RAG vs. fine-tuning:** When building a financial Q&A system, how would you decide between RAG (retrieving from documents) and fine-tuning on financial text?

---

## Hands-On: Live Demo

Let's see Lopez-Lira & Tang's core methodology in action. We'll classify 5 headlines with GPT-4o-mini and examine the sentiment scores:

```python
from openai import OpenAI

client = OpenAI()

headlines = [
    "Apple Reports Record Q4 Earnings, Beats Analyst Expectations",
    "Fed Signals More Rate Hikes Ahead, Markets Tumble",
    "Tesla Stock Plunges After Disappointing Delivery Numbers",
    "Microsoft Cloud Revenue Grows 27%, Exceeding Forecasts",
    "Bank of America Warns of Recession Risk in 2024",
]

for headline in headlines:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a financial analyst. Classify news as 'good news', 'bad news', or 'uncertain'."},
            {"role": "user", "content": f"Headline: {headline}"},
        ],
    )
    print(f"{headline[:50]}... → {response.choices[0].message.content}")
```

---

## In-Class Examples

All in-class exercises and code examples for this course are available in the [ai_inclass_examples](https://github.com/finm-33200/ai_inclass_examples) repository. To get started with the OpenAI API and run the Lopez-Lira sentiment classification example above, see the [basic OpenAI hello example](https://github.com/finm-33200/ai_inclass_examples/tree/main/basic_llm_api/02_openai_hello).

---

## Next Steps

Now that you've seen the landscape of LLM applications in finance, let's build our sentiment classification pipeline from scratch. Continue to [API and Structured Outputs](api_and_structured_outputs.md).
