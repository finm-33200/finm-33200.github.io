# Discussion 1: The New Quant - LLMs in Finance

**Duration:** 3 hours
**Format:** Hands-on workshop

In this opening session, we survey the transformative applications of LLMs in quantitative finance, then build our first sentiment classification pipeline. You'll see why scale matters, learn to call LLM APIs, extract structured data, and set up the development tools you'll use throughout the course.

## Learning Objectives

By the end of this session, you will be able to:

- **Describe 6+ applications** of LLMs in quantitative finance
- **Explain why scale matters** for financial sentiment (the Lopez-Lira finding)
- **Call LLM APIs** using OpenAI and OpenRouter
- **Extract structured data** from LLM responses using Pydantic
- **Use Claude Code** for basic coding tasks
- **Use Cursor** for AI-assisted development
- **Understand the measurement mindset** for evaluating AI systems

## What You'll Build

A **headline sentiment classifier** that can:
- Call GPT-4o-mini via the OpenAI API
- Return structured JSON with sentiment labels and confidence scores
- Process batches of financial news headlines
- Replicate the core methodology from Lopez-Lira (2023)

## Prerequisites

Complete [HW0](HW0.md) before class:
- Python 3.10+ (via Anaconda)
- Visual Studio Code
- Git
- Claude Code (`npm install -g @anthropic-ai/claude-code`)
- Cursor (from cursor.sh)
- API keys: OpenAI, OpenRouter, Anthropic Console
- Clone the [ai_inclass_examples](https://github.com/finm-33200/ai_inclass_examples) repo

## Session Outline

### Hour 1: The New Quant - LLMs Transforming Finance (60 min)

We begin with an exciting tour of LLM applications in finance, drawing from Fu (2025)'s comprehensive survey "The New Quant." This section showcases the breadth of what's possible before narrowing to our concrete implementation.

| Time | Topic | Activity |
|------|-------|----------|
| 0-10 min | **Course Overview** | What we'll build, how the abridged course is structured |
| 10-40 min | **Applications Showcase** | Tour of 6 application categories with paper highlights |
| 40-55 min | **Demo: Lopez-Lira in Action** | Live: classify 5 headlines with GPT-4o-mini |
| 55-60 min | **HW1 Preview** | Explain the replication assignment |

### Hour 2: Building the Pipeline (60 min)

Hands-on coding session where we build sentiment classification from scratch.

| Time | Topic | Activity |
|------|-------|----------|
| 0-15 min | **OpenAI API Setup** | Walk through hello.py, explain roles/messages |
| 15-30 min | **OpenRouter for Model Flexibility** | Multi-model access, cost awareness |
| 30-50 min | **Structured Outputs for Sentiment** | Pydantic schemas, headline classification |
| 50-60 min | **Benchmarking Preview** | Brief intro to promptfoo, measurement mindset |

### Hour 3: Developer Tools & Next Steps (60 min)

Set up your AI-powered development environment.

| Time | Topic | Activity |
|------|-------|----------|
| 0-25 min | **Claude Code** | Installation verification, demo, CLAUDE.md, basic commands |
| 25-45 min | **Cursor** | Quick demo, Cursor rules, comparison with Claude Code |
| 45-55 min | **HW1 Deep Dive** | Lopez-Lira replication scope, what you'll submit |
| 55-60 min | **Course Trajectory** | Preview of remaining weeks |

## Course Materials

```{toctree}
:maxdepth: 1
discussions/gen_ai_tech_stack.md
discussions/llms_transforming_finance.md
discussions/api_and_structured_outputs.md
discussions/ai_copilots_intro.md
```

## Additional Resources

### Documentation
- [OpenAI API Overview](https://platform.openai.com/docs/overview)
- [OpenRouter API](https://openrouter.ai/docs)
- [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
- [Cursor Documentation](https://docs.cursor.com/)

### Papers
- [Lopez-Lira (2023): Can ChatGPT Forecast Stock Price Movements?](https://arxiv.org/abs/2304.07619)
- [Fu (2025): The New Quant: A Survey of LLMs in Financial Prediction and Trading](https://arxiv.org/abs/2510.05533)
- [Wei (2022): Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)

### Repositories
- [In-Class Examples](https://github.com/finm-33200/ai_inclass_examples) - basic_llm_api/ directory
- [Lopez-Lira Exercise](https://github.com/finm-33200/ai_inclass_examples/tree/main/basic_llm_api/11_lopez_lira)

## Assessment

Students should be able to:
1. Explain why GPT-4 outperforms smaller models on financial sentiment (Lopez-Lira finding)
2. Make API calls to OpenAI and OpenRouter
3. Extract structured data using Pydantic schemas
4. Use Claude Code for basic coding tasks
5. Navigate Cursor for AI-assisted development

---

**Ready to get started?** Begin with [The Generative AI Tech Stack](discussions/gen_ai_tech_stack.md)!
