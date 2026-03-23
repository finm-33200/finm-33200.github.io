# Discussion 2: Building and Benchmarking Coding Agents

**Duration:** 3 hours
**Format:** Hands-on workshop

In this workshop, we'll build a complete coding assistant from first principles, then benchmark how context management improves agent performance. You'll progress from basic ReAct loops to a functional "FlawedCode" assistant, then learn to systematically measure agent quality.

## Learning Objectives

By the end of this session, you will be able to:

- **Understand the ReAct pattern** that powers all modern AI agents
- **Build agents progressively** from raw loops to framework-based implementations
- **Implement tool use** for file operations and code execution
- **Recognize context limitations** in multi-turn agent conversations
- **Design benchmarks** that measure agent capabilities objectively
- **Compare agent versions** using systematic evaluation

## What You'll Build

A **FlawedCode** assistant (minimal Claude Code clone) that can:
- Read and write files in a project
- Execute Python code safely
- Navigate and search codebases
- Remember conversation context
- Demonstrate both capabilities and limitations

![Claude Code in Action](discussions/assets/claude_code_product.jpg)

*Claude Code - the production assistant we're building toward. Source: [Anthropic](https://www.anthropic.com/claude-code)*

## Prerequisites

- Python 3.10+
- OpenRouter API key ([get one here](https://openrouter.ai/))
- ChartBook installed (we'll set this up in class)
- Familiarity with LangChain basics (helpful but not required)

## Session Outline

### Hour 1: Agent Walkthrough & Building "FlawedCode" (60 min)
Progressive tour through agent examples, culminating in a working coding assistant.

### Hour 2: ChartBook Setup and Exploration (60 min)
Install and explore ChartBook for real financial data integration.

### Hour 3: Context Management & Benchmarking (60 min)
Understand context limitations and build systematic agent evaluations.

## Course Materials

```{toctree}
:maxdepth: 1
discussions/agent_fundamentals.md
discussions/building_flawed_code.md
discussions/chartbook_integration.md
discussions/context_management.md
discussions/agent_benchmarking.md
```

## Additional Resources

### Documentation
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Context Management Blog](https://blog.langchain.com/context-management-for-deepagents/)
- [OpenRouter API](https://openrouter.ai/docs)
- [ReAct Prompting Guide](https://www.promptingguide.ai/techniques/react)

### Repositories
- [In-Class Examples](https://github.com/finm-33200/ai_inclass_examples) - agents/ directory
- [FlawedCode](https://github.com/finm-33200/ai_inclass_examples/tree/main/agents/08_flawed_code)

### Papers
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629) - Yao et al.
- [SWE-bench: Agent Evaluation](https://arxiv.org/abs/2310.06770) - Jimenez et al.

## Project Extensions

After completing the base workshop, consider extending your implementation:

1. **Context Offloading** - Save large tool results to disk
2. **Conversation Summarization** - Compress context at threshold
3. **ChartBook Integration** - Add financial data tools
4. **Custom Benchmarks** - Design finance-specific evaluation tasks
5. **Model Comparison** - Compare cheap vs expensive models

## Assessment

Students should be able to:
1. Explain the ReAct agent pattern
2. Build a working FlawedCode v1 assistant
3. Identify context limitations in multi-turn conversations
4. Design at least 3 benchmark tasks with success criteria
5. Discuss tradeoffs in context management strategies

---

**Ready to get started?** Begin with [Agent Fundamentals](discussions/agent_fundamentals.md)!
