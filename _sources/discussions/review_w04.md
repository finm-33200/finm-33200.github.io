# Study Questions

Review questions for Week 4: Benchmarking and LLM Tool Use.

---

## Chain-of-Thought Prompting

1. Wei et al. (2022) introduced chain-of-thought (CoT) prompting. What does CoT replace in the standard few-shot prompting format, and why does showing intermediate reasoning steps improve performance on multi-step tasks?

2. CoT prompting is described as an "emergent ability" of large language models. What does this mean in terms of model scale? What happens when small models (e.g., 1B parameters) are prompted with reasoning chains?

3. On the GSM8K math benchmark, PaLM 540B with CoT prompting surpassed a fine-tuned GPT-3 model with a learned verifier --- despite requiring no training at all. Why is this result significant for the prompting-vs-fine-tuning debate?

4. For what types of tasks does CoT prompting provide the most benefit? For what types of tasks does it offer little improvement?

## ReAct: Reasoning and Acting

5. Yao et al. proposed the ReAct framework, which interleaves reasoning traces with actions. Describe the three components of the ReAct loop (Thought, Action, Observation) and explain how they work together.

6. What is the "grounding" advantage of ReAct? Does grounding guarantee factual correctness? Why or why not?

7. Name at least three modern AI tools that implement the ReAct paradigm. In what sense do they follow the Thought--Action--Observation loop?

8. What does "auditability" mean in the context of ReAct? Why is a visible trace of reasoning and actions valuable compared to a black-box prediction?

## Toolformer

9. Schick et al. (2023) introduced Toolformer, which teaches language models to use tools through self-supervised learning. Describe the three steps of Toolformer's training pipeline: candidate sampling, execution, and filtering.

10. In Toolformer's training pipeline, not every candidate tool call is kept. At a high level, how does Toolformer decide whether a tool call is useful enough to include? Why is it important that unhelpful tool calls are filtered out?

11. How does Toolformer differ from approaches that require human annotators to label where tools should be called? Why does this make Toolformer easier to extend to new tools?

12. A 6.7B-parameter Toolformer with access to tools outperformed the much larger GPT-3 (175B parameters) without tools on several benchmarks. What does this result tell us about the relationship between model scale and tool use?

## Retrieval-Augmented Generation (RAG)

13. Describe the two phases of a standard RAG pipeline: the preprocessing phase and the runtime phase. What role does the vector database play?

14. What is the chunking tradeoff in RAG? Explain why chunks that are too small and chunks that are too large each cause problems.

## Benchmarking LLMs

15. What does it mean for a benchmark to be "saturated"?

16. Explain the LLM-as-judge evaluation pattern. What problem does it solve, and how does it work?

17. What is data contamination in the context of LLM benchmarks? Research shows models suffer an average 39.4% performance drop when benchmark questions are rephrased. What does this suggest about reported benchmark performance?

18. In the course's benchmarking notebook, how did adding a calculator tool affect GPT-4o-mini's arithmetic performance? What does this demonstrate about the value of tool use for mathematical tasks?

## HW2 Required Reading

```{admonition} Note
:class: tip
You do not need to know the detailed math --- focus on the intuition and the high-level structure.
```

20. What does it mean that "directions in the token embedding space correspond to semantic meaning"? The video gives the example that the vector *woman* $\rightarrow $ *man* is similar to the vector *queen* $\rightarrow$ *king*. Explain what this tells us about how the model organizes word meanings geometrically.

21. What is a token embedding? How does it differ from the embeddings in the hidden layers? The video uses the word "mole" appearing in three different phrases --- "American shrew mole," "one mole of carbon dioxide," and "take a biopsy of the mole" --- to illustrate. Why does the token embedding fail to distinguish these three meanings, and how do the hidden-layer embeddings fix this?

22. A transformer processes a sequence of $N$ tokens and produces $N$ vectors in its final hidden layer --- that is, a matrix, not a single vector. Yet in HW2, models like BERT and `text-embedding-3-small` return a *single* vector for an entire headline or document (like the full-text news articles used in Chen, Kelly, and Xiu (2022)). What is pooling, and how does it collapse a matrix of per-token hidden states into one document-level embedding vector?

23. Describe the high-level structure of a transformer: embedding, followed by repeated attention + MLP blocks, followed by an unembedding step. What is the role of each stage? 

24. What is temperature in the context of sampling the next token? What happens when temperature is near 0? What happens when it is high? The video demonstrates this by generating stories at different temperatures --- describe the tradeoff.

25. The video describes how the word "tower" preceded by "Eiffel" should be updated to encode something more specific --- correlated with Paris, France, and things made of steel. If additionally preceded by "miniature," the vector should shift again so it no longer correlates with large, tall things. What does this example illustrate about what attention must be able to do?

26. Why does the size of the attention pattern scale as the *square* of the context length?
