# Study Questions

Review questions for Week 3: LLM Fundamentals.

---

## Lookahead Bias in LLMs

1. What is lookahead bias in the context of using pretrained LLMs for financial research? How does it differ from the classical definition of lookahead bias in backtesting?

2. Sarkar and Vafa (2024) formalize lookahead bias as $\text{Cov}(\hat{\mu}(X_t; M, \theta),\, \varepsilon_{t+1}) \neq 0$. Explain what each term represents and why a nonzero covariance indicates information leakage from the model's pretraining data.

3. Sarkar and Vafa (2024) identify two mechanisms of lookahead bias: *language leakage* and *selection bias*. Define each and give a concrete example from financial research.

4. When Sarkar and Vafa (2024) asked an LLM to assess risk factors from Zoom's 2019 earnings call, the output mentioned "COVID-19" --- a term that did not exist during the analysis period. Why does this occur, and why are simple prompting fixes ("only consider information up to 2019") insufficient?

## Chronologically Consistent LLMs

5. He et al. (2025) propose ChronoBERT and ChronoGPT as a solution to lookahead bias. What is the key design principle behind these models? What does the "time subscript" (e.g., ChronoBERT$_{2019}$) mean?

6. How do He et al. (2025) validate that their chronologically consistent models do not encode future knowledge? Describe the presidential election validation test and what it reveals about standard BERT vs. ChronoBERT.

7. ChronoBERT (149M parameters) achieves a long-short Sharpe ratio of 4.80, compared to Llama 3.1's 4.90 --- despite being 5--50x smaller. What does this result suggest about (a) the magnitude of lookahead bias in larger models for this application, and (b) the value of chronological consistency?

## Entity Neutering

8. Engelberg et al. (2025) propose entity neutering as an alternative to training new models. Describe the iterative process: mask, check, paraphrase, repeat. Why is simple firm-name masking insufficient (i.e., why can the LLM still identify Coca-Cola after removing "KO" from the text)?

9. After entity neutering, ChatGPT identifies the subject firm only 0.11% of the time, compared to 69.1% with simple firm-name masking. Yet entity neutering preserves sentiment 96--97% of the time. Explain why preserving sentiment while removing entity identity is the key requirement.

10. How are the chronologically consistent LLM approach and the entity neutering approach complementary? What type of leakage does each address?

## microGPT and nanochat

```{admonition} Note
:class: tip
You do not need to know the math or code behind microGPT or nanochat. Focus on what they are, what we did with them in class, and how they connect to the broader course themes.
```

11. What is Karpathy's microGPT? What task does it perform (what does it predict?), and what data was it originally trained on? What data did we retrain it on in class, and what did the model learn to generate?

12. What is Karpathy's nanochat? What is the goal of the nanochat speedrun?

## OpenAI API Platform

13. Why should you set a monthly spend limit on the OpenAI API platform before writing any code? What happens if you hit the limit?

14. API costs scale with token count (input + output). Given the pricing table in the notes, approximately how much would it cost to process 1 million input tokens with GPT-4o-mini vs. GPT-4o? Why does the notes recommend starting with cheaper models during development?

## Structured Outputs

15. LLMs produce free-form text by default. Why is this a problem when you need to feed the model's output into a downstream data pipeline (e.g., storing extracted financial data in a database)? What does structured output solve?

16. In the Lopez-Lira & Tang (2023) replication, students needed the model to return exactly one of three labels (YES, NO, UNKNOWN) for each headline. Without structured output, what could go wrong? How does constraining the model's output format at the decoding level differ from simply asking the model nicely in the prompt to return valid JSON?

## Function Calling / Tool Use

17. Explain the tool-calling loop: (1) send messages + tool definitions, (2) model returns `tool_calls`, (3) execute locally and append results, (4) repeat. Why does the model return `content = None` when it wants to use a tool?

18. Why does the API design require *your code* to execute tool calls rather than having the model execute them on the server? Discuss at least two of the three reasons (security, control, generality).

19. LLM knowledge has a training cutoff date. How does the web search tool address this limitation?

## HW1 Required Reading: Claude Code and FinBERT

20. What is Claude Code's *plan mode*? When working on a multi-step task like writing code to solve one fo the homeworks, why is it useful to have the agent outline a plan before it starts writing code?

21. The Anthropic talk "Don't Build Agents, Build Skills Instead" argues for structuring work with Claude Code using *skills* and *slash commands* rather than writing one large prompt that tries to do everything. What is a skill in this context? How does a skill differ from a `CLAUDE.md` file?

22. In Matt Pocock's "5 Claude Code skills I use every single day," the *grill me* skill tells Claude to "interview me relentlessly about every aspect of this plan until we reach a shared understanding," walking down each branch of a *design tree* and resolving dependencies between decisions one by one. What fundamental problem in LLM-assisted development is this skill trying to solve? Why is reaching a shared understanding with the LLM *before* it writes code important?
