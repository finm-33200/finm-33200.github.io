# Study Questions

Review questions for Week 1: The New Quant -- LLMs in Finance.

---

## LLMs in Finance

1. What are the six categories of LLM applications in quantitative finance identified by Fu (2025)? For each, give one concrete example relevant to a quant practitioner.

2. What does it mean to say that LLMs are "statistical prediction models"? How does the Caddyshack example from Chen, Kelly & Xiu (2022) illustrate this?

3. Explain why GPT's training on 500 billion observations and approximately one trillion parameters enables it to capture subtle patterns in language. How does this contrast with the typical data-to-parameter ratio in financial econometrics?

## Scale and Financial Reasoning

4. In Lopez-Lira & Tang (2023), GPT-4's headline sentiment classification achieves a Sharpe ratio of 3.28, while GPT-1, GPT-2, and FinBERT achieve *negative* Sharpe ratios. What does this tell us about the relationship between model scale and financial reasoning?

5. Lopez-Lira & Tang (2023) find that the annualized Sharpe ratio of the GPT-4 news strategy has been declining over time. Explain why this is consistent with the hypothesis that markets are learning.

6. Why does the sample period in Lopez-Lira & Tang (2023) start in October 2021? What methodological concern does this design choice address?

## Prompting vs. Embeddings

7. Compare and contrast the prompting approach (Lopez-Lira & Tang 2023) with the embedding approach (Chen, Kelly & Xiu 2022) for extracting trading signals from news text. Discuss the bias-variance tradeoff between the two.

8. Chen, Kelly & Xiu (2022) describe the embedding approach as "fishing with a net" versus prompting as "fishing with a rod." Explain this analogy and its implications for portfolio construction.

9. In Chen, Kelly & Xiu (2022), why do LLM embeddings outperform bag-of-words methods on articles containing negation? Use the British Airways/Brexit example to illustrate.

10. Chen, Kelly & Xiu (2022) find that an equal-weighted ensemble of multiple LLMs achieves a higher Sharpe ratio (5.11) than any individual model. Why might different language models produce partially correlated but distinct trading strategies?

## Paper Identification

11. The figure below shows cumulative returns of a long-short portfolio based on LLM sentiment scores, starting from \$1. Which paper does this figure come from, and what is the main finding it illustrates?

    ![Cumulative returns of investing $1](assets/lopez_lira/slide_24.png)

12. The figure below shows average next-day returns by prediction score across different models. Which paper does this come from, and what pattern does it reveal about model scale?

    ![Average next day's return by prediction score across models](assets/lopez_lira/slide_37.png)

13. The figure below contrasts two approaches to extracting information from LLMs. Which paper presents this comparison, and what are the two approaches?

    ![Contrasting prompts and embeddings](assets/kelly_expected_returns/slide_10.png)

## Polyglot and Multi-Frequency Results

14. Chen, Kelly & Xiu (2022) find that the embedding-based strategy works across countries and languages. What pattern do they observe between the number of stocks in a country and strategy performance? Why does this relationship hold?

15. Chen, Kelly & Xiu (2022) find "two prominent frequencies" in news-based trading signals. What are they, and what does this imply for portfolio construction at monthly horizons?

## Coding Agents and the AI Tech Stack

16. Describe the stages of AI coding maturity (Stages 1--8). At which stage does the developer's role shift from "individual contributor" to "supervisor" or "director"?

17. What is the generative AI tech stack, and at which layers do most practitioners spend their time? Why?

## API and Practical Tools

18. In the Lopez-Lira & Tang (2023) study, why was the API necessary rather than the ChatGPT chat interface? What practical constraint does this address?

19. Explain the three message roles in the OpenAI chat completions API (system, user, assistant). How does the system message shape model behavior?

20. Approximately how many tokens does a typical word correspond to? If GPT-4's context window is 128K tokens, roughly how many words can fit in one context window?

## Markets Learning and Strategy Decay

21. Lopez-Lira & Tang (2023) report that the annualized Sharpe ratio of the GPT-4 long-short strategy declines from approximately 6.5 in 2021 to 2.0 in 2023. Give at least two explanations for this decline that are consistent with the Efficient Markets Hypothesis.

22. In the "Markets Learning" figure, the confidence intervals around the Sharpe ratio estimates narrow from 2021 to 2023. Why does this happen? Is it evidence that the strategy is becoming *less* volatile, or is there a simpler statistical explanation?

## Replication: Label Distributions and Portfolio Consequences

23. In the Homework 1 replication, GPT-3.5-Turbo classified 69.5% of headlines as UNKNOWN, compared to approximately 34% in the original paper's GPT-4 results. Explain why a high UNKNOWN rate reduces the long-short portfolio's Sharpe ratio, even though UNKNOWN signals are not traded.

24. The replication produced a YES:NO ratio of approximately 8:1 (27.4% YES vs. 3.2% NO). What does this imbalance imply for (a) the composition of the long vs. short legs on a typical day, and (b) the strategy's effective exposure?

25. In the FinBERT baseline from Homework 1, the model produces three labels (positive, neutral, negative). How are these mapped to trading signals? Why does FinBERT underperform GPT for return prediction, despite being fine-tuned on financial text?

## Replication: Overnight News Timing

26. The replication assigns headlines arriving after 4:00 PM ET to the *next* trading day and drops headlines arriving between 9:30 AM and 4:00 PM ET. Explain the rationale for each rule. What data would you need to include intraday headlines in the analysis?

27. RavenPack timestamps are stored in UTC. Describe the steps needed to correctly apply the overnight timing rule. What error would result from applying the 4:00 PM cutoff directly to UTC timestamps?
