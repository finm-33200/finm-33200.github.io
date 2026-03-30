# The Structure of Economic News

*These notes draw on two papers that use topic modeling on Wall Street Journal articles to understand how business news is structured and how it relates to macroeconomic outcomes and asset prices:*

> Bybee, L., Kelly, B., Manela, A., & Xiu, D. (2024). [Business News and Business Cycles.](../references/bybee_et_al_2024_business_news_and_business_cycles.pdf) *Journal of Finance, 79*(5), 3105--3147. <https://doi.org/10.1111/jofi.13377>

> Bybee, L., Kelly, B., & Su, Y. (2023). [Narrative Asset Pricing: Interpretable Systematic Risk Factors from News Text.](../references/bybee_kelly_su_2023_narrative_asset_pricing.pdf) *The Review of Financial Studies, 36*(12), 4759--4787. <https://doi.org/10.1093/rfs/hhad042>

*The core idea is simple: if we can measure what the news is talking about at any point in time, we can use that to understand --- and even predict --- economic activity and asset prices. Both papers apply **Latent Dirichlet Allocation (LDA)**, a topic model, to roughly 800,000 Wall Street Journal articles from 1984--2017 to extract 180 interpretable topics. They then show that fluctuations in news attention to these topics track the business cycle and carry information for asset pricing.*

*Figure images below are reproduced from the authors' paper and website. Sources: Bybee, Kelly, Manela, & Xiu, "Business News and Business Cycles," 2024; structureofnews.com. For illustrative purposes only.*

---

## The Data and the Website

The authors built an interactive website at [structureofnews.com](https://structureofnews.com/) where you can explore the full topic taxonomy, view time series of individual topics, and download the data.

![The Structure of Economic News website](assets/structure_of_news/structure_of_news_website.png)

*Source: [structureofnews.com](https://structureofnews.com/). For illustrative purposes only.*

The underlying data is the full text of roughly 800,000 Wall Street Journal articles spanning 1984--2017. Using LDA, the authors estimate a topic model with **180 topics**, each defined by a distribution over words. At each point in time, they measure the proportion of news attention allocated to each topic --- how much of the WSJ is talking about recessions, the Fed, technology, oil markets, and so on.

---

## News Taxonomy

The 180 topics are not random --- they cluster into an intuitive hierarchy. Using hierarchical agglomerative clustering, the authors organize topics into broad **metatopics** that correspond to recognizable areas of business news.

![News taxonomy dendrogram](assets/structure_of_news/structure_of_news_taxonomy.png)

*Source: [structureofnews.com](https://structureofnews.com/). For illustrative purposes only.*

For example, "Financial Intermediaries" splits into Banks, Asset Managers, Buyouts & Bankruptcy, and Financial Markets --- each of which further splits into specific topics like Mortgages, Credit Ratings, IPOs, Bond Yields, and so on. The taxonomy is entirely data-driven, yet it reads like a table of contents for a financial newspaper.

---

## Investor Beliefs Over Time

At the metatopic level, news attention is remarkably stable over time. Banks, markets, corporate earnings, technology, international affairs --- these broad categories maintain roughly constant shares of WSJ coverage across decades.

![Investor beliefs in the WSJ (metatopics)](assets/structure_of_news/investor_belief_metatopics.png)

*Source: Bybee, Kelly, Manela, & Xiu, "Business News and Business Cycles," 2024. For illustrative purposes only.*

But the interesting action happens within these broad categories. The fluctuations in attention to specific topics within each metatopic --- how much we're talking about recession versus economic growth, or oil versus technology --- are where the economic signal lives.

---

## Topic Detail: Recession

We can drill into individual topics to see their time series behavior. The "Recession" topic is a good example --- its attention share spikes during and around NBER recessions, and the associated word cloud shows exactly what language drives this topic.

![Recession topic detail from structureofnews.com](assets/structure_of_news/structure_of_news_recession_attention.png)

*Source: [structureofnews.com](https://structureofnews.com/). For illustrative purposes only.*

Words like "recession," "unemployment," "downturn," "slowed," and "weak economy" define this topic. When the WSJ devotes more column inches to these words, the economy tends to be in trouble. This is intuitive, but the power of the approach is that it gives us a **quantitative, real-time measure** of how much the news is focused on recession --- something we can track daily and feed into statistical models.

---

## Reconstructing Macroeconomic Time Series

The payoff: news topic attention can **reconstruct and predict macroeconomic time series**. Using lasso regression to select the five most relevant topics, the authors show that news attention alone explains substantial variation in key macro outcomes.

![Table I: Reconstructing macroeconomic time series (production and employment)](assets/structure_of_news/table1a.png)

*Source: Bybee, Kelly, Manela, & Xiu, "Business News and Business Cycles," 2024. For illustrative purposes only.*

For **industrial production growth**, the "Recession" topic enters with a coefficient of --0.38 ($p = 0.00$), and the five-topic model achieves an $R^2$ of 0.21. For **employment growth**, the "Recession" topic is even more dominant (coefficient --0.61, $p = 0.00$), with an $R^2$ of 0.59. The fitted values track the actual series remarkably well --- look at how the red predicted line follows the blue actual line through the 2008 financial crisis.

![Table I continued: Market returns and volatility](assets/structure_of_news/table1b.png)

*Source: Bybee, Kelly, Manela, & Xiu, "Business News and Business Cycles," 2024. For illustrative purposes only.*

The same approach works for financial variables. News topics explain 25% of the variation in **market returns** and 63% of **market volatility**. The "Recession" topic shows up as the most important predictor in every specification --- negative for production, employment, and returns; positive for volatility. This is consistent with the ICAPM interpretation in the companion paper (Bybee, Kelly, & Su, 2023): news narratives proxy for updates to investor beliefs about the future investment opportunity set.

---

## From Topics to Tokens

Both of these papers rely on **Latent Dirichlet Allocation** --- a bag-of-words topic model. LDA represents each article as a mixture of topics, and each topic as a distribution over words. Critically, like all bag-of-words methods, LDA **discards word order**. It knows that an article contains words like "recession," "unemployment," and "growth," but it doesn't know the context in which those words appear.

This is a natural segue into our next topic: how do we represent text numerically? We'll start with the bag-of-words model that underlies LDA, see its strengths and limitations, and then trace the evolution through Word2Vec embeddings to the modern subword tokenization (BPE) used by GPT and other large language models.
