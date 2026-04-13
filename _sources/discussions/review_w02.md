# Study Questions

Review questions for Week 2: Text Representation and Tokenization.

---

## Bag of Words

1. Define the Bag of Words (BoW) representation. Given a vocabulary and a document, how is the BoW vector constructed?

2. In the BoW example from the notes, the documents "The Fed raised rates today. This lowered bond prices." and "The Fed lowered rates today. This raised bond prices." produce *identical* BoW vectors. Explain why, and why this is a devastating limitation for financial applications.

3. What information does BoW discard, and why does this matter for tasks where the direction of causality is important (e.g., rate hikes vs. rate cuts)?

## TF-IDF

4. Write the TF-IDF formula. Explain each component: what does term frequency measure, what does inverse document frequency measure, and how do they interact?

5. Why does the IDF formula use a logarithm? What would happen if the raw ratio $N / \text{df}(w_i)$ were used without the log transformation? Give a concrete numerical example.

6. Compute the IDF for a word that appears in 1 out of 10,000 documents, a word that appears in half the documents, and a word that appears in every document. Explain what the values tell you about each word's discriminative power.

## Word2Vec

7. What problem does Word2Vec solve that Bag of Words cannot? How does representing words as dense vectors in $\mathbb{R}^d$ (where $d \ll V$) differ from one-hot encoding?

8. Describe the Skip-Gram model. Given a center word, what does the model predict, and how is the training objective formulated?

9. What is negative sampling, and why is it used instead of a full softmax over the vocabulary? Explain the role of the $3/4$ exponent in the noise distribution.

10. Explain the geometric meaning of Word2Vec embeddings using the classic analogy: $\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$. What does this tell us about the structure of the learned vector space?

## Byte Pair Encoding (BPE)

```{admonition} Worked Examples
:class: warning
Review and be able to reproduce from scratch all worked examples in the [Words to Tokens](words_to_tokens.md) notes, including the BPE training example (3 iterations on the toy corpus) and the encoding example (tokenizing "lowest" with the learned merge table).
```

11. Describe the BPE training algorithm. What is the input, what does each iteration do, and what is the output?

12. Given the toy corpus from the notes (low: 5, lower: 2, newest: 6, widest: 3), perform the first two iterations of BPE. Show the pair frequency counts and updated representations after each merge.

13. After three BPE merges produce the symbol `est</w>`, the algorithm has "discovered morphology." Explain what this means -- how did the algorithm learn the English superlative suffix without any linguistic rules?

14. Using the merge table $\mathcal{M} = [(e,s) \to es,\; (es,t) \to est,\; (est,\texttt{</w>}) \to est\texttt{</w>}]$, show step-by-step how the word "lowest\</w\>" is tokenized. What is the final token sequence?

15. What are the key properties of BPE? Specifically, explain: (a) why there are no out-of-vocabulary words, (b) how vocabulary size is controlled, and (c) why the algorithm is greedy and order-dependent.

## Byte-Level BPE

16. How does byte-level BPE (the GPT-2 variant) differ from character-level BPE? Why does starting with 256 raw byte values guarantee coverage of any language?

17. What is the complexity of BPE training and encoding? Express each in terms of the number of merge operations $M$ and the corpus/string size.

## Tiktoken

18. What is Tiktoken, and how does it relate to the BPE algorithm described in the notes? Name at least three encoding standards and which models they correspond to.

19. What are special tokens (e.g., `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`)? How are they used in the ChatML message format?

20. Why does tokenization matter for financial applications? Discuss at least two practical considerations (e.g., API costs, context window budgeting, multilingual efficiency).

## AI Copilots

21. Why did Cursor and Windsurf fork VS Code rather than build extensions? Name at least two capabilities that the VS Code extension API does not support.

22. What distinguishes a terminal-based agent harness (e.g., Claude Code, Codex CLI) from an IDE fork (e.g., Cursor)? What architectural advantage do terminal-based harnesses have?

## Embeddings and Return Prediction (Chen, Kelly & Xiu 2022)

23. In the Chen, Kelly & Xiu (2022) replication (Homework 2), students used mean pooling to convert BERT's token-level hidden states into a single headline-level embedding. Write the mean-pooling formula. What role does the attention mask play, and what goes wrong if it is omitted?

24. In the Homework 2 pipeline, binary labels are created from a 3-day compound return window centered on the article date: $r = \prod_{k=-1}^{1}(1+r_{t+k}) - 1$, with label = 1 if $r > 0$. What information is lost by converting continuous returns to binary labels? Why is this still useful for a classification-based approach?

25. Describe the rolling-window evaluation scheme used in Homework 2 (train on years $T{-}8$ to $T{-}3$, validate on $T{-}2$ to $T{-}1$, test on year $T$). Which of the following would introduce look-ahead bias: (a) fitting the StandardScaler on train + test data, (b) fitting TF-IDF vocabulary on the entire corpus, (c) selecting the regularization parameter $C$ on the validation set?

26. Chen, Kelly & Xiu (2022) show that an equal-weighted ensemble of multiple LLMs achieves a Sharpe ratio of 5.11, compared to 4.62 for the best individual model (ChatGPT). Given pairwise correlations between models ranging from 0.3 to 0.8, explain why the ensemble outperforms. How is this analogous to diversification in portfolio theory?

27. The "Negation Portfolios" table in Chen, Kelly & Xiu (2022) shows that word-based methods (SESTM, LMMD) suffer large declines in long-short Sharpe ratio on articles containing negation, while LLM-based methods do not. Explain why bag-of-words methods fail on negation and why simply removing negation words from a stop-word list does not fix the problem.

28. Chen, Kelly & Xiu (2022) show that the embedding-based approach works across 25+ countries and languages. Explain why LLM embeddings are "polyglot" — why do they work without translation? What is the relationship between the number of stocks in a country and the strategy's Sharpe ratio?
