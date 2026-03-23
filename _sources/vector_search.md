# Discussion 4: Text Representation & Embeddings

**Duration:** 3 hours
**Format:** Hands-on workshop

This session explores how text becomes numbers—the representations that underlie everything from topic models to LLM embeddings. We trace the progression from bag-of-words (Bybee et al. 2024) to contextual embeddings (Chen et al. 2022) and see how each representation captures increasingly rich information from financial text. Students build embedding pipelines using both BERT and OpenAI's `text-embedding-3-large`.

## Learning Objectives

By the end of this session, you will be able to:

- **Explain the progression** from bag-of-words to Word2Vec to contextual embeddings
- **Use the OpenAI Embeddings API** to generate dense vector representations of text
- **Use HuggingFace Transformers** to extract BERT embeddings
- **Compare text representation methods** and understand their trade-offs for financial applications
- **Connect embeddings to return prediction** following Chen, Kelly & Xiu (2022)

## Course Materials

```{toctree}
:maxdepth: 1
discussions/vector_embeddings_and_search.md
discussions/vector_embeddings_literature.md
```

## Additional Resources

### Papers
- [Chen, Kelly & Xiu (2022): Expected Returns and Large Language Models](https://ssrn.com/abstract=4416687)
- [Bybee, Kelly, Manela & Xiu (2024): Business News and Business Cycles](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13377)

### Documentation
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)

### Data
- [Structure of News](https://structureofnews.com/) — Pre-constructed topic attention time series from Bybee et al.

## Assessment

See [Homework 2: Embedding-Based Return Prediction](HW2.md) for the assignment associated with this session.
