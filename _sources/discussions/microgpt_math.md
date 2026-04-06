# The Mathematics of microGPT

```{admonition} Resources
:class: tip
- {download}`Download the full mathematical notes as PDF <assets/notes_microgpt_math.pdf>`
- [Blog post introducing microGPT](https://karpathy.github.io/2026/02/12/microgpt/) by Andrej Karpathy
- [Google Colab notebook](https://colab.research.google.com/drive/1vyN5zo6rqUp_dYNbT4Yrco66zuWCZKoN?usp=sharing)
```

*A High-Level Companion to Andrej Karpathy's 200-Line GPT Implementation*

Prepared for FINM 33200: Generative and Agentic AI for Finance, The University of Chicago.

Based on [`microgpt.py`](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) by Andrej Karpathy.

## Overview

In February 2026, Andrej Karpathy released `microGPT`---a single Python file of approximately 200 lines, with no external dependencies, that implements the complete training and inference loop for a GPT-class language model. The code trains a character-level transformer on a dataset of 32,000 human names and learns to generate new, plausible-sounding names.

`microGPT` is pedagogically extraordinary because it contains the *entire* algorithmic content of modern large language models: a tokenizer, a scalar-valued automatic differentiation engine, a GPT-2-style transformer architecture with multi-head causal attention, the Adam optimizer, a training loop, and an autoregressive inference loop. Everything beyond these 200 lines---GPU parallelism, tensor operations, BPE tokenization, distributed training---is, as Karpathy puts it, "just efficiency."

### Model Specifications

| Hyperparameter | Value | Variable |
|---|---|---|
| Vocabulary size | 27 | `vocab_size` |
| Embedding dimension | 16 | `n_embd` |
| Number of layers | 1 | `n_layer` |
| Number of attention heads | 4 | `n_head` |
| Head dimension | 4 | `head_dim` |
| Max sequence length | 16 | `block_size` |
| Total parameters | 4,192 | `len(params)` |

The vocabulary consists of the 26 lowercase English letters plus a special Beginning-of-Sequence (`BOS`) token, for a total of $V = 27$ tokens.


## Architecture

The model is a causal (autoregressive) transformer decoder following the GPT-2 architecture, with a few simplifications: RMSNorm instead of LayerNorm, ReLU instead of GeLU, and no bias terms. Despite these simplifications, the core data flow is identical to production LLMs.

At a high level, the forward pass works as follows:

1. **Tokenization**: Each name is converted to a sequence of character indices, wrapped with `BOS` delimiters.
2. **Embedding**: The token and its position are each looked up in learned embedding tables and summed.
3. **Transformer Block** (repeated $L$ times):
   - **Multi-Head Attention**: The current token queries all past tokens via scaled dot-product attention, allowing information to flow across positions.
   - **Feed-Forward Network (MLP)**: A two-layer network (expand to 4x width, apply ReLU, project back down) processes each position independently.
   - Both sub-layers use **residual connections** and **RMSNorm**.
4. **Output**: The final hidden state is projected to vocabulary size, producing logits that are converted to a probability distribution via softmax.

### Architecture Diagram

![microGPT Architecture](./assets/architecture_microgpt.png)

*Architecture of `microGPT`. Dashed lines indicate residual connections. The transformer block (between the braces) is repeated $L$ times ($L=1$ by default).*

### The KV Cache

During autoregressive processing, each position's key and value vectors are cached so that when the model processes position $t$, attention can access all positions $0, 1, \ldots, t$. This is what makes the model *causal*: each position can only attend to the current and past positions, never the future.


## Training and Inference

**Training** proceeds by feeding each name through the model one token at a time, computing the cross-entropy loss (negative log-probability of the true next token), backpropagating through the computation graph, and updating parameters with the Adam optimizer. Over 1,000 steps, the loss decreases from $\approx 3.37$ (random guessing at $-\log(1/27)$) to $\approx 2.37$.

**Inference** uses temperature sampling: logits are divided by a temperature $\tau$ before softmax. Lower temperature ($\tau < 1$) sharpens the distribution for more conservative output; higher temperature flattens it for more diversity. `microGPT` uses $\tau = 0.5$.


## Connecting to Production LLMs

`microGPT` and a model like GPT-4 or Claude share the same fundamental algorithm. Every difference between them is a matter of scale and engineering efficiency rather than algorithmic novelty:

| Component | microGPT | Production LLMs |
|---|---|---|
| Data | 32K names | Trillions of tokens of internet text |
| Tokenizer | Character-level (27 tokens) | BPE / SentencePiece (~100K tokens) |
| Autograd | Scalar `Value` in Python | Tensor ops on GPU (PyTorch/JAX) |
| Architecture | 4,192 params, 1 layer | $10^{11}$+ params, 100+ layers |
| Normalization | RMSNorm (no learnable params) | RMSNorm with learnable gain |
| Activation | ReLU | SwiGLU / GeLU |
| Positional | Learned absolute | RoPE (relative) |
| Attention | Full MHA | GQA (Grouped Query Attention) |
| Optimizer | Adam ($\beta_1{=}0.85$) | AdamW with warmup + cosine decay |
| Training | 1 doc/step, 1000 steps | Millions of tokens/step, months of compute |
| Post-training | None | SFT + RLHF/RLAIF |

None of these differences alter the core loop: encode tokens $\to$ embed $\to$ transform (attention + MLP) $\to$ predict next token $\to$ compute loss $\to$ backpropagate $\to$ update parameters. A conversation with ChatGPT is, from the model's perspective, just a document completion---exactly like `microGPT` completing a name.


## Summary of All Equations

For reference, we collect the key equations that define the forward pass of `microGPT`:

**Embedding:**

$$
\boldsymbol{x}^{(0)} = \operatorname{RMSNorm}\!\big(W^{\mathrm{te}}[t,:] + W^{\mathrm{pe}}[p,:]\big)
$$

**For each layer $\ell = 0, \ldots, L-1$:**

*Attention:*

$$
\begin{align*}
\boldsymbol{q} &= W_\ell^Q \cdot \operatorname{RMSNorm}(\boldsymbol{x}), \quad \boldsymbol{k} = W_\ell^K \cdot \operatorname{RMSNorm}(\boldsymbol{x}), \quad \boldsymbol{v} = W_\ell^V \cdot \operatorname{RMSNorm}(\boldsymbol{x}) \\
\alpha^{(h)}_s &= \frac{\boldsymbol{q}^{(h)} \cdot \boldsymbol{k}^{(h)}_s}{\sqrt{d_h}}, \quad \boldsymbol{w}^{(h)} = \operatorname{softmax}(\boldsymbol{\alpha}^{(h)}), \quad \boldsymbol{h}^{(h)} = \textstyle\sum_s w^{(h)}_s \boldsymbol{v}^{(h)}_s \\
\boldsymbol{x} &\leftarrow W_\ell^O \cdot [\boldsymbol{h}^{(0)} \| \cdots \| \boldsymbol{h}^{(H-1)}] + \boldsymbol{x}_{\mathrm{residual}}
\end{align*}
$$

*MLP:*

$$
\boldsymbol{x} \leftarrow W_\ell^{\mathrm{fc2}} \cdot \operatorname{ReLU}\!\big(W_\ell^{\mathrm{fc1}} \cdot \operatorname{RMSNorm}(\boldsymbol{x})\big) + \boldsymbol{x}_{\mathrm{residual}}
$$

**Output:**

$$
\boldsymbol{z} = W^{\mathrm{lm}} \boldsymbol{x}, \qquad \boldsymbol{p} = \operatorname{softmax}(\boldsymbol{z})
$$

**Loss:**

$$
\mathcal{L} = -\frac{1}{n} \sum_{t=0}^{n-1} \log p^{(t)}_{s_{t+1}}
$$


## Further Reading: nanochat

```{note}
The material in this section is provided for interested readers and is **not covered on the exam**.
```

The PDF version of these notes includes an appendix on `nanochat`, Karpathy's follow-up project that scales the same ideas from `microGPT` into a production-quality GPT model trained on real internet text. While `microGPT` is a 200-line, dependency-free teaching tool that trains on 32K names, `nanochat` is a full-featured codebase that trains a 124M-parameter transformer on the FineWeb dataset using modern techniques: Rotary Position Embeddings (RoPE), QK-Norm, grouped-query attention, SwiGLU activations, logit softcapping, and a combined Muon/AdamW optimizer. The [nanochat appendix](nanochat_math.md) walks through the mathematics of each of these components, mapping every equation to the corresponding line in the `nanochat` source code---the same approach used in this document for `microGPT`, but at production scale.


## Acknowledgments

This document is based entirely on Andrej Karpathy's `microGPT` project (MIT License), available at [https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95), and his accompanying blog post at [https://karpathy.github.io/2026/02/12/microgpt/](https://karpathy.github.io/2026/02/12/microgpt/). All mathematical formulations presented here are derived directly from the source code to ensure exact correspondence.
