# The Mathematics of Karpathy's nanochat

```{admonition} PDF Version
:class: tip
{download}`Download these notes as PDF <assets/notes_nanochat_math.pdf>`
```

> *These notes provide a self-contained mathematical treatment of Karpathy's `nanochat` GPT model. Every equation is derived from first principles and then mapped to the exact corresponding line in `nanochat/gpt.py` and `nanochat/optim.py`. We cover the full forward pass (token embeddings, RMSNorm, Rotary Position Embeddings, QK-Norm, Grouped-Query Attention with sliding windows, value embeddings with gated residuals, ReluSquared MLP, logit softcapping, and cross-entropy loss), the weight initialization scheme, the combined Muon/AdamW optimizer (including the Polar Express orthogonalization and NorMuon variance reduction), and the scaling-law-driven hyperparameter schedule. The codebase reference is `nanochat` commit circa February 2026 (`master` branch).*

---

## Architecture Diagram

The figure below presents the complete forward pass of the `nanochat` model, with annotations referencing the equations and sections derived in these notes. The diagram should be read top-to-bottom: tokens enter at the top, flow through the embedding and $L$ repeated Transformer blocks, and exit through the logit head and loss at the bottom. Key architectural departures from the original GPT-2 are summarized in the legend.

![Nanochat Architecture](assets/nanochat_architecture.pdf)

*The `nanochat` architecture. Each component is labeled with the equation number and section where its mathematics is derived. The dashed blue lines are residual skip connections; the dashed orange line is the $x_0$ skip connection that feeds the initial normalized embedding back into every layer via learnable scalars. The repeating Transformer block (dashed blue border) is instantiated $L$ times, where $L$ is the single "depth" hyperparameter that controls the entire model.*

---

## From Text to Training Signal

Before diving into the model architecture, we establish how raw text becomes the mathematical objects that the model trains on. This section answers the question: *given a collection of newspaper articles, what exactly does the model see, and what is it trying to predict?*

### Terminology

We begin with precise definitions. These terms are used throughout the notes and in the codebase.

| **Term** | **Definition** |
|---|---|
| **Token** | The atomic unit of text after tokenization. Not a word, not a character---a BPE (Byte Pair Encoding) subword piece. The nanochat vocabulary has $V = 32{,}768$ tokens. Common words like "the" are single tokens; rare words are split into pieces (e.g., "quantitative" $\to$ ["quant", "itative"]). Each token is represented by an integer ID $\in \{0, \dots, V-1\}$. |
| **Document** | A single coherent text: one newswire article, one SEC filing, one earnings call transcript. Has a beginning and an end. This is the fundamental unit of your raw data. |
| **Corpus** | The complete collection of documents used for training. For example: all Reuters financial newswires from January 1996 through December 2006. |
| **Sequence** | A fixed-length chunk of $T$ tokens ($T = 2048$ by default) that the model processes in one forward pass. This is the unit the *model* sees. A single sequence may contain parts of multiple documents packed together (separated by special BOS tokens). |
| **Context window** | Synonym for sequence length $T$. The maximum number of tokens the model can attend to at once. |
| **Batch** | Multiple sequences processed in parallel for computational efficiency. The total batch size in nanochat is measured in tokens (e.g., $B_{\text{total}} = 524{,}288$ tokens $= 256$ sequences $\times$ $2048$ tokens each). |
| **Epoch** | One complete pass through the entire corpus. Nanochat's scaling laws target roughly 1 epoch for compute-optimal training---each document is seen approximately once. |
| **Shard** | Nanochat stores the corpus as Parquet files (shards) for efficient I/O. An implementation detail, not a conceptual unit. |
| **BOS token** | "Beginning of Sequence" (`<|bos|>`). A special token prepended to every document to signal its start. When multiple documents are packed into one sequence, BOS tokens act as delimiters between them. |

The conceptual hierarchy is:

$$
\underbrace{\text{Corpus}}_{\text{all your data}} \;\longrightarrow\; \underbrace{\text{Documents}}_{\text{individual articles}} \;\longrightarrow\; \underbrace{\text{Tokens}}_{\text{BPE subwords}} \;\xrightarrow{\text{pack}}\; \underbrace{\text{Sequences}}_{\text{model's input}} \;\xrightarrow{\text{group}}\; \underbrace{\text{Batches}}_{\text{parallel processing}}.
$$

### A Single Training Example

The fundamental training signal for a language model is **next-token prediction**. Consider a short financial newswire:

> Fed cuts rates by 50bp to 4.75%

After tokenization with nanochat's BPE tokenizer (vocabulary size $V = 32{,}768$), this becomes a sequence of token IDs. For illustration, suppose:

| **Text** | `<\|bos\|>` | Fed | cuts | rates | by | 50 | bp | to | 4.75% |
|---|---|---|---|---|---|---|---|---|---|
| **Token ID** | 0 | 8412 | 1937 | 5281 | 482 | 2150 | 19743 | 614 | 31006 |

From this single sequence of 9 tokens, nanochat constructs the training pair by shifting:

| | pos 1 | pos 2 | pos 3 | pos 4 | pos 5 | pos 6 | pos 7 | pos 8 |
|---|---|---|---|---|---|---|---|---|
| **Input $\boldsymbol{x}$** | `<\|bos\|>` | Fed | cuts | rates | by | 50 | bp | to |
| **Target $\boldsymbol{y}$** | Fed | cuts | rates | by | 50 | bp | to | 4.75% |

````{admonition} Worked Example: Every position is a training example
:class: tip
A single sequence of $T+1$ tokens produces $T$ simultaneous training examples. At each position $i$, the model takes tokens $(t_1, \dots, t_i)$ as input and must predict $t_{i+1}$.

Crucially, this happens *in parallel*---the causal attention mask ensures that position $i$ cannot see positions $i+1, i+2, \dots$, so all $T$ predictions are independent and the loss (cross-entropy) is averaged over all positions:

$$
\mathcal{L} = -\frac{1}{T} \sum_{i=1}^{T} \log P_\theta(t_{i+1} \mid t_1, \dots, t_i).
$$

This is why language model training is so data-efficient: a single 2048-token sequence produces 2048 gradient signals, one at each position. The model simultaneously learns that "`<|bos|>` Fed" should predict "cuts," that "Fed cuts" should predict "rates," that "Fed cuts rates by 50" should predict "bp," and so on.
````

### Building a Training Corpus

In practice, a corpus contains millions of documents of varying lengths. For a financial newswire corpus, these might range from 50-token breaking-news flashes to 2000-token analysis pieces. The challenge is: how do we turn this heterogeneous collection into fixed-length sequences of $T = 2048$ tokens?

#### Step 1: Tokenize and Prepend BOS

Each document is tokenized independently, and a BOS token (`<|bos|>`, ID 0) is prepended:

$$\begin{aligned}
\text{Article 1 (Jan 2007):}& \quad [\texttt{BOS},\; \underbrace{t_1^{(1)},\; t_2^{(1)},\; \dots,\; t_{320}^{(1)}}_{320\text{ tokens}}] \\
\text{Article 2 (Mar 2007):}& \quad [\texttt{BOS},\; \underbrace{t_1^{(2)},\; t_2^{(2)},\; \dots,\; t_{1850}^{(2)}}_{1850\text{ tokens}}] \\
\text{Article 3 (Mar 2007):}& \quad [\texttt{BOS},\; \underbrace{t_1^{(3)},\; t_2^{(3)},\; \dots,\; t_{95}^{(3)}}_{95\text{ tokens}}]
\end{aligned}$$

```
# dataloader.py, line 107
token_lists = tokenizer.encode(doc_batch, prepend=bos_token, ...)
```

#### Step 2: Best-Fit Packing into Sequences

Nanochat packs documents into fixed-length rows of $T + 1 = 2049$ tokens (the extra token is needed so that the last position has a target). The algorithm, from `dataloader.py`, is:

1. Maintain a buffer of tokenized documents.
2. For each row: repeatedly pick the **largest document** from the buffer that fits entirely in the remaining space. Place it in the row.
3. When no document fits, **crop** the shortest document in the buffer to fill the remaining space exactly.
4. Result: 100% utilization (no padding tokens), at the cost of ~35% of tokens being discarded due to cropping.

````{admonition} Worked Example: Packing three articles into one sequence
:class: tip
Suppose $T + 1 = 2049$ and we have the three articles above (321, 1851, and 96 tokens including BOS).

**Row construction:**
1. Place Article 2 (1851 tokens). Remaining: $2049 - 1851 = 198$ slots.
2. Scan buffer: Article 1 (321 tokens) doesn't fit. Article 3 (96 tokens) fits. Place it. Remaining: $198 - 96 = 102$ slots.
3. No complete document fits. Crop Article 1 to 102 tokens: take $[\texttt{BOS}, t_1^{(1)}, \dots, t_{101}^{(1)}]$.

The resulting row of 2049 tokens:

$$
[\underbrace{\texttt{BOS},\; t_1^{(2)}, \dots, t_{1850}^{(2)}}_{\text{Article 2 (full)}},\;
\underbrace{\texttt{BOS},\; t_1^{(3)}, \dots, t_{95}^{(3)}}_{\text{Article 3 (full)}},\;
\underbrace{\texttt{BOS},\; t_1^{(1)}, \dots, t_{101}^{(1)}}_{\text{Article 1 (cropped)}}]
$$

The first 2048 tokens become the input $\boldsymbol{x}$; the last 2048 tokens (shifted by one) become the target $\boldsymbol{y}$. Every position in this row is a valid training example. The BOS tokens tell the model "a new document starts here"---the model learns to reset its predictions at these boundaries.

Note that Article 1 lost $321 - 102 = 219$ tokens to cropping. Those tokens are simply discarded. Nanochat accepts this waste (~35%) in exchange for a crucial property: **every row starts with BOS**, so every token can attend back to the start of its document.
````

#### Step 3: The Shifted Input/Target Pair

From each row of $T + 1 = 2049$ tokens, the dataloader produces:

$$\begin{aligned}
\boldsymbol{x} &= (\text{row}[0],\; \text{row}[1],\; \dots,\; \text{row}[T-1]) \in \{0,\dots,V-1\}^T, \\
\boldsymbol{y} &= (\text{row}[1],\; \text{row}[2],\; \dots,\; \text{row}[T]) \in \{0,\dots,V-1\}^T.
\end{aligned}$$

The target at position $i$ is simply the token at position $i+1$. This one-position shift is the entire training signal.

```
# dataloader.py, lines 154--155
cpu_inputs.copy_(row_buffer[:, :-1])
cpu_targets.copy_(row_buffer[:, 1:])
```

### What the Model Learns: Embeddings

A trained nanochat model is useful not only for generating text but also as a source of **dense vector representations** (embeddings) of tokens and documents. These embeddings capture semantic relationships learned during pretraining, and can serve as features for downstream tasks such as time-series prediction, document retrieval, or similarity measurement.

There are three distinct types of embeddings available from a nanochat model, each living at a different point in the forward pass:

#### Token Embeddings (Static)

The embedding matrix $\boldsymbol{W}_E \in \mathbb{R}^{V \times d_{\text{model}}}$ assigns a fixed vector to each token in the vocabulary, regardless of context:

$$
\boldsymbol{e}_{\text{token}}(t) = \boldsymbol{W}_E[t] \in \mathbb{R}^{d_{\text{model}}}.
$$

These are analogous to Word2Vec embeddings. The vector for "rates" is the same whether it appears in "interest rates" or "mortality rates." Extracting them requires no forward pass---just index into $\boldsymbol{W}_E$.

#### Contextual Embeddings (Rich, Layer-Dependent)

The hidden state $\boldsymbol{x}^{(\ell)}_i \in \mathbb{R}^{d_{\text{model}}}$ at layer $\ell$, position $i$, is a context-dependent representation. The vector for "rates" at position $i$ encodes everything the model has inferred from the preceding tokens $(t_1, \dots, t_i)$ through $\ell$ layers of attention and MLP processing.

To extract contextual embeddings from nanochat, run a standard forward pass and intercept the hidden states:

$$
\boldsymbol{x}^{(\ell)}_i = \text{output of layer } \ell \text{ at position } i, \qquad \ell \in \{0, 1, \dots, L\}.
$$

The most common choice for downstream use is the **final-layer representation** $\boldsymbol{x}^{(L)}_i$ (after the last Transformer block but before the LM head), since it integrates information from all layers.

````{admonition} Worked Example: Extracting a document embedding from nanochat
:class: tip
Suppose you want a single vector to represent the article "Fed cuts rates by 50bp to 4.75%."

**Step 1.** Tokenize: $\boldsymbol{t} = [\texttt{BOS}, 8412, 1937, 5281, 482, 2150, 19743, 614, 31006]$ (9 tokens).

**Step 2.** Forward pass through all $L$ layers. This produces $\boldsymbol{x}^{(L)} \in \mathbb{R}^{9 \times d_{\text{model}}}$---one $d_{\text{model}}$-dimensional vector per token position.

**Step 3.** Pool across positions to get a single document vector. Common strategies:

| **Strategy** | **Description** |
|---|---|
| Last-token | $\boldsymbol{e}_{\text{doc}} = \boldsymbol{x}^{(L)}_9$. Use the representation at the final position, which has attended to the entire document via causal attention. Simple and effective. |
| Mean pooling | $\boldsymbol{e}_{\text{doc}} = \frac{1}{T}\sum_{i=1}^{T} \boldsymbol{x}^{(L)}_i$. Average across all positions. More robust to sequence length variation. Often excludes the BOS position. |
| BOS-token | $\boldsymbol{e}_{\text{doc}} = \boldsymbol{x}^{(L)}_1$. Use only the BOS position. In some architectures this is trained to be a summary representation, but nanochat does not specifically train it this way. |

For financial newswires, last-token pooling is a natural choice: by position 9, the model has "read" the entire headline and its final hidden state summarizes the content through the lens of "what would come next."

*Nanochat-specific note:* Because of the untied embeddings ($\boldsymbol{W}_E \neq \boldsymbol{W}_U$) and the $x_0$ residual skip connection, the final-layer representation $\boldsymbol{x}^{(L)}_i$ contains a direct linear pathway back to the initial token embedding $\boldsymbol{x}^{(0)}_i$, weighted by the learned scalars $\lambda_{x_0}^{(\ell)}$. This is architecturally unusual---it means the contextual embedding retains a strong "fingerprint" of the raw token identity even after $L$ layers of processing.
````

#### Unembedding Vectors

The LM head matrix $\boldsymbol{W}_U \in \mathbb{R}^{V \times d_{\text{model}}}$ also provides useful vectors. Each row $\boldsymbol{W}_U[v]$ can be interpreted as the "direction in embedding space that represents token $v$ as a prediction target." The logit for predicting token $v$ at position $i$ is the dot product $\boldsymbol{x}^{(L)}_i \cdot \boldsymbol{W}_U[v]$, so tokens whose unembedding vectors are close to a contextual embedding are the tokens the model considers likely continuations.

Since nanochat uses untied embeddings, $\boldsymbol{W}_E[v]$ and $\boldsymbol{W}_U[v]$ are different vectors for the same token $v$---one represents "$v$ as input," the other represents "$v$ as prediction."

### Inference: Generating Text

At inference time, the model generates text one token at a time in an **autoregressive loop**:

1. Start with a prompt: $\boldsymbol{t} = (t_1, \dots, t_k)$ (e.g., the tokenized string `<|bos|> The Federal Reserve`).
2. Run a full forward pass to get logits $\boldsymbol{z} \in \mathbb{R}^{V}$ at the *last* position $k$.
3. Sample the next token $t_{k+1}$ from the distribution over the vocabulary.
4. Append $t_{k+1}$ to the sequence: $\boldsymbol{t} \leftarrow (t_1, \dots, t_k, t_{k+1})$.
5. Repeat from step 2 until a stopping condition is met.

#### Sampling Controls

The logits $\boldsymbol{z}$ are converted to a probability distribution via two controls:

**Temperature** $\tau > 0$ scales the logits before softmax:

$$
P(v) = \frac{\exp(z_v / \tau)}{\sum_{v'} \exp(z_{v'} / \tau)}.
$$

Setting $\tau < 1$ sharpens the distribution (more deterministic); $\tau > 1$ flattens it (more random); $\tau \to 0$ gives greedy decoding (always pick the most likely token).

**Top-$k$ filtering** zeroes out all but the $k$ highest-probability tokens before sampling:

$$
z_v \leftarrow \begin{cases} z_v & \text{if } v \text{ is among the top-}k, \\ -\infty & \text{otherwise.} \end{cases}
$$

```python
# gpt.py, lines 451--462
logits = self.forward(ids)
logits = logits[:, -1, :]  # only the last position
if top_k is not None:
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float('Inf')
if temperature > 0:
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    next_ids = torch.multinomial(probs, num_samples=1)
else:
    next_ids = torch.argmax(logits, dim=-1)
```

````{admonition} Worked Example: Generating one token
:class: tip
Given the prompt "`<|bos|> The Federal Reserve`", suppose the model produces logits (after softcapping) for the top-5 most likely next tokens:

| **Token** | **Logit $z_v$** | $P(v)$ at $\tau = 0.8$ |
|---|---|---|
| announced | $8.2$ | $0.52$ |
| raised | $6.9$ | $0.18$ |
| cut | $6.5$ | $0.13$ |
| said | $5.8$ | $0.07$ |
| is | $5.4$ | $0.05$ |

With $\tau = 0.8$, the distribution is sharpened: "announced" gets 52% probability. The model samples from this distribution (say it draws "announced"), appends it to the sequence, and repeats. With $\tau = 0$ (greedy), it would deterministically pick "announced" every time.

*Connection to training:* The model learned these probabilities from the training corpus. If in the newswire data, "The Federal Reserve announced" appeared frequently before rate decisions, the model assigns high probability to "announced" in this context. This is the direct payoff of the next-token prediction objective.
````

### Summary: The Data Pipeline at a Glance

| **Stage** | **What happens** | **Nanochat implementation** |
|---|---|---|
| Raw corpus | Collection of documents | Parquet files with `text` column |
| Tokenization | Text $\to$ integer IDs, BOS prepended | `tokenizer.encode(text, prepend=bos)` |
| Packing | Docs packed into $T{+}1$ rows | Best-fit algorithm, 100% utilization |
| Shift | Row $\to$ (input, target) pair | `inputs = row[:, :-1]`, `targets = row[:, 1:]` |
| Batching | $B$ rows per batch | Total batch size in tokens |
| Training | Predict every next token | Cross-entropy loss at all $T$ positions |
| Embeddings | Hidden states as features | $\boldsymbol{x}^{(L)}_i \in \mathbb{R}^{d_{\text{model}}}$, pool across positions |
| Inference | Autoregressive generation | Sample from $P(v) \propto \exp(z_v / \tau)$ |

With this pipeline in mind, we now turn to the mathematical details of each component inside the model.

---

## Architecture Overview

The `nanochat` model is a decoder-only GPT Transformer parameterized by a single integer, the *depth* $L$ (number of layers). All other hyperparameters are derived:

$$\begin{aligned}
d_{\text{base}} &= L \cdot r, \qquad r = 64 \text{ (aspect ratio, default)}, \\
d_{\text{model}} &= \left\lceil \frac{d_{\text{base}}}{d_h} \right\rceil \cdot d_h, \qquad d_h = 128 \text{ (head dim, default)}, \\
H &= \frac{d_{\text{model}}}{d_h} \quad \text{(number of query heads = number of KV heads)},
\end{aligned}$$

where the ceiling ensures $d_{\text{model}}$ is divisible by $d_h$.

```
# base_train.py, lines 133--138
base_dim = depth * aspect_ratio
model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
num_heads = model_dim // head_dim
```

```{admonition} Remark
:class: note
For the "GPT-2 speedrun" ($L=24$), this yields $d_{\text{base}} = 1536$, $d_{\text{model}} = 1536$, $H = 12$. In nanochat, $H_{\text{kv}} = H$ (multi-head attention, not grouped-query at training time), though the architecture supports $H_{\text{kv}} < H$ for inference.
```

The config dataclass (`GPTConfig`) stores: sequence length $T = 2048$, vocabulary size $V = 32{,}768$, and a sliding window pattern string (default `"SSSL"`).

### High-Level Forward Pass

Given a token sequence $\boldsymbol{t} = (t_1, \dots, t_T) \in \{0,\dots,V-1\}^T$, the model computes:

1. **Token embedding**: $\boldsymbol{x}^{(0)} = \boldsymbol{W}_E[\boldsymbol{t}] \in \mathbb{R}^{T \times d_{\text{model}}}$, followed by RMSNorm.
2. **Save initial embedding**: $\boldsymbol{x}_0 \leftarrow \boldsymbol{x}^{(0)}$ (for the "x0 residual" connection).
3. **For each layer $\ell = 1, \dots, L$**:
   1. Rescale: $\boldsymbol{x}^{(\ell-1)} \leftarrow \lambda_{\text{resid}}^{(\ell)} \cdot \boldsymbol{x}^{(\ell-1)} + \lambda_{x_0}^{(\ell)} \cdot \boldsymbol{x}_0$.
   2. Self-attention with pre-norm: $\boldsymbol{x}^{(\ell-\frac{1}{2})} = \boldsymbol{x}^{(\ell-1)} + \text{Attn}_\ell\bigl(\operatorname{RMSNorm}(\boldsymbol{x}^{(\ell-1)})\bigr)$.
   3. MLP with pre-norm: $\boldsymbol{x}^{(\ell)} = \boldsymbol{x}^{(\ell-\frac{1}{2})} + \text{MLP}_\ell\bigl(\operatorname{RMSNorm}(\boldsymbol{x}^{(\ell-\frac{1}{2})})\bigr)$.
4. **Final norm**: $\boldsymbol{x}^{(L)} \leftarrow \operatorname{RMSNorm}(\boldsymbol{x}^{(L)})$.
5. **Logit head**: $\boldsymbol{z} = \boldsymbol{x}^{(L)} \boldsymbol{W}_U^\top \in \mathbb{R}^{T \times V}$, with softcap.
6. **Loss**: cross-entropy against shifted targets.

```python
# gpt.py, lines 410--431 (forward method)
x = self.transformer.wte(idx)
x = norm(x)
x0 = x
for i, block in enumerate(self.transformer.h):
    x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
    ve = self.value_embeds[str(i)](idx) ...
    x = block(x, ve, cos_sin, ...)
x = norm(x)
logits = self.lm_head(x)
```

---

## RMS Normalization

```{admonition} Definition: RMSNorm (no learnable parameters)
:class: note
For $\boldsymbol{x} \in \mathbb{R}^d$,

$$
\operatorname{RMSNorm}(\boldsymbol{x}) = \frac{\boldsymbol{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}}\,.
$$
```

Note the absence of learnable gain/bias parameters and the absence of an $\epsilon$ term in the denominator---both deliberate simplifications in nanochat.

```
# gpt.py, line 43
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```

```{admonition} Remark
:class: note
PyTorch's `F.rms_norm` internally adds a small $\epsilon = 10^{-6}$ for numerical stability, but nanochat does not expose or configure this. The computation runs in `bfloat16` (the comment in the code notes this "seems ok").
```

````{admonition} Worked Example: Tracing a token through RMSNorm
:class: tip
Suppose the embedding lookup for a single token produces $\boldsymbol{x} = (1.2,\; -0.5,\; 0.8,\; -1.0)^\top$ (a $d=4$ toy example).

**Step 1.** Mean square: $\frac{1}{4}(1.44 + 0.25 + 0.64 + 1.00) = \frac{3.33}{4} = 0.8325$.

**Step 2.** RMS $= \sqrt{0.8325} = 0.9124$.

**Step 3.** Normalize: $\operatorname{RMSNorm}(\boldsymbol{x}) = \boldsymbol{x} / 0.9124 = (1.315,\; {-0.548},\; 0.877,\; {-1.096})^\top$.

*Key observation:* The output has unit RMS (verify: mean of squares $= 1.0$), but it is *not* zero-mean. This is the distinction from LayerNorm, which would also center the vector. In nanochat, there is no learnable gain $\gamma$ or bias $\beta$---the raw normalized vector is passed forward.
````

---

## Rotary Position Embeddings (RoPE)

Nanochat uses RoPE *instead of* learned or sinusoidal positional embeddings. There is no additive position embedding at all.

### Frequency Construction

For head dimension $d_h$ and base frequency $\theta_{\text{base}} = 100{,}000$, define the inverse frequency vector:

$$
\omega_j = \frac{1}{\theta_{\text{base}}^{\,2j/d_h}}, \qquad j = 0, 1, \dots, \tfrac{d_h}{2} - 1.
$$

```
# gpt.py, lines 259--260
channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
inv_freq = 1.0 / (base ** (channel_range / head_dim))
```

For each position $t = 0, 1, \dots, T-1$, form the angle matrix:

$$
\phi_{t,j} = t \cdot \omega_j\,.
$$

Then precompute $\cos \phi_{t,j}$ and $\sin \phi_{t,j}$ for all $(t,j)$.

### Rotation Application

Given a query or key vector $\boldsymbol{u} \in \mathbb{R}^{d_h}$ at position $t$, split it into two halves: $\boldsymbol{u}_1 = \boldsymbol{u}_{1:d_h/2}$ and $\boldsymbol{u}_2 = \boldsymbol{u}_{d_h/2+1:d_h}$. The rotary embedding applies:

$$
\operatorname{RoPE}(\boldsymbol{u}, t) = \begin{pmatrix} \boldsymbol{u}_1 \cos\boldsymbol{\phi}_t + \boldsymbol{u}_2 \sin\boldsymbol{\phi}_t \\ -\boldsymbol{u}_1 \sin\boldsymbol{\phi}_t + \boldsymbol{u}_2 \cos\boldsymbol{\phi}_t \end{pmatrix},
$$

where all operations are element-wise and $\boldsymbol{\phi}_t = (\phi_{t,0}, \dots, \phi_{t,d_h/2-1})^\top$.

```python
# gpt.py, lines 57--63
def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)
```

```{admonition} Proposition: Inner product depends only on relative position
:class: note
For two vectors $\boldsymbol{q}$ at position $m$ and $\boldsymbol{k}$ at position $n$:

$$
\langle \operatorname{RoPE}(\boldsymbol{q}, m),\, \operatorname{RoPE}(\boldsymbol{k}, n) \rangle = \sum_{j=0}^{d_h/2-1} \bigl[(q_{1,j}k_{1,j} + q_{2,j}k_{2,j})\cos(\phi_{m-n,j}) + (q_{2,j}k_{1,j} - q_{1,j}k_{2,j})\sin(\phi_{m-n,j})\bigr],
$$

which depends on $m$ and $n$ only through $(m-n)$, giving the model relative positional awareness without any learnable positional parameters.
```

````{admonition} Worked Example: RoPE rotation for $d_h = 4$, $\theta_{\text{base}} = 100{,}000$
:class: tip
With head dimension $d_h = 4$, we have $d_h/2 = 2$ frequency pairs.

**Frequencies:**
$\omega_0 = 1/100000^{0/4} = 1.0$, $\quad \omega_1 = 1/100000^{2/4} = 1/\sqrt{100000} \approx 0.00316$.

**At position $t = 5$:** Angles are $\phi_{5,0} = 5.0$ rad, $\phi_{5,1} = 0.0158$ rad.

So $\cos\boldsymbol{\phi}_5 = (0.2837,\; 0.9999)$ and $\sin\boldsymbol{\phi}_5 = (-0.9589,\; 0.0158)$.

**Rotate** a query $\boldsymbol{q} = (0.3,\; -0.7,\; 0.5,\; 0.1)^\top$. Split: $\boldsymbol{q}_1 = (0.3, -0.7)$, $\boldsymbol{q}_2 = (0.5, 0.1)$.

$$\begin{aligned}
y_1 &= \boldsymbol{q}_1 \odot \cos\boldsymbol{\phi}_5 + \boldsymbol{q}_2 \odot \sin\boldsymbol{\phi}_5 = (0.3 \cdot 0.284 + 0.5 \cdot (-0.959),\;\; -0.7 \cdot 1.000 + 0.1 \cdot 0.016) \\
    &= (-0.394,\; -0.698), \\
y_2 &= \boldsymbol{q}_1 \odot (-\sin\boldsymbol{\phi}_5) + \boldsymbol{q}_2 \odot \cos\boldsymbol{\phi}_5 = (0.3 \cdot 0.959 + 0.5 \cdot 0.284,\;\; -0.7 \cdot (-0.016) + 0.1 \cdot 1.000) \\
    &= (0.430,\; 0.111).
\end{aligned}$$

Result: $\operatorname{RoPE}(\boldsymbol{q}, 5) = (-0.394,\; -0.698,\; 0.430,\; 0.111)^\top$.

*Key observation:* The first pair of dimensions (indices 0,2) rotated substantially (5 radians $\approx 286°$) because $\omega_0$ is large. The second pair (indices 1,3) barely moved (0.016 rad $\approx 0.9°$) because $\omega_1$ is tiny. This is the "multi-resolution" nature of RoPE: low-frequency dimensions encode long-range position, high-frequency dimensions encode local position.
````

---

## Causal Self-Attention

### Projections

Given the pre-normed input $\tilde{\boldsymbol{x}} = \operatorname{RMSNorm}(\boldsymbol{x}) \in \mathbb{R}^{T \times d_{\text{model}}}$, compute queries, keys, and values via separate linear projections (no bias):

$$\begin{aligned}
\boldsymbol{Q} &= \tilde{\boldsymbol{x}}\, \boldsymbol{W}_Q^\top \in \mathbb{R}^{T \times H \cdot d_h}, &
\boldsymbol{W}_Q &\in \mathbb{R}^{(H \cdot d_h) \times d_{\text{model}}}, \\
\boldsymbol{K} &= \tilde{\boldsymbol{x}}\, \boldsymbol{W}_K^\top \in \mathbb{R}^{T \times H_{\text{kv}} \cdot d_h}, &
\boldsymbol{W}_K &\in \mathbb{R}^{(H_{\text{kv}} \cdot d_h) \times d_{\text{model}}}, \\
\boldsymbol{V} &= \tilde{\boldsymbol{x}}\, \boldsymbol{W}_V^\top \in \mathbb{R}^{T \times H_{\text{kv}} \cdot d_h}, &
\boldsymbol{W}_V &\in \mathbb{R}^{(H_{\text{kv}} \cdot d_h) \times d_{\text{model}}}.
\end{aligned}$$

Reshape into per-head tensors: $\boldsymbol{q}_h, \boldsymbol{k}_h, \boldsymbol{v}_h \in \mathbb{R}^{T \times d_h}$ for $h = 1, \dots, H$.

### Value Embeddings (ResFormer)

On alternating layers (and always on the final layer), nanochat adds a *value embedding*---a second embedding lookup that is mixed into the value vector through an input-dependent gate.

Let $\boldsymbol{e}_V^{(\ell)} = \boldsymbol{W}_{VE}^{(\ell)}[\boldsymbol{t}] \in \mathbb{R}^{T \times H_{\text{kv}} \cdot d_h}$ be the value embedding for layer $\ell$. The gate is computed from the first 12 channels of the input:

$$
\boldsymbol{g}^{(\ell)} = 3 \cdot \sigma\!\left(\boldsymbol{x}_{:,:12}\, \boldsymbol{W}_{\text{gate}}^{(\ell)\top}\right) \in \mathbb{R}^{T \times H_{\text{kv}}},
$$

where $\sigma$ is the sigmoid function and $\boldsymbol{W}_{\text{gate}}^{(\ell)} \in \mathbb{R}^{H_{\text{kv}} \times 12}$. The factor of 3 means the gate range is $(0, 3)$.

The gated value embedding is added to the values:

$$
\boldsymbol{v}_h \leftarrow \boldsymbol{v}_h + g_h \cdot \boldsymbol{e}_{V,h}^{(\ell)}, \qquad h = 1, \dots, H_{\text{kv}},
$$

where $g_h \in \mathbb{R}^{T \times 1}$ is broadcast over the head dimension.

The layer selection rule for which layers receive value embeddings is:

$$
\text{has\_ve}(\ell) = \bigl[\ell \bmod 2 = (L-1) \bmod 2\bigr],
$$

which gives alternating layers, with the last layer always included.

````{admonition} Worked Example: Value embedding gating for a single head
:class: tip
Suppose at layer $\ell$, for a single token and one KV head, we have:

**Value from projection:** $\boldsymbol{v} = (0.4, -0.2, 0.1)^\top$ (toy $d_h=3$).

**Value embedding:** $\boldsymbol{e}_V = (0.6, 0.3, -0.5)^\top$ (from the VE lookup table).

**Gate input:** The first 12 channels of $\boldsymbol{x}$ are fed through $\boldsymbol{W}_{\text{gate}} \in \mathbb{R}^{1 \times 12}$ (one KV head), yielding a scalar logit of, say, $1.2$.

**Gate value:** $g = 3 \cdot \sigma(1.2) = 3 \cdot 0.769 = 2.307$.

**Mixed value:** $\boldsymbol{v}_{\text{new}} = \boldsymbol{v} + g \cdot \boldsymbol{e}_V = (0.4, -0.2, 0.1) + 2.307 \cdot (0.6, 0.3, -0.5)$

$= (0.4 + 1.384,\; -0.2 + 0.692,\; 0.1 - 1.154) = (1.784,\; 0.492,\; -1.054)$.

*Key observation:* The gate of $2.307$ means the value embedding contributes more than $2\times$ the original value vector. At initialization, the gate weights are small positive values, so $g \approx 3\sigma(0) \approx 1.5$, which starts the mixing near the midpoint. As training progresses, the model learns to modulate this per-head, per-token.
````

### QK-Norm with Sharpening

After applying RoPE to $\boldsymbol{q}_h$ and $\boldsymbol{k}_h$, nanochat applies RMSNorm *per head* to both queries and keys, then scales by a sharpening factor $\alpha = 1.15$:

$$\begin{aligned}
\hat{\boldsymbol{q}}_h &= \alpha \cdot \operatorname{RMSNorm}(\operatorname{RoPE}(\boldsymbol{q}_h)), \\
\hat{\boldsymbol{k}}_h &= \alpha \cdot \operatorname{RMSNorm}(\operatorname{RoPE}(\boldsymbol{k}_h)).
\end{aligned}$$

```{admonition} Remark
:class: note
QK-Norm prevents attention logits from growing with model dimension or training time, which stabilizes training. The split scaling ($\alpha$ on both $Q$ and $K$ rather than $\alpha^2$ on one) ensures the effective temperature is $\alpha^2 = 1.3225$, sharpening the attention distribution. Conventional scaled dot-product attention uses $1/\sqrt{d_h}$ scaling; here, since the RMSNorm already normalizes $\boldsymbol{q}_h$ and $\boldsymbol{k}_h$ to unit RMS, the inner products are bounded regardless of $d_h$, and the $\alpha$ factor is the *only* temperature control.
```

````{admonition} Worked Example: QK-Norm sharpens attention --- $T=3$ sequence, one head, $d_h = 4$
:class: tip
Consider three post-RoPE query/key vectors (one head, $d_h=4$):

$$\begin{aligned}
\boldsymbol{q}_1 &= (0.8, -0.3, 0.5, 0.2), &
\boldsymbol{k}_1 &= (0.7, -0.4, 0.6, 0.1), \\
&& \boldsymbol{k}_2 &= (0.1, 0.9, -0.2, 0.3), \\
&& \boldsymbol{k}_3 &= (-0.5, 0.1, 0.4, -0.8).
\end{aligned}$$

**Step 1: RMSNorm.** For $\boldsymbol{q}_1$: RMS $= \sqrt{(0.64+0.09+0.25+0.04)/4} = \sqrt{0.255} = 0.505$.
So $\boldsymbol{q}_1^{\text{norm}} = \boldsymbol{q}_1 / 0.505 = (1.584, -0.594, 0.990, 0.396)$.

Similarly normalize all keys. Each now has unit RMS.

**Step 2: Sharpen.** Multiply by $\alpha = 1.15$:
$\hat{\boldsymbol{q}}_1 = 1.15 \cdot \boldsymbol{q}_1^{\text{norm}} = (1.822, -0.683, 1.139, 0.455)$.

**Step 3: Attention logits** (no $1/\sqrt{d_h}$ --- already normalized!).
After normalizing all keys and scaling, suppose the dot products yield:

$$
s_{1,1} = 1.28, \quad s_{1,2} = -0.41, \quad s_{1,3} = 0.15.
$$

**Step 4: Softmax.**
$\boldsymbol{a}_1 = \operatorname{softmax}(1.28, -0.41, 0.15) = (0.605, 0.111, 0.195)$ (after causal masking, all visible).

*Why the sharpening matters:* Without $\alpha$, the logits would be $s/1.3225$, giving $\operatorname{softmax}(0.968, -0.310, 0.113) = (0.507, 0.141, 0.216)$ --- much flatter. The $\alpha=1.15$ factor increases the effective temperature by $\alpha^2 = 1.3225$, concentrating more mass on the strongest match.
````

### Scaled Dot-Product Attention with Sliding Window

The attention output for head $h$ at position $i$ is:

$$
\boldsymbol{o}_{h,i} = \sum_{j=1}^{T} a_{h,i,j}\, \boldsymbol{v}_{h,j}, \qquad
a_{h,i,j} = \frac{\exp(s_{h,i,j})}{\sum_{j'} \exp(s_{h,i,j'})},
$$

where the logit $s_{h,i,j}$ is:

$$
s_{h,i,j} = \begin{cases}
\hat{\boldsymbol{q}}_{h,i}^\top \hat{\boldsymbol{k}}_{h,j} & \text{if } j \leq i \text{ and } i - j < w^{(\ell)}, \\
-\infty & \text{otherwise},
\end{cases}
$$

and $w^{(\ell)}$ is the window size for layer $\ell$. Note there is **no** $1/\sqrt{d_h}$ factor: the QK-Norm already controls the scale.

#### Sliding Window Pattern

The window pattern `"SSSL"` is tiled across $L$ layers. Each character maps to a window size:

$$\begin{aligned}
w_L &= T \quad (\text{"L" = long, full context}), \\
w_S &= \left\lceil \frac{T/3}{128} \right\rceil \cdot 128 \quad (\text{"S" = short, ceil to FA3 tile size}).
\end{aligned}$$

For $T = 2048$: $w_S = \lceil 682.67 / 128 \rceil \cdot 128 = 6 \cdot 128 = 768$. The last layer *always* gets $w_L$ regardless of the pattern.

### Output Projection

Concatenate across heads and project:

$$
\text{Attn}(\boldsymbol{x}) = \bigl[\boldsymbol{o}_1 \,\|\, \boldsymbol{o}_2 \,\|\, \cdots \,\|\, \boldsymbol{o}_H\bigr]\, \boldsymbol{W}_O^\top, \qquad \boldsymbol{W}_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}.
$$

---

## Feed-Forward Network (MLP)

The MLP uses a **squared ReLU** activation ("ReluSquared"):

$$
\text{MLP}(\boldsymbol{x}) = \bigl(\text{ReLU}(\boldsymbol{x}\, \boldsymbol{W}_1^\top)\bigr)^2\, \boldsymbol{W}_2^\top,
$$

where $\boldsymbol{W}_1 \in \mathbb{R}^{4d_{\text{model}} \times d_{\text{model}}}$ and $\boldsymbol{W}_2 \in \mathbb{R}^{d_{\text{model}} \times 4d_{\text{model}}}$. All biases are zero (not present).

```{admonition} Remark
:class: note
ReluSquared ($\phi(z) = (\max(0,z))^2$) is sparser than GELU and produces sharper gradients. Its derivative is $\phi'(z) = 2\max(0,z)$, which is zero for $z \leq 0$ and linear for $z > 0$. Unlike SwiGLU (used in LLaMA), this does not gate with a parallel branch---the expansion factor is the classical $4\times$.
```

````{admonition} Worked Example: ReluSquared amplifies sparsity
:class: tip
Consider a single token's hidden state $\boldsymbol{x} = (0.5, -0.3)^\top$ ($d_{\text{model}}=2$, so MLP expands to $4d = 8$). Suppose $\boldsymbol{W}_1 \in \mathbb{R}^{8 \times 2}$ maps this to:

$$
\boldsymbol{h} = \boldsymbol{W}_1 \boldsymbol{x} = (1.2,\; -0.8,\; 0.3,\; -1.5,\; 0.05,\; 2.1,\; -0.1,\; 0.7)^\top.
$$

**After ReLU:** $(1.2,\; 0,\; 0.3,\; 0,\; 0.05,\; 2.1,\; 0,\; 0.7)$. *(5 of 8 survive, 3 zeroed.)*

**After squaring:** $(1.44,\; 0,\; 0.09,\; 0,\; 0.0025,\; 4.41,\; 0,\; 0.49)$.

*Key observation:* The squaring dramatically amplifies the contrast. The ratio between the largest and smallest nonzero activations went from $2.1/0.05 = 42\times$ (after ReLU) to $4.41/0.0025 = 1764\times$ (after squaring). Small activations become negligible; large ones dominate. This is much sparser than GELU, where $\text{GELU}(-0.8) \approx -0.17$ would still contribute. The $\boldsymbol{W}_2$ projection then maps this sparse 8-vector back to a 2-vector update to the residual stream.
````

---

## Residual Stream with Learnable Scalars

The Transformer block computes:

$$\begin{aligned}
\boldsymbol{x} &\leftarrow \lambda_{\text{resid}}^{(\ell)}\, \boldsymbol{x} + \lambda_{x_0}^{(\ell)}\, \boldsymbol{x}_0, \\
\boldsymbol{x} &\leftarrow \boldsymbol{x} + \text{Attn}_\ell\bigl(\operatorname{RMSNorm}(\boldsymbol{x})\bigr), \\
\boldsymbol{x} &\leftarrow \boldsymbol{x} + \text{MLP}_\ell\bigl(\operatorname{RMSNorm}(\boldsymbol{x})\bigr).
\end{aligned}$$

Here $\lambda_{\text{resid}}^{(\ell)}$ and $\lambda_{x_0}^{(\ell)}$ are **per-layer learnable scalars**. At initialization: $\lambda_{\text{resid}}^{(\ell)} = 1.0$ (standard residual) and $\lambda_{x_0}^{(\ell)} = 0.1$ (small skip from initial embedding).

```{admonition} Remark
:class: note
The $x_0$ residual connection allows the model to "look back" at the initial embedding throughout the network, inspired by work on residual stream re-injection. The scalars are optimized with AdamW (separate from the Muon optimizer used for matrix parameters).
```

---

## Logit Softcapping and Loss

### Logit Softcap

The raw logits $\boldsymbol{z} = \boldsymbol{x}^{(L)} \boldsymbol{W}_U^\top \in \mathbb{R}^{T \times V}$ are squashed via:

$$
\hat{z}_i = c \cdot \tanh\!\left(\frac{z_i}{c}\right), \qquad c = 15.
$$

This smoothly constrains logits to $(-15, 15)$, preventing extreme probability spikes.

````{admonition} Worked Example: Logit softcapping in action
:class: tip
Consider five raw logits from the LM head for a single position across five vocabulary tokens:

$$
z = (-30.0,\; -2.0,\; 0.5,\; 8.0,\; 50.0).
$$

Apply $\hat{z}_i = 15 \cdot \tanh(z_i / 15)$:

| $z_i$ | $z_i/15$ | $\tanh(z_i/15)$ | $\hat{z}_i$ | *Effect* |
|---|---|---|---|---|
| $-30.0$ | $-2.000$ | $-0.964$ | $-14.46$ | heavily compressed |
| $-2.0$ | $-0.133$ | $-0.133$ | $-1.99$ | nearly unchanged |
| $0.5$ | $0.033$ | $0.033$ | $0.50$ | unchanged (linear regime) |
| $8.0$ | $0.533$ | $0.489$ | $7.33$ | mild compression |
| $50.0$ | $3.333$ | $1.000$ | $15.00$ | saturated at cap |

*Key observation:* Small logits ($|z| \ll 15$) pass through nearly unchanged because $\tanh(x) \approx x$ for small $x$. But the extreme logit of $50.0$ is crushed to $15.0$, and $-30.0$ is pulled to $-14.46$. Without the cap, $\operatorname{softmax}$ would assign essentially all probability mass to the $z=50$ token; with the cap, the distribution remains peaky but not degenerate. The gradient $\partial\hat{z}/\partial z = 1 - \tanh^2(z/c)$ also vanishes for saturated logits, providing an implicit gradient clipping effect.
````

### Cross-Entropy Loss

For targets $y_t \in \{0, \dots, V-1\}$, the loss is:

$$
\mathcal{L} = -\frac{1}{|\mathcal{T}|}\sum_{t \in \mathcal{T}} \log \frac{\exp(\hat{z}_{t, y_t})}{\sum_{v=1}^{V} \exp(\hat{z}_{t,v})},
$$

where $\mathcal{T} = \{t : y_t \neq -1\}$ excludes padding tokens.

### Untied Embeddings

Unlike GPT-2, nanochat uses **separate** weight matrices for the token embedding ($\boldsymbol{W}_E$) and the language model head ($\boldsymbol{W}_U$). They are *not* tied.

---

## Weight Initialization

The initialization is designed so that at step 0:

- The output projection ($\boldsymbol{W}_O$) and MLP output ($\boldsymbol{W}_2$) are **zero**, making each block an identity at init.
- Input projections use uniform distributions (fewer outliers than Gaussian) with $\text{std} = 1/\sqrt{d_{\text{model}}}$.

Let $s = \sqrt{3/d_{\text{model}}}$ (the $\sqrt{3}$ factor ensures $\text{Uniform}(-s, s)$ has the same standard deviation as $\mathcal{N}(0, 1/d_{\text{model}})$).

| **Parameter** | **Distribution** | **Scale** | **Code ref** |
|---|---|---|---|
| $\boldsymbol{W}_E$ (embedding) | $\mathcal{N}(0, \sigma^2)$ | $\sigma = 0.8$ | line 213 |
| $\boldsymbol{W}_U$ (lm\_head) | $\mathcal{N}(0, \sigma^2)$ | $\sigma = 0.001$ | line 214 |
| $\boldsymbol{W}_Q, \boldsymbol{W}_K, \boldsymbol{W}_V$ | $\text{Uniform}(-s, s)$ | $s = \sqrt{3/d_{\text{model}}}$ | lines 220--222 |
| $\boldsymbol{W}_O$ (c\_proj, attn) | zeros | $0$ | line 223 |
| $\boldsymbol{W}_1$ (c\_fc, MLP) | $\text{Uniform}(-s/2, s/2)$ | $s/2$ | line 224 |
| $\boldsymbol{W}_2$ (c\_proj, MLP) | zeros | $0$ | line 225 |
| Value embeds $\boldsymbol{W}_{VE}$ | $\text{Uniform}(-s, s)$ | $s = \sqrt{3/d_{\text{model}}}$ | line 233 |
| VE gate $\boldsymbol{W}_{\text{gate}}$ | $\text{Uniform}(0, 0.02)$ | small positive | line 238 |

```{admonition} Remark
:class: note
The zero-init of $\boldsymbol{W}_O$ and $\boldsymbol{W}_2$ means the Transformer starts as an identity function (each block contributes zero). This is a form of "fixup" initialization that allows training deep networks without learning rate warmup for residual contributions. The MLP input weights ($\boldsymbol{W}_1$) get half the scale ($s/2$) of the attention input weights, which empirically improves convergence.
```

---

## The AdamW Optimizer

AdamW is used for embeddings ($\boldsymbol{W}_E$), the unembedding head ($\boldsymbol{W}_U$), value embeddings ($\boldsymbol{W}_{VE}$), and the per-layer scalars ($\lambda_{\text{resid}}, \lambda_{x_0}$).

### Update Rule

Given parameter $\boldsymbol{\theta}$, gradient $\boldsymbol{g}_t = \nabla_\theta \mathcal{L}_t$, and hyperparameters $(\eta, \beta_1, \beta_2, \epsilon, \lambda)$:

1. **Decoupled weight decay**: $\boldsymbol{\theta} \leftarrow (1 - \eta\lambda)\,\boldsymbol{\theta}$.
2. **First moment**: $\boldsymbol{m}_t \leftarrow \beta_1\, \boldsymbol{m}_{t-1} + (1-\beta_1)\,\boldsymbol{g}_t$.
3. **Second moment**: $\boldsymbol{v}_t \leftarrow \beta_2\, \boldsymbol{v}_{t-1} + (1-\beta_2)\,\boldsymbol{g}_t^2$.
4. **Bias correction**: $\hat{\boldsymbol{m}}_t = \boldsymbol{m}_t / (1 - \beta_1^t)$, $\quad \hat{\boldsymbol{v}}_t = \boldsymbol{v}_t / (1 - \beta_2^t)$.
5. **Update**: $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\, \hat{\boldsymbol{m}}_t / (\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon)$.

### AdamW Parameter Groups

Each group has different hyperparameters, with learning rates scaled by $\sqrt{768/d_{\text{model}}}$ (a $\mu$P-style correction):

| **Group** | $\eta_{\text{base}}$ | $(\beta_1, \beta_2)$ | $\epsilon$ | $\lambda$ |
|---|---|---|---|---|
| lm\_head | $0.004$ | $(0.8, 0.96)$ | $10^{-10}$ | $0.01$ |
| wte (embedding) | $0.2$ | $(0.8, 0.995)$ | $10^{-10}$ | $0.001$ |
| value embeds | $0.1$ | $(0.8, 0.995)$ | $10^{-10}$ | $0.01$ |
| $\lambda_{\text{resid}}$ | $0.005$ | $(0.8, 0.95)$ | $10^{-10}$ | $0.05$ |
| $\lambda_{x_0}$ | $0.5$ | $(0.96, 0.95)$ | $10^{-10}$ | $0.0$ |

All base learning rates are further multiplied by $\sqrt{B/B_{\text{ref}}}$ (batch-size scaling; see the Scaling Laws section).

---

## The Muon Optimizer

Muon (MomentUm Orthogonalized by Newton-Schulz) is used for all 2D matrix parameters in the Transformer blocks. The key idea: after computing a momentum-based gradient estimate, replace it with the **nearest orthogonal matrix**, which serves as a steepest descent direction under the spectral norm.

### Nesterov Momentum

Given stacked gradients $\boldsymbol{G}_t$ and momentum buffer $\boldsymbol{M}_{t-1}$, with momentum coefficient $\mu$:

$$\begin{aligned}
\boldsymbol{M}_t &= \mu\, \boldsymbol{M}_{t-1} + (1 - \mu)\, \boldsymbol{G}_t, \\
\tilde{\boldsymbol{G}}_t &= \mu\, \boldsymbol{M}_t + (1 - \mu)\, \boldsymbol{G}_t.
\end{aligned}$$

This is the Nesterov form, implemented via the two `lerp_` calls.

```{admonition} Remark
:class: note
The second line modifies `stacked_grads` in-place. Writing $\boldsymbol{b}$ for the buffer and $\boldsymbol{g}$ for the gradient: after line 1, $\boldsymbol{b} \leftarrow \mu \boldsymbol{b} + (1-\mu)\boldsymbol{g}$. After line 2, $\boldsymbol{g} \leftarrow (1-\mu)\boldsymbol{g} + \mu \boldsymbol{b}$, which is the Nesterov "look-ahead" combining current gradient with the updated momentum.
```

### Polar Express Orthogonalization

The goal is to approximate the *polar factor* of $\tilde{\boldsymbol{G}}_t$. Recall that any matrix $\boldsymbol{G}$ with SVD $\boldsymbol{G} = \boldsymbol{U}\boldsymbol{S}\boldsymbol{V}^\top$ has polar decomposition $\boldsymbol{G} = (\boldsymbol{U}\boldsymbol{V}^\top)(\boldsymbol{V}\boldsymbol{S}\boldsymbol{V}^\top)$, and the orthogonal factor $\boldsymbol{U}\boldsymbol{V}^\top$ is the nearest orthogonal matrix to $\boldsymbol{G}$ in Frobenius norm.

Rather than computing the SVD (expensive), nanochat uses the **Polar Express Sign Method**, an iterative algorithm from Amsel et al. (2025).

#### Algorithm

Normalize: $\boldsymbol{X}_0 = \tilde{\boldsymbol{G}}_t / (1.01 \cdot \lVert \tilde{\boldsymbol{G}}_t \rVert_F + 10^{-6})$.

Then for $k = 0, 1, \dots, N_s - 1$ (with $N_s = 5$ steps), using pre-computed coefficients $(a_k, b_k, c_k)$:

**Case 1: Tall matrix** ($m \geq n$, i.e., rows $\geq$ columns):

$$\begin{aligned}
\boldsymbol{A}_k &= \boldsymbol{X}_k^\top \boldsymbol{X}_k, \\
\boldsymbol{B}_k &= b_k \boldsymbol{A}_k + c_k \boldsymbol{A}_k^2, \\
\boldsymbol{X}_{k+1} &= a_k \boldsymbol{X}_k + \boldsymbol{X}_k \boldsymbol{B}_k.
\end{aligned}$$

**Case 2: Wide matrix** ($m < n$):

$$\begin{aligned}
\boldsymbol{A}_k &= \boldsymbol{X}_k \boldsymbol{X}_k^\top, \\
\boldsymbol{B}_k &= b_k \boldsymbol{A}_k + c_k \boldsymbol{A}_k^2, \\
\boldsymbol{X}_{k+1} &= a_k \boldsymbol{X}_k + \boldsymbol{B}_k \boldsymbol{X}_k.
\end{aligned}$$

After $N_s$ iterations, $\boldsymbol{X}_{N_s} \approx \boldsymbol{U}\boldsymbol{V}^\top$ (or a close approximation thereof).

The coefficients $(a_k, b_k, c_k)$ for $k=0,\dots,4$ are:

| $k$ | $a_k$ | $b_k$ | $c_k$ |
|---|---|---|---|
| 0 | $8.1566$ | $-22.4833$ | $15.8788$ |
| 1 | $4.0429$ | $-2.8089$ | $0.5000$ |
| 2 | $3.8917$ | $-2.7725$ | $0.5061$ |
| 3 | $3.2858$ | $-2.3681$ | $0.4645$ |
| 4 | $2.3465$ | $-1.7098$ | $0.4232$ |

```{admonition} Remark
:class: note
Why the tall/wide distinction? For a tall matrix $\boldsymbol{X} \in \mathbb{R}^{m \times n}$ with $m \geq n$, the Gram matrix $\boldsymbol{X}^\top\boldsymbol{X} \in \mathbb{R}^{n \times n}$ is smaller, so multiplying $\boldsymbol{X} \boldsymbol{B}$ is cheaper. For wide matrices, the analogous trick uses $\boldsymbol{X}\boldsymbol{X}^\top \in \mathbb{R}^{m \times m}$ instead. Both converge to the same orthogonal factor.
```

````{admonition} Worked Example: Polar Express on a $2\times 2$ gradient matrix
:class: tip
Consider a (post-momentum) gradient matrix that we want to orthogonalize:

$$
\boldsymbol{G} = \begin{pmatrix} 3.0 & 1.0 \\ 0.5 & 2.0 \end{pmatrix}.
$$

**True polar factor.** The SVD is $\boldsymbol{G} = \boldsymbol{U}\boldsymbol{S}\boldsymbol{V}^\top$, and the orthogonal polar factor is $\boldsymbol{U}\boldsymbol{V}^\top$. Computing numerically:

$$
\boldsymbol{U}\boldsymbol{V}^\top \approx \begin{pmatrix} 0.932 & -0.362 \\ 0.362 & 0.932 \end{pmatrix}.
$$

**Polar Express iteration.** This is a square (wide) matrix, so we use the wide-matrix variant. Normalize: $\lVert\boldsymbol{G}\rVert_F = \sqrt{9+1+0.25+4} = 3.775$, so $\boldsymbol{X}_0 = \boldsymbol{G} / (1.01 \cdot 3.775) = \boldsymbol{G} / 3.813$.

$$
\boldsymbol{X}_0 \approx \begin{pmatrix} 0.787 & 0.262 \\ 0.131 & 0.525 \end{pmatrix}.
$$

**Iteration 0** ($a_0=8.157$, $b_0=-22.483$, $c_0=15.879$):
$\boldsymbol{A}_0 = \boldsymbol{X}_0 \boldsymbol{X}_0^\top$,
$\boldsymbol{B}_0 = b_0 \boldsymbol{A}_0 + c_0 \boldsymbol{A}_0^2$,
$\boldsymbol{X}_1 = a_0 \boldsymbol{X}_0 + \boldsymbol{B}_0 \boldsymbol{X}_0$.

After this single iteration:
$\boldsymbol{X}_1 \approx \begin{pmatrix} 0.97 & -0.31 \\ 0.31 & 0.97 \end{pmatrix}$
--- already very close to $\boldsymbol{U}\boldsymbol{V}^\top$!

After all 5 iterations: $\boldsymbol{X}_5 \approx \begin{pmatrix} 0.932 & -0.362 \\ 0.362 & 0.932 \end{pmatrix}$ --- converged to the true polar factor.

*Key observation:* The Polar Express iteration converges rapidly. The first iteration does most of the work (the aggressive coefficients $a_0, b_0, c_0$ are designed to have maximal slope at zero). Later iterations refine. In practice, for the $768 \times 3072$ weight matrices of a d12 model, 5 iterations in `bfloat16` suffice.
````

### NorMuon Variance Reduction

After orthogonalization, the update matrix $\boldsymbol{X}_{N_s}$ may have non-uniform variance across rows (or columns). The NorMuon correction normalizes these scales using an exponential moving average of per-neuron variances.

Let $\boldsymbol{g} = \boldsymbol{X}_{N_s}$ (the orthogonalized update). Choose the reduction dimension:

$$
\text{red\_dim} = \begin{cases} -1 & \text{if } m \geq n \text{ (reduce over columns)},\\ -2 & \text{if } m < n \text{ (reduce over rows)}.\end{cases}
$$

1. Compute per-neuron mean squared values:

$$
\boldsymbol{v}_{\text{mean}} = \frac{1}{d_{\text{red}}} \sum_{\text{red\_dim}} \boldsymbol{g}^2 \in \mathbb{R}^{m \times 1} \text{ (or } \mathbb{R}^{1 \times n}).
$$

2. Current total variance: $\lVert\boldsymbol{v}\rVert_{\text{norm}} = \sqrt{\sum \boldsymbol{v}_{\text{mean}} \cdot d_{\text{red}}}$.

3. Update EMA of per-neuron variance:

$$
\boldsymbol{s}_t = \beta_2'\, \boldsymbol{s}_{t-1} + (1 - \beta_2')\, \boldsymbol{v}_{\text{mean}}, \qquad \beta_2' = 0.9.
$$

4. Compute scale: $\text{step\_size} = 1/\sqrt{\max(\boldsymbol{s}_t, 10^{-10})}$.

5. Compute rescaled variance norm:

$$
\lVert\boldsymbol{v}\rVert_{\text{new}} = \sqrt{\sum (\boldsymbol{v}_{\text{mean}} \cdot d_{\text{red}}) \cdot \text{step\_size}^2}.
$$

6. Final per-neuron scale: $\text{final\_scale} = \text{step\_size} \cdot \lVert\boldsymbol{v}\rVert_{\text{norm}} / \lVert\boldsymbol{v}\rVert_{\text{new}}$. This rescales each neuron while preserving the overall norm of the update.

7. Apply: $\boldsymbol{g} \leftarrow \boldsymbol{g} \odot \text{final\_scale}$.

### Cautious Weight Decay and Parameter Update

The final update combines cautious masking with weight decay:

$$
\boldsymbol{m}_{i,j} = \mathbb{1}\bigl[g_{i,j} \cdot \theta_{i,j} \geq 0\bigr],
$$

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\, \boldsymbol{g} - \eta\lambda\, \boldsymbol{\theta} \odot \boldsymbol{m}.
$$

The "cautious" mask $\boldsymbol{m}$ only applies weight decay in directions where the update and the parameter agree in sign, preventing decay from fighting the gradient.

### Muon Learning Rate Scaling

For non-square matrices, the Muon learning rate is scaled:

$$
\eta_{\text{Muon}} = \eta_{\text{base}} \cdot \sqrt{\max\!\left(1,\; \frac{m}{n}\right)},
$$

where $m \times n$ is the shape of the weight matrix.

---

## Scaling Laws and Hyperparameter Transfer

All hyperparameters are tuned at depth $L=12$ ("d12") and transferred to other depths via $\mu$P-style scaling laws.

### Compute-Optimal Training Horizon

The training token budget is:

$$
D^* = R \cdot N_{\text{scale}},
$$

where $R$ is the target param-to-data ratio (default: tuned experimentally) and $N_{\text{scale}}$ counts "scaling parameters" (transformer matrix params + lm\_head, excluding embeddings and scalars).

### Optimal Batch Size (Power Lines)

Following Sardana et al. (2025), the optimal batch size scales as:

$$
B^* = B_{\text{ref}} \cdot \left(\frac{D^*}{D_{\text{ref}}}\right)^{0.383},
$$

with $B_{\text{ref}} = 2^{19} = 524{,}288$ tokens (the optimal batch size at d12). The result is rounded to the nearest power of 2 for hardware efficiency.

### Learning Rate Scaling

For AdamW (and Muon, by assumption):

$$
\eta = \eta_{\text{base}} \cdot \sqrt{\frac{B}{B_{\text{ref}}}}.
$$

Additionally, all AdamW learning rates are scaled by $\mu$P correction:

$$
\eta_{\text{AdamW}} = \eta_{\text{base}} \cdot \sqrt{\frac{768}{d_{\text{model}}}}.
$$

### Weight Decay Scaling

Following the $T_{\text{epoch}}$ framework (Wen et al., 2024), to keep $T_{\text{epoch}} = B/(\eta\lambda D)$ constant:

$$
\lambda = \lambda_{\text{ref}} \cdot \sqrt{\frac{B}{B_{\text{ref}}}} \cdot \frac{D_{\text{ref}}}{D^*}.
$$

### Learning Rate Schedule

The schedule is **linear warmup, constant, linear warmdown**:

$$
\text{lr\_mult}(t) = \begin{cases}
(t+1) / T_w & \text{if } t < T_w, \\
1 & \text{if } T_w \leq t \leq T - T_d, \\
\frac{T-t}{T_d} + \left(1 - \frac{T-t}{T_d}\right) \cdot \eta_f & \text{if } t > T - T_d,
\end{cases}
$$

where $T_w = 40$ (warmup steps), $T_d = \lfloor 0.3 T \rfloor$ (warmdown ratio), and $\eta_f = 0.05$ (final LR fraction).

### Muon Momentum Schedule

The Muon momentum warms up linearly from 0.85 to 0.97 over the first 400 steps:

$$
\mu(t) = 0.85 + \min\!\left(\frac{t}{400},\, 1\right) \cdot (0.97 - 0.85).
$$

### Muon Weight Decay Schedule

Muon weight decay follows a cosine schedule decaying to zero:

$$
\lambda(t) = \frac{\lambda_{\text{scaled}}}{2}\left(1 + \cos\frac{\pi t}{T}\right).
$$

---

## FLOPs Estimation

The per-token FLOP count (forward + backward) is:

$$
F = 6 \cdot (N - N_{\text{excl}}) + \sum_{\ell=1}^{L} 12\, H\, d_h\, \tilde{T}^{(\ell)},
$$

where:

- $N$ is total parameter count.
- $N_{\text{excl}}$ excludes non-matmul params (embeddings, value embeddings, scalars).
- Factor 6 = 2 (forward) + 4 (backward: 2 for param grad, 2 for activation grad).
- $\tilde{T}^{(\ell)} = \min(w^{(\ell)}, T)$ is the effective sequence length at layer $\ell$.
- $12 H d_h \tilde{T}$ counts the QK and QKV matmuls in attention (2 matmuls, each $2 H d_h \tilde{T}$, times forward+backward factor of 3).

---

## Summary: The Complete Algorithm

We close with a complete end-to-end numerical trace and then the pseudocode.

````{admonition} Worked Example: End-to-end trace: two tokens through a $d_{\text{model}}=4$, $L=2$ model
:class: tip
We trace the string `"Hi"` tokenized as two tokens $\boldsymbol{t} = (t_1, t_2)$ through a tiny model with $d_{\text{model}} = 4$, $H = 2$, $d_h = 2$, $L = 2$ layers.

**1. Token Embedding.**
Look up $\boldsymbol{W}_E[t_1]$ and $\boldsymbol{W}_E[t_2]$ (rows of the $V \times 4$ embedding matrix):

$$
\boldsymbol{x} = \begin{pmatrix} 0.8 & -0.3 & 0.5 & 0.2 \\ 0.1 & 0.7 & -0.4 & 0.6 \end{pmatrix} \in \mathbb{R}^{2 \times 4}.
$$

**2. Post-embedding RMSNorm.**
For row 1: RMS $= \sqrt{(0.64+0.09+0.25+0.04)/4} = 0.505$, so $\boldsymbol{x}_1^{\text{norm}} = (1.584, -0.594, 0.990, 0.396)$.
For row 2: RMS $= \sqrt{(0.01+0.49+0.16+0.36)/4} = 0.506$, so $\boldsymbol{x}_2^{\text{norm}} = (0.198, 1.383, -0.791, 1.186)$.
Save $\boldsymbol{x}_0 = \boldsymbol{x}^{\text{norm}}$.

**3. Layer $\ell = 1$: Residual scaling.**
At init: $\lambda_{\text{resid}}^{(1)} = 1.0$, $\lambda_{x_0}^{(1)} = 0.1$.

$$
\boldsymbol{x} \leftarrow 1.0 \cdot \boldsymbol{x} + 0.1 \cdot \boldsymbol{x}_0 = 1.1 \cdot \boldsymbol{x}_0 \quad \text{(since } \boldsymbol{x} = \boldsymbol{x}_0 \text{ in layer 1)}.
$$

**4. Layer $\ell = 1$: Pre-norm $\to$ Attention.**
RMSNorm the current $\boldsymbol{x}$ to get $\tilde{\boldsymbol{x}}$. Project to get $\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}$ (each $\mathbb{R}^{2 \times 2}$ per head).
Apply RoPE at positions 0 and 1. Apply QK-Norm ($\times 1.15$). Compute causal attention:

- Token 1 attends only to itself: $a_{1,1} = 1.0$.
- Token 2 attends to both: e.g., $a_{2,1} = 0.35$, $a_{2,2} = 0.65$ (from softmax of QK logits).

But at initialization, $\boldsymbol{W}_O$ is **zero**, so $\text{Attn}_1(\cdot) = \boldsymbol{0}$. The residual add: $\boldsymbol{x} \leftarrow \boldsymbol{x} + \boldsymbol{0} = \boldsymbol{x}$.

**5. Layer $\ell = 1$: Pre-norm $\to$ MLP.**
Similarly, $\boldsymbol{W}_2$ is zero at init, so $\text{MLP}_1(\cdot) = \boldsymbol{0}$. The residual add: $\boldsymbol{x} \leftarrow \boldsymbol{x} + \boldsymbol{0} = \boldsymbol{x}$.

**6. Layer $\ell = 2$:** Same story. At initialization, every block is the identity.

**7. Final RMSNorm + LM Head.**
$\boldsymbol{x}^{(L)}$ (still $\approx 1.1 \cdot \boldsymbol{x}_0$) gets RMSNorm'd, then multiplied by $\boldsymbol{W}_U \in \mathbb{R}^{V \times 4}$ ($V = 32768$ vocabulary tokens).

**8. Softcap.** $\hat{z}_{i} = 15 \cdot \tanh(z_i / 15)$ squashes any extreme logits.

**9. Loss.** Suppose the target for position 1 is token $t_2$, and target for position 2 is some next token $t_3$.
Cross-entropy loss $\mathcal{L}$ measures how well the softmax over $\hat{\boldsymbol{z}}$ predicts the targets.
At initialization, $\boldsymbol{W}_U$ is near-zero ($\sigma = 0.001$), so logits $\approx 0$ for all vocab tokens, giving $\mathcal{L} \approx \ln V = \ln 32768 \approx 10.4$. This is what you see in the first training step!

*Key insight:* The zero initialization of $\boldsymbol{W}_O$ and $\boldsymbol{W}_2$ means the model starts as "embedding $\to$ normalize $\to$ lm\_head." Training gradually "turns on" each block as $\boldsymbol{W}_O$ and $\boldsymbol{W}_2$ move away from zero. This prevents the instabilities that plague deep networks with random initialization of all layers simultaneously.
````

### Pseudocode: One Training Step

```
Algorithm: Nanochat — One Training Step

Input: Batch of token sequences (t, y), current parameters Θ

1.  x ← RMSNorm(W_E[t])                          # Embed + normalize
2.  x₀ ← x                                        # Save for x0 residual
3.  FOR ℓ = 1 to L:
4.      x ← λ_resid^(ℓ) · x + λ_x0^(ℓ) · x₀     # Learnable residual scaling
5.      x̃ ← RMSNorm(x)
6.      Compute Q, K, V from x̃; optionally add gated value embedding to V
7.      Apply RoPE to Q, K; apply QK-Norm with α = 1.15
8.      x ← x + W_O · FlashAttn(Q, K, V; w^(ℓ))
9.      x ← x + MLP_ℓ(RMSNorm(x))                # ReluSquared activation
10. z ← 15 · tanh(W_U · RMSNorm(x) / 15)         # Softcapped logits
11. L ← CrossEntropy(z, y)
12. Backpropagate ∇_Θ L
13. Update embeddings/scalars with AdamW; update matrices with Muon (Polar Express + NorMuon)
```

---

## References

1. A. Karpathy. `nanochat`: The best ChatGPT that $100 can buy. GitHub, 2025. <https://github.com/karpathy/nanochat>
2. J. Su, Y. Lu, S. Pan, et al. RoFormer: Enhanced Transformer with Rotary Position Embedding. *arXiv:2104.09864*, 2021.
3. B. Zhang and R. Sennrich. Root Mean Square Layer Normalization. *NeurIPS*, 2019.
4. N. Amsel, D. Persson, C. Musco, R. M. Gower. Polar Express Sign Method. *arXiv:2505.16932*, 2025.
5. K. Jordan. Muon: MomentUm Orthogonalized by Newton-Schulz. <https://kellerjordan.github.io/posts/muon/>, 2024.
6. NorMuon variance reduction. *arXiv:2510.05491*, 2025.
7. I. Loshchilov and F. Hutter. Decoupled Weight Decay Regularization. *ICLR*, 2019.
8. J. Hoffmann et al. Training Compute-Optimal Large Language Models (Chinchilla). *arXiv:2203.15556*, 2022.
9. T. Sardana et al. Power Lines: Scaling Laws for Batch Size. *arXiv:2505.13738*, 2025.
10. Y. Wen et al. Understanding $T_\text{epoch}$ in Weight Decay. *arXiv:2405.13698*, 2024.
11. ResFormer: Scaling Transformers via Residual Value Embeddings.
12. S. Narang, H. Chung, et al. Do Transformer Modifications Transfer? *arXiv:2102.11972*, 2021.
