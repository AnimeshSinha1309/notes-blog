
This is what the original "Attention is all you need" transformer looks like. We want to understand each bit and piece and how it has evolved.

![[image-transformer-architecture.png]]

## Tokenizer
### What makes a good tokenizer

These are some metrics that we have 
* **Compression Ratio**: Bytes of text that are encoded per token, on average. Higher compression implies smaller sequences, hence better.
* **Vocabulary Size**: Number of different tokens in our dictionary, should not be too huge and should be efficiently used (i.e. some tokens shouldn't appear almost never)
* **Probability of out-of-vocabulary tokens**: When a token cannot be encoded and has to be replaced with a UNK placeholder, it makes computing perplexity hard. This is bad and should happen less often.

Another property we want from tokenizers:
* **Reversibility**: From the tokens it should be possible to 

### Tokenization Algorithms of Today

| Tokenization Algorithm   | How it works                                                                                                                   | Benefits / Losses                                                                                                                                                                                                                                                    |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Character Encoding       | Each unicode character (~150k, may span multiple bytes) gets it own encoding                                                   | Con: Very large and sparse token dictionary, most of the characters never appear in text.<br>Con: Bad compression ratio (~1)                                                                                                                                         |
| Byte Encoding            | Each byte gets it's own encoding, a single character can become multiple tokens.                                               | Pro: Very small token dictionary, only 256 supported characters.<br>Con: Let's to high Sequence Length (bad compression ratio = 1).                                                                                                                                  |
| Word Encoding            | Each word in our dictionary gets its own token.                                                                                | Con: Number of words is huge and most words are rare enough that the model doesn't learn about them. (word parts aren't understood)<br>Con: Doesn't provide a fixed vocabulary size<br>Con: New words get mapped to UNK tokens, cause issues in computing perplexity |
| Byte Pair Encoding (BPE) | Take all bytes in our raw text, find the most commonly co-occuring pairs and reassign it to a new token, continue recursively. | Pro: Understanding of word parts gets developed<br>Pro: Balances well between vocabulary size and                                                                                                                                                                    |

Some interesting insights about specifics of tokenizers:
* In many algorithms, e.g. GPT-4 tokenizer, `[ hello]` (space preceeding) and `[Hello]` are completely different tokens, so words in the start vs. middle of sentence are just different tokens.
* Numbers get chunked up into sets 2-3 of digits, the splitting up is left to right, so adding more digits doesn't rechunk the prefix. 
* GPT-2 has a pre-tokenizer which splits the input text via a regex, splitting on spaces and other special characters to ensure that no token spans multiple words

## Architecture of the Transformer
### Pre vs. Post Normalization

**Pre-norm is a stability inducing aid for training**.

Pre-norm + other stability inducing aids help
* Gradient attenuation across layer is more in post norm. Some layer gradients are very small, others are very high.
* Pre-norm is a more stable architecture to train, loss spikes are more in post-norm in the norm gradient.
* We possibly don't want to put in any normalizers in the residual stream. It is supposed to be an identity connection which helps in training very deep networks. Post norm messes with that.

A new technique called double norm exists where in the non-residual branch, we normalize pre and post both.

<span><img src="image-pre-vs-post-norm.png" height ="400px"/>  <img src="image-double-norm.png" height ="400px"/></span>
### Layer Norm vs. RMS Norm

Layer norm converts input distribution to a $\beta$ mean, $\gamma$ variance distribution.
$$y = \frac{x - \mathbb{E}(x)}{\sqrt{\text{Var}(x) + \epsilon}} \times \gamma + \beta$$

RMS norm doesn't subtract mean or add a bias terms, making it simpler.
Empirically it's faster and just as good.
* No mean to compute and subtract - so faster (but this is very small in num-flops)
* No bias term to store and retrieve - so less memory movement
$$y = \frac{x}{\sqrt{\big\vert \vert x \vert \big\vert_2^2 + \epsilon}} \times \gamma$$
Surprisingly, RMSNorm transformers also do slightly better in final performance.

## Feed Forward Layers (Activation Function + Gating)

### No bias terms in FFNs

Original Transformer: $$\text{FF}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2$$
Modern transformers: $$\text{FF}(x) = \sigma(xW_1)W_2$$

### Popular activation function variants 

Beyond ReLU, people attempt to make things more differentiable at 0.

| Activation        | Definition                                                | Color in Graph |
| ----------------- | --------------------------------------------------------- | -------------- |
| $\text{ReLU}(x)$  | $\max(x, 0)$                                              | red            |
| $\text{GeLU}(x)$  | $x \cdot \Phi_{\text{Gaussian-CDF}}(x)$                   | green          |
| $\text{Swish}(x)$ | $x \cdot \frac{1}{1 + e^{-x}}$                            | blue           |
| $\text{ELU}(x)$   | $\begin{cases}x&x\geq0\\\alpha(e^{x}-1)&x\lt0\end{cases}$ | black          |

```desmos-graph
left=-5;right=5;top=5;bottom=-2;
---
y=\max(x, 0)|dashed|red
y=x*0.5*(1 + \erf(x*\sqrt{1/2}))|dashed|green
y=x/(1+e^{-x})|dashed|blue
y=\max(x,e^x-1)|dashed|black
```

### Gated Linear Units

Learned parameters $V$ act as a gating term. 
$$\text{FFN}(x) = \big(\sigma(xW_1)\otimes xV\big)W_2$$
* GLU
* GeGLU ✅ ($\sigma = \text{GeLU}$)
* ReGLU
* SeLU
* SwiGLU ✅ ($\sigma = \text{Swish}$)
* LiGLU

Gates units are not always better, but SwiGLU and GeGLU usually show better results.
This gating has learned parameters V which add some cost.
### Serial vs. Parallel Layers

MLP is applied to the attention output, so it's serial and hard to shard across worker GPUs.
$$y = x + \text{MLP}(\text{LayerNorm}(x + \text{Attention}(\text{LayerNorm}(x))))$$

MLP and Attention are computed in parallel, with fused kernels it's a lot of performance (~15%) gain. However, it is not common, as it doesn't always work great on the final loss.
$$y = x + \text{MLP}(\text{LayerNorm}(x)) + \text{Attention}(\text{LayerNorm}(x))$$

## Position Embeddings

#### Conventional Embedding Techniques

Sine Embeddings
$$
\text{embedding}(\text{token}, pos) =
\begin{bmatrix}
\\\\\text{token embedding}\\\\\\\end{bmatrix} +
\begin{bmatrix}
\sin(pos/10000^{2/d_{model}})\\
\cos(pos/10000^{2/d_{model}})\\
\sin(pos/10000^{4/d_{model}})\\
\cos(pos/10000^{4/d_{model}})\\
\ldots
\end{bmatrix}
$$

Absolute Embeddings
$$
\text{embedding}(\text{token}, pos) =
\begin{bmatrix}
\\\text{token embedding}\\\\\end{bmatrix} +
\begin{bmatrix}
\\
u_{pos, 0} \\
u_{pos, 1} \\
\ldots \\\\
\end{bmatrix}
$$
Relative Embeddings (instead of being attached at the start, they affect attention computation by adding a distance term $a_{ij}$ to it)
$$\text{Attention}_{i, j} = \frac{x_i W^Q \bigg(x_j W^K + \mathbf{a_{ij}^K}\bigg)^T}{\sqrt{d_z}}$$
#### Building up to Rope

What do we want really? We want absolute position invariance, only sensitivity to relative positions. Sine embeddings leak absolute position on computing dot product, and even relative embeddings don't separate out with a 
$$f(token_x, pos_i) \cdot f(token_y, pos_j) = g(token_x, token_y, pos_j - pos_i)$$

**So this is ROPE:**
Rotate Queries and Keys before attention.
$$\begin{Bmatrix}Q\\ V\end{Bmatrix} =
\begin{bmatrix}
cos(m\theta_1) & -sin(m\theta_1) & 0 & 0 & \ldots & 0 & 0 \\
sin(m\theta_1) & cos(m\theta_1) & 0 & 0 & \ldots & 0 & 0 \\
0 & 0 & cos(m\theta_2) & -sin(m\theta_2) & \ldots & 0 & 0 \\
0 & 0 & sin(m\theta_2) & cos(m\theta_2) & \ldots & 0 & 0 \\
0 & 0 & 0 & 0 & \ldots & cos(m\theta_{d/2}) & -sin(m\theta_{d/2}) \\
0 & 0 & 0 & 0 & \ldots & sin(m\theta_{d/2}) & cos(m\theta_{d/2}) \\
\end{bmatrix} W_{\begin{Bmatrix}Q \\ K\end{Bmatrix}} \cdot x$$

We rotate the inputs to attention by pairing 2 consecutive dimensions and rotating them in 2-D (like complex numbers). The $\theta$s are chosen from some schedule like sine embeddings, e.g. $(pos / 10000^{2i/d})$ to capture low and high frequency information.

## Picking Hyper-parameters out of a Hat

### Feed-Forward (hidden) layer size to Model dimension
Outputs of MLPs is projected down to give the model outputs.

* $d_{\text{FFN}} \approx 4 \cdot d_{\text{model}}$. It will always be larger as we will project down from the FFN output to the final output, a ratio 4 works out well empirically.
* For gated networks, $d_{\text{FFN}} \approx \frac{8}{3} \cdot d_{\text{model}}$ (since GLU variants by convention scale down FFN size by 2/3 to keep learned parameters constant).
* For T5, this ratio was 64, and it still worked well. However in a follow-up release, they went for the more standard 8/3 ratio.

### Num-heads and Size-of-attention head vs. Model dimensions
As we increase the number of heads, we usually keep
$$\frac{\text{number-of-heads} \times \text{head-dimension}}{\text{model-dimension}} \approx 1$$

Argument against this: as we have more heads, our heads will become low rank, which will make the attention head less expressive.

### Model depth vs. width
Deeper networks are smarter or more expressive.
Wider networks are more efficient for the same parameter count.
$$\frac{d_\text{model}}{n_\text{layer}} \approx 128$$
But there are also more concerns:
* Pipeline parallel: helps parallelize deeper networks.
* Tensor parallel: helps more with wider networks. Needs really fast inter-GPU communication.