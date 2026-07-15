# Lesson 1: Introduction to GPT, Transformers, and the nanoGPT Project

### Learning Objectives
* Understand the history of sequence modeling and the limitations of RNNs/LSTMs.
* Comprehend the self-attention mechanism's advantages in routing information and parallelization.
* Differentiate between Encoder-Decoder and Decoder-Only Transformer architectures.
* Map the overall directory structure of `nanoGPT` and trace its global data flow.

---

## 1. The Pre-Transformer Era: The Reign & Ruin of RNNs

Before the Transformer, processing sequential data (like text or time-series) relied on architectures that processed items step-by-step. The most common of these were **Recurrent Neural Networks (RNNs)** and **Long Short-Term Memory (LSTM)** networks.

### What is an RNN?
An RNN maintains a dynamic internal **hidden state** $h_t$ that represents its "memory" of all tokens processed up to time step $t$.

```
RNN Sequential Processing (O(T) steps):
[Token 1] ---> RNN ---> [Hidden State 1]
                          |
[Token 2] ---> RNN <------+ ---> [Hidden State 2]
                                   |
[Token 3] ---> RNN <---------------+ ---> [Hidden State 3]
```

### Mathematical Formulation
At each step $t$, the network receives an input vector $x_t \in \mathbb{R}^d$ and the previous hidden state $h_{t-1} \in \mathbb{R}^h$. It updates its state and computes an output $y_t$:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

Where:
* $W_{hh} \in \mathbb{R}^{h \times h}$ is the recurrent weight matrix.
* $W_{xh} \in \mathbb{R}^{h \times d}$ is the input projection matrix.
* $b_h, b_y$ are bias vectors.
* $\tanh$ is the non-linear activation.

---

### The Three Fatal Bottlenecks of RNNs

#### 1. The Sequential Bottleneck (O(T) Steps)
Because $h_t$ is a mathematical function of $h_{t-1}$, you cannot compute the hidden state at step $t$ until the step $t-1$ has finished.
* **Hardware Mismatch**: Modern graphics cards (GPUs) perform computation via massive parallelism (doing thousands of matrix multiplications at once). RNNs force GPUs to work sequentially, which severely slows down training speeds.

#### 2. Vanishing and Exploding Gradients
When training on a long sequence, backpropagating gradients from the loss at step $T$ back to step $1$ requires repeatedly multiplying by the transpose of the recurrent weight matrix, $W_{hh}^T$.
* If the eigenvalues of $W_{hh}$ are $< 1$, the gradient vanishes exponentially:
  $$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_T} \frac{\partial h_T}{\partial h_{T-1}} \dots \frac{\partial h_2}{\partial h_1} \approx 0$$
* If the eigenvalues are $> 1$, the gradient explodes to infinity, destabilizing training.
* **Result**: RNNs are effectively "blind" to long-range dependencies in text.

#### 3. Information Bottleneck
An RNN must compress all historical context (no matter how long) into a single, fixed-size vector $h_t$. This leads to loss of information as sequences grow.

---

## 2. The Transformer Revolution

In 2017, Vaswani et al. introduced the **Transformer** in the paper *"Attention Is All You Need"*. It completely discarded recurrence in favor of **Self-Attention**.

```
Transformer Parallel Processing (O(1) sequential steps):
[Token 1] ---\
[Token 2] ----+---> Self-Attention Layer ---> [Context Vectors 1, 2, 3] (Computed in parallel!)
[Token 3] ---/
```

### How Self-Attention Solves the Bottlenecks:
1. **Parallel Training**: The self-attention operation processes the entire sequence in a single forward pass. Every token is compared with all other tokens simultaneously via matrix operations, utilizing GPU hardware efficiency.
2. **$O(1)$ Path Length**: Any token can directly attend to any other token across the sequence, regardless of distance. The gradient does not have to travel through time steps; it flows directly back from the attending token to the attended token.
3. **No Dynamic Summarization**: Instead of compressing all context into a single vector, every token maintains its own representation, updating itself by gathering information from relevant neighbors dynamically.

---

## 3. Decoder-Only GPT Architecture

While the original Transformer was designed for seq-to-seq translation (Encoder-Decoder), GPT uses a **Decoder-Only** architecture.

```
       Original Transformer                     GPT (Decoder-Only)
      
       +---------------+                     +--------------------+
       |  Source Text  |                     |  Prompt + Context  |
       +---------------+                     +--------------------+
               |                                       |
        [Encoder Blocks]                               v
               |                             [Decoder Blocks (Causal)]
               v                                       |
        [Decoder Blocks]                               v
               |                             +--------------------+
               v                             |  Next Token Logits |
       +---------------+                     +--------------------+
       |  Translation  |
       +---------------+
```

### Unidirectional (Causal) Masking
Standard self-attention allows tokens to look at the entire sequence (bidirectional). However, for autoregressive generation (predicting the next token), the model must not see future tokens.
GPT uses a **Causal Mask** that zeros out attention weights for future positions, ensuring the model only utilizes past context.

---

## 4. nanoGPT Project Walkthrough

The `nanoGPT` repository is a clean, minimal implementation of a GPT-2 style model, designed for readability and optimization.

### Repository Files
* **[model.py](file:///c:/Users/admin/Documents/Projects/nanoGPT/model.py)**: Defines the model architecture.
* **[train.py](file:///c:/Users/admin/Documents/Projects/nanoGPT/train.py)**: Training loop with support for learning rate scheduling, mixed precision, and distributed processing.
* **[sample.py](file:///c:/Users/admin/Documents/Projects/nanoGPT/sample.py)**: Text generation script.
* **[configurator.py](file:///c:/Users/admin/Documents/Projects/nanoGPT/configurator.py)**: Parses command-line arguments and configuration overrides.
* **[data/](file:///c:/Users/admin/Documents/Projects/nanoGPT/data)**: Scripts to download and preprocess datasets (like Shakespeare or OpenWebText) into raw binary token buffers.

---

## 5. Exercises

1. **Path Length Calculation**: Assume a sequence of length $T=1024$. Calculate the maximum number of layer-to-layer steps needed for information to pass from token index $1$ to token index $1024$ in:
   * A standard unidirectional RNN.
   * A Transformer layer.
2. **Computational Complexity**: Write out the sequence length dependency ($T$) for the cost of:
   * Forward pass of an RNN layer over $T$ steps.
   * Forward pass of a Self-Attention layer over $T$ steps (assuming embedding dimension is $d$).

---

## 6. Quiz

1. **Why did the Transformer discard recurrence?**
   * A) Because recurrence required too much memory.
   * B) To allow training sequence operations to run in parallel on GPUs.
   * C) Because sequential processing was mathematically incorrect.
2. **What problem does causal masking solve?**
   * A) It prevents the model from generating toxic text.
   * B) It ensures that tokens cannot look ahead to target tokens during training.
   * C) It compresses the sequence to save GPU memory.

---

## 7. Interview Questions (Answer in the main chat)

To complete this lesson, please attempt to answer these questions from first principles:

1. **Easy**: Explain the difference between an **Encoder-Decoder** Transformer (like T5) and a **Decoder-Only** Transformer (like GPT-2) in terms of target tasks.
2. **Medium**: Mathematically, why do standard RNNs struggle to learn associations between tokens that are hundreds of tokens apart? (Focus on the backward pass).
3. **Hard / FAANG-level**: If sequence length is $T$ and embedding dimension is $d$, what is the computational time complexity of the sequence processing in:
   - A single RNN Layer?
   - A single Self-Attention Layer?
   Discuss how this affects training speeds for short vs. very long sequences.
4. **Research-level**: State-Space Models (SSMs) like Mamba or linear attention models (like RWKV) claim to achieve $O(T)$ computational complexity during training and inference while preserving long-context memory. From a hardware perspective, why did Transformers succeed in the industry despite their $O(T^2)$ attention bottleneck, and what challenges do these new architectures face in replacing Transformers?
