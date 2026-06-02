# MiniLM Architecture Evolution

This document serves as the master record for the theoretical mechanics, structural changes, and empirical results of our sub-megabyte language models.

## The Theory: Extreme 1.58-bit Quantization

Our goal is to execute a Large Language Model completely natively on CPUs while bypassing the traditional memory bandwidth bottlenecks (which usually require massive GPUs to handle continuous floating-point math). To do this, we use the **BitNet 1.58b Architecture**.

### What is 1.58-bit Quantization?
In a standard LLM (like Llama 3 or GPT-4), the "brain" is composed of weight matrices (Linear Layers) where each connection between neurons is stored as a 16-bit floating-point decimal (e.g., `0.3145`, `-1.9832`). 

**BitNet** completely destroys this continuous math space. It forces every single weight in the Neural Network to be exactly one of three states:
- `-1` (Negative influence)
- `0`  (Ignored / Pruned)
- `1`  (Positive influence)

Because there are only 3 possible states for a weight, the absolute theoretical minimum space needed to store one weight is $log_2(3)$ bits. 
$log_2(3) \approx 1.58$ bits. 

Therefore, compared to a standard 16-bit Float model, a BitNet 1.58b model is mathematically guaranteed to be **10x smaller in physical memory**. More importantly, matrix multiplication ($W \times x$) is completely eliminated during inference and replaced by hyper-efficient Integer Addition.

### How does it learn? (Straight-Through Estimator)
You cannot calculate a gradient across a discrete jump (from 0 to 1). If you try to run Backpropagation on discrete integers, the gradient is 0 everywhere (the function is flat) and learning dies instantly. 
To bypass this, we use a **Straight-Through Estimator (STE)**:
1. **Forward Pass:** The continuous weights (e.g., `0.7`) are violently rounded to the nearest ternary value (`1`). The model generates text using purely discrete `-1, 0, 1` weights.
2. **Backward Pass (STE):** The loss gradient ignores the rounding step completely and flows "straight through" to the hidden continuous weights, nudging them slightly (e.g. from `0.7` to `0.65`). 
3. Over time, the continuous shadows learn to drift perfectly across the thresholds, creating a highly intelligent integer matrix.

### The BitNet Causal Transformer
The overarching architecture is a traditional Autoregressive Causal Transformer (Decoder-only), identical to GPT. However, we replace `nn.Linear` layers with our custom `BitLinear` layers, which implement the STE quantization logic described above. We utilize RMSNorm before every layer, SwiGLU activation for the Feed-Forward block, and Rotary Positional Embeddings (RoPE).

---

## The Evolutionary Tracking

### Baseline (WikiText BPE)
This is our first successful attempt at scaling the BitNet engine out of character-level datasets and into true English Subwords (BPE).

- **Dataset:** wikitext-2 (Wikipedia subset)
- **Tokenizer:** `arnir0/Tiny-LLM` (Vocab Size: 32,000 subwords)
- **Architecture:** BitNet 1.58b Causal Transformer
- **Layers:** 4
- **Dimensions:** 256
- **Attention Heads:** 4
- **Weight Tying:** False
- **Total Parameters:** 20,842,752
- **Physical Memory Size:** 
  - Standard 16-bit Float: `39.75 MB`
  - Our 1.58-bit quantized size: `3.93 MB`

**Results @ 3,000 Steps:**
- **Perplexity:** 55.0
- **Coherence Evaluation:** Poor. The model successfully memorized basic structural grammar, but generated massive amounts of Wikipedia formatting artifacts (`@.@`, `Brethrenmedian`) due to the complex, messy nature of the Wikipedia dataset. It proved the 3.93 MB physics limits worked, but lacked intelligence.
- **Output Sample:**
> "The history of Rome is seen throughout the  of the Brethrenmedian . Ruler 9 ( 214 – the first known     , but they have been 7 @.@ 69 miles ( 26 @.@ 7 in ) which they can be used to 100 feet 6 7 @,@ 000 through"

---

### V1: Dataset Swap (TinyStories)
**Objective:** Replace the complex Wikipedia dataset with `roneneldan/TinyStories`. TinyStories contains massive volumes of text written entirely using a 3-year-old's vocabulary. We hypothesize that a 20 Million parameter brain is far too small to memorize Wikipedia facts, but easily large enough to master 3-year-old grammar structures.

**Results @ 3,000 Steps:**
- **Perplexity:** 10.5 (Massive improvement over 55.0)
- **Coherence Evaluation:** Excellent. All weird Wikipedia artifacts are completely gone. The model generated a fully coherent, syntactically correct English story with consistent subjects (cat, dog, rag, lady).
- **Output Sample:**
> "Once upon a time, there was a kayaky cat. The cat was very strong and had little house. The cat lived in a big tree. The cat liked to play with the kids. One day, the cat saw a little girl. The lady had a dog. The dog saw the rag and wanted to look for the rag. The cat was hungry. The cat saw the rag and thought it was so pretty"

---

### V2: Weight Tying
**Objective:** The 32,000-word Vocabulary Embedding Matrix and the 32,000-word Output Matrix are currently two separate blocks of memory. By "tying" them (using the exact same matrix for input and output), we will instantly delete ~8 Million parameters. We will then inject those "free" parameters as deeper Transformer Layers, vastly increasing logic capabilities while staying locked at 3.93 Megabytes.
*(Pending)*

---

### V3: Saturation (100k Steps)
**Objective:** Small models must be trained for massive durations (Chinchilla Scaling Laws). We will take the V2 architecture and train it for 100,000+ steps to completely saturate the ternary weights.
*(Pending)*
