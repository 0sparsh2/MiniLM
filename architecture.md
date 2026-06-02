# MiniLM Architecture Evolution

This document tracks the iterative structural changes and performance metrics of our sub-megabyte language models.

## Baseline (WikiText BPE)
- **Dataset:** wikitext-2 (Wikipedia)
- **Tokenizer:** `arnir0/Tiny-LLM` (Vocab: 32,000)
- **Architecture:** BitNet 1.58b Causal Transformer
- **Layers:** 4
- **Dimensions:** 256
- **Heads:** 4
- **Weight Tying:** False
- **Total Parameters:** 20,842,752
- **Physical Size:** 3.93 MB (1.58-bit quantized)

### Results @ 3,000 Steps
- **Perplexity:** 55.0
- **Coherence Evaluation:** Poor. Produced Wikipedia formatting artifacts (`@.@`, `Brethrenmedian`) and hallucinatory phrasing.
- **Output Sample:**
> "The history of Rome is seen throughout the  of the Brethrenmedian . Ruler 9 ( 214 – the first known     , but they have been 7 @.@ 69 miles ( 26 @.@ 7 in ) which they can be used to 100 feet 6 7 @,@ 000 through"

---

## V1: Dataset Swap (TinyStories)
*(Pending)*

---

## V2: Weight Tying
*(Pending)*

---

## V3: Saturation (100k Steps)
*(Pending)*
