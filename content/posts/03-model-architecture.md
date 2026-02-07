---
title: "Scaling Up: From Tiny Model to GPT-2 Small"
date: 2026-01-27
draft: false
tags: ["architecture", "transformer", "scaling", "infrastructure", "GPT-2"]
summary: "How I went from 'cute toy model' to '134 million parameters that need an A100 to breathe.'"
weight: 3
---

## Phase 2: Building Infrastructure I'll Actually Need Later

Here's the thing about Phase 1: it was too easy. Everything fit in RAM. Training took 15 minutes. The model was small enough to run on a microwave.

Phase 2 was about building infrastructure for the 12GB monster coming in Phase 3. Because debugging memory issues is way more fun with 250MB than with 12GB. (It's not fun either way, but at least it's faster.)

### The Setup

| Attribute | Value |
|-----------|-------|
| **Dataset** | ~250MB text — still baby size |
| **Tokenization** | Character-level (int16) |
| **Model** | 4-6 layers, 256-384 dim — teenager transformer |
| **Parameters** | ~10-50M — finally respectable |
| **Training** | Google Colab T4/V100 — the free tier hustle |
| **Sequence Length** | 512 — because why not |

### The Numbers

| Metric | Value |
|--------|-------|
| **Training Time** | ~2-4 hours (coffee break compatible) |
| **Epochs** | 10-15 |
| **Speed** | ~0.3-0.5s/step |

### What The Phase 2 Model Thought "Intelligence" Was

I asked it some questions. It tried its best.

| Prompt | Model Output | My Reaction |
|--------|--------------|-------------|
| `Where is the capital of USA?` | `The capital of USA is USA.` | Technically... no. |
| `What is 2 + 2?` | `2 + 2 is a number.` | I mean, you're not wrong. |
| `Who is the president?` | `The president is the president of the United States.` | Circular logic achievement unlocked. |
| `Tell me a joke.` | `A man walks into a bar and the bar and the bar and` | It's avant-garde comedy. |

The model had mastered the art of saying nothing with great confidence.

### Why Memory-Mapped Files?

"But the 250MB dataset fits in RAM!" you say.

Yes. But 12GB doesn't. And I'd rather debug memory-mapping code with a small dataset than stare at OOM errors for hours with a big one.

```
250MB text → ~500MB as int16 → fits in Colab's 12GB RAM ✓
12GB text → ~11GB as int32 → LOL no ✗
```

Future me thanked past me for this one.

### What I Actually Built

**1. Pre-tokenization Pipeline**

Because processing text every epoch is for people who enjoy watching progress bars.

**2. Memory-Mapped Loading**

`numpy.memmap` is basically magic. Your 11GB file pretends to be a numpy array, but it's actually reading from disk. The OS handles caching. It's like having infinite RAM with extra steps.

**3. Colab Survival Kit**

- Checkpoints to Google Drive (because Colab WILL disconnect)
- Session timeout handling (because Colab WILL disconnect)
- Did I mention Colab disconnects?

---

## Phase 3: The Real Deal 💪

Time to stop playing around. GPT-2 Small. 134 million parameters. On a dataset that makes my laptop cry.

### The Configuration

| Attribute | Value |
|-----------|-------|
| **Dataset** | 12GB of ChatGPT-style conversations |
| **Tokenization** | BPE 32K vocab — like a grown-up |
| **Total Tokens** | 2.8 billion — with a B |
| **Model** | GPT-2 Small (12 layers, 768 dim, 12 heads) |
| **Parameters** | 134 million |
| **Training** | Colab A100 (40GB) — the big guns |
| **Batch Size** | 64 |
| **Learning Rate** | 3e-4 |

### The Glow Up

| Parameter | Phase 1 (Toy) | Phase 2 (Meh) | Phase 3 (Finally) |
|-----------|---------------|---------------|-------------------|
| Layers | 2 | 4-6 | 12 |
| Heads | 4 | 4-8 | 12 |
| Embed Dim | 128 | 256-384 | 768 |
| Context | 256 | 512 | 512 |
| Parameters | ~400K | ~10-50M | 134M |

Look at that growth. They grow up so fast.

### Runtime Reality Check

| Metric | Value |
|--------|-------|
| **Steps/Epoch** | ~79,507 (yes, really) |
| **Tokens/Step** | 32,768 (64 batches × 512 tokens) |
| **Speed (after optimization)** | ~0.225s/step |
| **Time/Epoch** | ~5 hours |
| **Total Epochs** | 10 |
| **Total Training Time** | ~50 hours (optimized) |
| **Colab Compute Units** | ~400 (at 8 units/hr for A100) |

50 hours of GPU time. On the "cheap" tier. ML is an expensive hobby.

---

## The Files That Made This Possible

All code available at [github.com/GPUburnout/gpt2-from-scratch](https://github.com/GPUburnout/gpt2-from-scratch)

| File | What It Does |
|------|--------------|
| [`model.py`](https://github.com/GPUburnout/gpt2-from-scratch/blob/main/model.py) | The transformer itself (fully parameterized, because hardcoding is for cowards) |
| [`tokenizer.py`](https://github.com/GPUburnout/gpt2-from-scratch/blob/main/tokenizer.py) | Character-level tokenizer (Phase 1-2) |
| [`tokenizer_bpe.py`](https://github.com/GPUburnout/gpt2-from-scratch/blob/main/tokenizer_bpe.py) | BPE tokenizer (Phase 3, the good stuff) |
| [`tokenize_local_bpe.py`](https://github.com/GPUburnout/gpt2-from-scratch/blob/main/tokenize_local_bpe.py) | Converts text to binary with BPE |
| [`train_colab_mmap.py`](https://github.com/GPUburnout/gpt2-from-scratch/blob/main/train_colab_mmap.py) | The training script that keeps me up at night |
| [`generate.py`](https://github.com/GPUburnout/gpt2-from-scratch/blob/main/generate.py) | Makes the model spit out text |

---

## Lessons From Scaling 335x

(From 400K params to 134M params. I did the math so you don't have to.)

1. **Build for scale before you need it.** The infrastructure I built for 250MB worked perfectly for 12GB. Zero changes. That's the dream.

2. **Pre-tokenize everything.** Processing text during training is like doing your taxes during a job interview. Just... don't.

3. **Parameterize obsessively.** Every magic number should be a config value. Future you will forget why you used 768 instead of 512.

4. **Use `.get()` with defaults.** Config files evolve. Your code shouldn't crash because you added a new field.

5. **Colab will disconnect.** Save checkpoints like your career depends on it. Because your training run does.
