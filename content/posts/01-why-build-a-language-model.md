---
title: "Why I Decided to Build a Language Model from Scratch"
date: 2026-01-15
draft: false
tags: ["intro", "motivation", "transformer", "phase-1"]
summary: "Because apparently using someone else's model was too easy. Here's how I tortured myself by training GPT from scratch."
weight: 1
---

## The Goal

Build a language model from scratch. Why? Because sleep is for people who don't debug tensor shapes at 2 AM. 🌙 The tutorials made it look so easy.

Spoiler: much much harder than I expected. But also way more educational than reading another tutorial that glosses over the painful bits.

**What I actually achieved:** GPT-2 Small (134M params) trained on 12GB of data, with a 7x speedup after I stopped doing stupid things. More on that later.

**All the code:** [github.com/GPUburnout/gpt2-from-scratch](https://github.com/GPUburnout/gpt2-from-scratch)

### The Damage Report

| Phase | Model Size | Dataset | Training Time | My Sanity |
|-------|------------|---------|---------------|-----------|
| Phase 1 | 400K params | 1MB | ~15 min | Intact |
| Phase 2 | 10-50M params | 250MB | ~3 hrs | Slightly frayed |
| Phase 3 | 134M params | 12GB | ~140 hrs* | What sanity? |

*Phase 3 breakdown: ~105 hrs of "why is this so slow" + ~35 hrs after I learned to optimize

---

## Phase 1: Baby's First Transformer

Started small. Like, embarrassingly small. 400K parameters on Shakespeare text. The kind of model that fits in a potato.

### The Setup

| What | The Reality |
|------|-------------|
| **Dataset** | Shakespeare (~1MB) — fancy way of saying "small enough to not break my laptop" |
| **Tokenization** | Character-level (~200 vocab) — because words are hard |
| **Model** | Tiny (2 layers, 128 dim, 4 heads) — basically a neural network that identifies as a transformer |
| **Parameters** | ~400K — my phone calculator has more weights |
| **Training** | Local machine — aka "please don't crash" |

### How Long Did This Take?

| Metric | Value |
|--------|-------|
| **Training Time** | ~15 minutes (I've waited longer for coffee) |
| **Epochs** | 10-20 |
| **Hardware** | Whatever wasn't on fire |

### Why Bother?

Look, I could've just downloaded GPT-2 and called it a day. But then I wouldn't have learned:

- That attention masks will silently destroy your gradients if you get them wrong
- Positional encodings are annoyingly important
- "Autoregressive" isn't just a fancy word — mess it up and your model sees the future

### The Pain Points

**1. Transformers Are Deceptively Complex**

Implementing multi-head attention from scratch sounds cool until you're staring at tensor shapes at 2 AM wondering why `(batch, seq, heads, dim)` doesn't match `(batch, heads, seq, dim)`.

**2. Training Loops Are Boring Until They Break**

Setting up train/val splits, tracking loss, picking learning rates... it's all straightforward until your loss goes to NaN and you have no idea why.

### The Loss Curve (It Actually Worked!)

```
Loss
4.5 |*
    |  *
4.0 |    *
    |      *
3.5 |        *
    |          *  *
3.0 |              *  *
    |                    *  *  *
2.5 |                              *  *  *  *
    +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20
                              Epoch
```

### What The Model Actually Produced

| Epoch | Loss | Sample Output |
|-------|------|---------------|
| 1 | ~4.5 | `ttttttttttttttttttttttttttttt` |
| 5 | ~3.5 | `the the the the the the the` |
| 10 | ~2.8 | `ROMEO: What art thou dost the` |
| 15 | ~2.5 | `ROMEO: What say you to my love?` |
| 20 | ~2.4 | `ROMEO: I do beseech thee, hear me speak.` |

From keyboard spam to almost-Shakespeare. Progress.

### What I Actually Learned

- **Start embarrassingly small.** My ego wanted 7B parameters. My debugging skills needed 400K.
- Character-level tokenization is like counting grains of sand. Works, but there's a better way.
- Even a tiny model can learn. That felt like magic.

---

## What's Next (aka How I Made Things Harder)

Coming up in this series:
- **Data Prep** — Where I download 12GB and regret my life choices
- **Architecture** — Scaling from "toy model" to "actual GPT-2"
- **Training** — 10 errors that made me question my career
- **Results** — Did it work? (Mostly. Kind of. Define "work.")
