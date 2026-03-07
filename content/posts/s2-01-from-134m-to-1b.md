---
title: "From 134M to 1B: Building GPUburnout-1B From Scratch"
date: 2026-02-27
draft: false
tags: ["GPUburnout-1B", "1B", "scaling", "architecture", "season-2"]
description: "GPT-2 was the warmup. Now I'm building a 1 billion parameter model from scratch, training it on 12 billion tokens, and documenting every step — including what it costs."
season: 2
chapter: 1
---

## Season 1 is over. Time to scale up.

Six weeks ago, I started this blog with a simple question: what actually happens inside a language model? The answer turned into a six-post series where I built GPT-2 from scratch — 134 million parameters, 2.8 billion tokens, and a Colab session that crashed more often than it didn't.

I learned a lot. Not just about transformers and tokenizers, but about the thousand small decisions that determine whether your training run produces coherent English or expensive gibberish. I took training time from 90 minutes down to 21 minutes. I watched a random pile of floating-point numbers slowly learn that Paris is a city and "the" comes before nouns.

But here's the thing — 134M parameters is tiny by modern standards. The architecture is from 2019. And while I'm proud of what I built, I kept looking at the loss curves thinking: *what happens if I go bigger?*

So I went bigger. A lot bigger.

## Introducing GPUburnout-1B

GPUburnout-1B is a 1.04 billion parameter language model that I built and trained from scratch. Not a download from HuggingFace. Not a fine-tune of someone else's checkpoint. Random initialization, my own dataset, my own training code, my own mistakes.

The architecture is inspired by Meta's LLaMA design — the same family of techniques that powers most modern open-source models — but this isn't a LLaMA model. I didn't use Meta's weights, their tokenizer, or their data. I took the architectural ideas (they're public, and they're good ideas) and built my own implementation. Think of it like building your own car using the same engineering principles as a Tesla, but fabricating every part yourself in your garage.

Here's what changed from Season 1:

| | GPUburnout-134M (Season 1) | GPUburnout-1B (Season 2) |
|---|---|---|
| **Parameters** | 134M | 1.04B |
| **Architecture** | GPT-2 (2019) | LLaMA-style (2023) |
| **Context length** | 1024 | 2048 |
| **Training tokens** | 2.8B | 11.8B |
| **Attention** | Learned positional embeddings | RoPE (Rotary Position Embeddings) |
| **Normalization** | LayerNorm | RMSNorm |
| **Activation** | GELU | SiLU + GLU (SwiGLU) |
| **Attention mechanism** | Multi-Head | Grouped Query (GQA) |
| **Tokenizer** | GPT-2 BPE (50K vocab) | Custom BPE (32K vocab) |
| **Dataset** | Single source | 3-source mix |
| **Hardware** | Google Colab | RunPod A100 80GB |

This isn't just "same thing but bigger." Almost every component is different — modern techniques like Rotary Position Embeddings, Grouped Query Attention, and SwiGLU activations that didn't exist when GPT-2 was designed. If Season 1 was learning to drive, Season 2 is building the engine.

## Why these architectural choices?

**Rotary Position Embeddings (RoPE)** replace learned position embeddings. Instead of learning a separate embedding for "position 1," "position 2," etc., RoPE encodes position by rotating the query and key vectors. The model can generalize to sequence lengths it's never seen during training — GPT-2's learned embeddings can't do that.

**Grouped Query Attention (GQA)** is a compromise between standard multi-head attention and multi-query attention. GPUburnout-1B uses 32 attention heads but only 8 key-value heads — every 4 query heads share one key-value pair. This cuts the KV cache by 4x during inference with minimal quality loss. At 1B parameters, the memory savings aren't critical. At the scale this architecture was designed for (70B+), they're essential. I used GQA anyway because I wanted to implement it correctly and understand it from the inside.

**SwiGLU** replaces GELU in the feed-forward layers. It's a gated activation — the network learns to modulate its own activations, adding a multiplicative interaction that helps the model learn more complex functions. The tradeoff is that SwiGLU has 3 weight matrices per feed-forward block instead of 2, so the hidden dimension changes. The standard formula `d_ff = 4 * d_model` becomes `d_ff = (8/3) * d_model`, rounded to the nearest multiple of 256 for hardware efficiency.

**RMSNorm** replaces LayerNorm. It drops the mean-centering step and only normalizes by the root mean square. One less operation per layer, slightly faster, and empirically works just as well for language modeling.

**Flash Attention** isn't an architectural choice — it's the same attention computation, just implemented to be memory-efficient by avoiding materializing the full attention matrix. But it's why a 1B model fits comfortably on a single 80GB GPU.

## The dataset: three sources, one tokenizer

One of the biggest lessons from Season 1: data quality matters more than data quantity. So instead of grabbing the largest text dump I could find, I built a curated three-source mix:

- **FineWeb-Edu** (87%) — High-quality, educationally-scored web text. This is the backbone. Clean, diverse content filtered for educational value, which means less noise and more of the structured knowledge that teaches a model how language actually works.

- **FineMath** (10%) — Mathematical content. Because I want this model to at least attempt arithmetic, unlike GPUburnout-134M, which thought 2+2 was "the."

- **Python-Edu** (3%) — Quality-filtered Python code. Just enough to give the model a foundation in programming syntax without drowning out general knowledge.

Total: 30.6 billion tokens available across 282 shards. I ended up training on 11.8 billion tokens before stopping — more on why in the next post.

The tokenizer is a custom BPE with a 32,005 token vocabulary, trained on the same data mix. Smaller than GPT-2's 50K vocab, which means slightly longer sequences for the same text, but better compression on the specific domains I'm training on.

## The infrastructure: leaving Colab behind

Season 1 ran entirely on Google Colab. That meant session timeouts, reconnection battles, and the constant anxiety of losing your runtime mid-training. For a 134M model that trains in 21 minutes, that's annoying but survivable.

A 1B model doesn't train in 21 minutes.

So I moved to RunPod — a cloud GPU platform where you rent dedicated hardware by the hour. My primary setup:

- **GPU:** NVIDIA A100 80GB (SXM variant)
- **Cost:** $1.45–$1.49/hr depending on availability
- **Storage:** 150GB network volume for persistent data

The workflow changed too. With SSH access to a persistent machine, I could tail logs in real time, edit training configs on the fly, and relaunch runs without the Colab dance of "reconnect, remount Drive, re-run all cells, pray." When something crashed at 2 AM, the checkpoint was still there in the morning. When I wanted to test a different learning rate schedule, it was a config change and a restart — not a 15-minute setup ritual.

It's a small thing, but having a stable environment where you can iterate quickly makes a surprising difference in how many ideas you actually test.

## The training plan: four phases

Rather than committing hundreds of dollars to a single uninterrupted run, I broke training into phases. Each phase has a clear goal, and each checkpoint gives me an off-ramp to evaluate before spending more.

**Phase 1 — Smoke Test (200 steps):** Does it crash? Does loss drop from random initialization (~10.4)? Is the data pipeline working? Budget: ~$5.

**Phase 2 — Proof of Life (10,000 steps):** Is the model actually learning language? Can it form coherent sentences? What's the throughput? Budget: ~$25.

**Phase 3 — The Long Run (60,000 steps):** Push through the steep part of the loss curve. Save milestones at 30K and 60K for comparison. Budget: ~$100.

**Phase 4 — Diminishing Returns (90,000 steps):** Squeeze out the remaining easy gains, then evaluate whether to keep going. Budget: ~$50.

Total planned budget: under $200 for a from-scratch 1B model. Whether that's a lot or a little depends on your perspective. Compared to TinyLlama (16 A100s for 90 days) or Pythia (institutional compute grants), it's pocket change. Compared to a nice dinner, it's... well, it's a lot of dinners.

## What's coming next

The training is done. The checkpoints are saved. The loss curve tells an interesting story — one that involves beating my own predictions, discovering that cheaper GPUs aren't always cheaper, and making the call to stop training before the original plan said to.

**Next post: [The $175 Experiment — Training GPUburnout-1B on a Single GPU.](/posts/s2-02-the-175-dollar-experiment/)**

I'll walk through every phase, every dollar, and every surprise — including the text samples that show a model going from random noise to citing scientific journals.

---

*This is Post 7 of an ongoing series documenting my journey building language models from scratch. Season 1 covered [GPUburnout-134M](/tags/gpt-2/) (GPT-2 architecture). Season 2 covers GPUburnout-1B.*

*Follow along: [GitHub](https://github.com/GPUburnout) · [RSS](/index.xml)*
