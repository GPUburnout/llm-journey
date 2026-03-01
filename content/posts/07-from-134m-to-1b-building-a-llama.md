---
title: "From 134M to 1B: Building a Llama From Scratch"
date: 2026-03-01
draft: false
tags: ["llama", "1B", "scaling", "architecture", "season-2"]
description: "GPT-2 was the warmup. Now I'm building a 1 billion parameter Llama model, training it on 30B tokens, and documenting every step — including what it costs."
---

## Season 1 is over. Time to scale up.

Five weeks ago, I started this blog with a confession: I wanted to build a language model from scratch because using someone else's model was too easy.

Six posts and one mass-produced GPT-2 later, I had a working 134M parameter model trained on 2.8B tokens. It could write coherent English, attempt (bad) Python, and occasionally say something that made me laugh. Total cost: about $15 in Colab compute.

But 134M parameters is tiny. GPT-2 small was a 2019 model. The world has moved on.

So I'm doing what any reasonable person would do: I'm building something 7.5x bigger.

## The plan: 1 billion parameters, from scratch

This time I'm building a Llama-architecture model. 1 billion parameters. Trained on 30 billion tokens of FineWeb-Edu. The full stack: RoPE embeddings, RMSNorm, SwiGLU activations, grouped-query attention — everything Meta's Llama uses, implemented from scratch.

The training happens in phases on RunPod (A100 80GB):
- Phase 1: Smoke test (200 steps) — does it crash?
- Phase 2: Proof of life (10K steps) — is the loss dropping?
- Phase 3: First benchmarks (50K steps) — real eval scores
- Phase 4: The full run (90K steps) — final results

Total budget: roughly $250-350 for pretraining plus a few dollars for SFT.

## What I expect (and what would surprise me)

At 30B tokens, this model will have seen roughly 30 tokens per parameter — about 1.5x the Chinchilla-optimal ratio. That sounds great until you realize that Meta's Llama 3.2 1B was trained on 9 trillion tokens. My model will have seen 300x less data.

So what should 30B tokens buy? Coherent text generation, basic Python patterns, elementary math, and surface-level general knowledge. What it won't do: compete with any model you can download from HuggingFace. And that's fine. The point is understanding what happens inside these systems at every level.

## Why document this publicly?

Because when I started Season 1, I couldn't find a single blog that showed the real experience of training a model from scratch. Not a tutorial with toy data. Not a research paper with unlimited compute. A real project, with real bugs, real costs, and real compromises.

Every post in this series will include loss curves, benchmark scores, exact costs, text samples at each checkpoint, and mistakes.

If you followed Season 1, you know the deal. If you're new here, welcome — grab a coffee, your GPU is going to be busy for a while.

## What's next

The code is written. The data is tokenized. Training is already underway on RunPod.

Next post: Phase 1 — Smoke Test. Did it crash? Let's find out.

---

Follow along: GitHub · RSS
