---
title: "The Results Are In (And My Wallet Is Empty)"
date: 2026-02-06
draft: false
tags: ["results", "loss-curves", "lessons", "evaluation"]
summary: "Final loss curves, the damage to my compute budget, and 22 lessons I paid dearly to learn."
weight: 5
---

So. After ~130 hours of GPU time, countless Colab disconnects, and one existential crisis about whether I should've just used a pretrained model... here's what I got.

---

## Training Progress (The Numbers Don't Lie)

### The Loss Curve

```
Loss
4.5 |*
    |
4.0 |    *
    |
3.5 |        *
    |            *
3.0 |                *  *  *  *  *  *  *   <- Epoch 11: Done!
    |
2.5 |                                      <- Would be nice
    |
2.0 |                                      <- Maybe in my dreams
    +--+--+--+--+--+--+--+--+--+--+--+--+
    1  2  3  4  5  6  7  8  9  10 11
                  Epoch
```

Loss plateaued around 3.0 — that's as good as it gets for 134M params on this dataset.

### The Epoch-by-Epoch Damage Report

| Epoch | Train Loss | Speed | Runtime | What Was Happening |
|-------|------------|-------|---------|-------------------|
| 1 | ~4.5 | 1.6s/step | ~35 hrs | Model: "What are words?" |
| 2 | ~3.8 | 1.6s/step | ~35 hrs | Model: "Oh, patterns exist!" |
| 3 | ~3.4 | 1.6s/step | ~35 hrs | Me: "Why is this so slow?" |
| 4 | ~3.24 | 0.225s/step | ~5 hrs | Optimizations kick in |
| 5 | 3.093 | 0.225s/step | ~5 hrs | Finally making progress |
| 6 | ~3.05 | 0.1s/step | ~2.5 hrs | Flash Attention enabled! |
| 7 | ~3.03 | 0.1s/step | ~2.5 hrs | Loss go down |
| 8 | 3.014 | 0.1s/step | ~2.5 hrs | Loss go down more |
| 9 | ~3.01 | 0.1s/step | ~2.5 hrs | Diminishing returns |
| 10 | ~3.00 | 0.1s/step | ~2.5 hrs | Plateau reached |
| 11 | ~2.99 | 0.1s/step | ~2.5 hrs | Done! ✓ |

Notice the jump from epoch 3 to 4. That's the 16x optimization kicking in. I literally wasted 105 hours being dumb.

---

## What The Model Actually Says Now

Here's the real test. Same prompts, different epochs. Watch it get less stupid:

### Prompt: "What is the capital of France?"

| Epoch | Output | Verdict |
|-------|--------|---------|
| 1 | `What is the capital of France? France France France France` | Brain damage |
| 4 | `What is the capital of France? The capital of France is a city.` | Technically true? |
| 11 | `What is the capital of France? The capital of France is Paris.` | We did it 🎉 |

### Prompt: "Explain machine learning in simple terms."

| Epoch | Output |
|-------|--------|
| 1 | `Explain machine learning in simple terms. terms terms terms` |
| 4 | `Explain machine learning in simple terms. Machine learning is a type of learning.` |
| 11 | `Explain machine learning in simple terms. In AI, agents are trained to provide input and output that can help humans generate text with ease. Machine learning models can perform tasks such as sentiment analysis, language translation, and image generation.` |

### Prompt: "Write a poem about coffee."

| Epoch | Output |
|-------|--------|
| 1 | `Write a poem about coffee. coffee coffee coffee coffee` |
| 4 | `Write a poem about coffee. Coffee is a drink. The drink is good.` |
| 11 | `Write a poem about coffee. In my city, where streets and rivers meet, Where the heart beats with rhythm and sweet flow, I am filled to death with hope, bright stars. And as I wander through this night's light...` |

Not Shakespeare, and sometimes the model drifts into training data (the joys of 134M params). But it's coherent text, not keyboard spam. Progress.

---

## Try It Yourself

Don't take my word for it. Here's the actual model — pick a checkpoint and see how it generates:

<iframe
	src="https://gpuburnout-gpt2-from-scratch.hf.space"
	frameborder="0"
	width="100%"
	height="700"
></iframe>

*Three models to compare: Tiny Shakespeare (3.2M), Medium (3.3M), and GPT-2 Small (134M, checkpoint 7 = epoch 11). The difference is... noticeable.*

---

## The Two Sessions: A Tale of Suffering and Enlightenment

**Session 1 (Epochs 1-3):** The Dark Ages
- ~105 hours of my life I'm not getting back
- Memory-mapped like an amateur
- 1.6s per step on an A100. AN A100.
- The GPU was probably playing solitaire

**Session 2 (Epochs 4-5):** Getting Smarter
- RAM preload + vectorization + torch.compile + AMP
- 0.225s/step — 7x faster
- ~10 hours for 2 epochs
- GPU starting to earn its keep

**Session 3 (Epochs 6-11):** Full Speed Ahead
- Added Flash Attention (PyTorch 2.0+)
- 0.1s/step — **16x faster than baseline**
- ~15 hours for 6 epochs
- GPU finally working at full power

*Want the full technical breakdown of each optimization? See [Training Optimizations Deep Dive](/posts/06-training-optimizations-deep-dive).*

---

## The Financial Damage 💸

| Item | Compute Units |
|------|---------------|
| A100 burn rate | ~8 units/hour |
| Epochs 1-3 (suffering @ 1.6s/step) | ~840 units |
| Epochs 4-5 (better @ 0.225s/step) | ~80 units |
| Epochs 6-11 (optimized @ 0.1s/step) | ~120 units |
| **Total** | **~1,040 units** |

Let me do some fun math:

- **If I had optimized from the start:** ~220 units (all 11 epochs @ 0.1s/step)
- **What I actually used:** ~1,040 units
- **Units wasted on stupidity:** ~820 units (~79% of total)

Cool. Cool cool cool.

---

## 22 Lessons I Paid To Learn

### Architecture & Design (Don't Be Clever)

1. **Start embarrassingly small.** 400K params first. Your ego can wait.
2. **Parameterize everything.** Magic numbers are technical debt with compound interest.
3. **Use `.get()` with defaults.** Config files change. Your code shouldn't explode.

### Data Pipeline (Prepare for the Worst)

4. **Pre-tokenize to binary.** Process text once. Not every epoch. Once.
5. **BPE > Character tokenization.** This isn't even debatable.
6. **Store metadata with data.** Token counts, vocab size, dtype. Trust nothing.

### Dataset Creation (The Boring Stuff)

7. **Quality > Quantity.** 10GB clean beats 20GB garbage.
8. **Stream large files.** Loading 12GB into RAM is for people who hate their computers.
9. **Compress before uploading.** 4GB uploads faster than 11GB. Revolutionary.
10. **Use Rust-based tokenizers.** Python tokenizers are cute. Rust tokenizers are fast.
11. **Save intermediate files.** Re-running 2-hour jobs is a special kind of pain.

### Training Optimization (Read This First)

12. **Profile before optimizing.** Find the actual bottleneck before changing random things.
13. **RAM beats mmap for random access.** If it fits, preload it.
14. **Vectorize or suffer.** Python loops in hot paths are crimes against GPUs.
15. **torch.compile + AMP + Flash Attention = mandatory.** It's 2026. There's no excuse.

*Deep dive into each optimization with code examples: [Training Optimizations Deep Dive](/posts/06-training-optimizations-deep-dive)*

### Colab Survival Guide

16. **Checkpoints go to Drive.** Colab WILL disconnect. Plan accordingly.
17. **OOM = restart runtime.** `empty_cache()` is a polite suggestion.
18. **Budget compute units.** A100 eats 8 units/hour. Track it.

### Debugging (Check These First)

19. **Check dtypes.** Then check again. Then `.long()` anyway.
20. **Configs must match exactly.** SEQ_LEN, vocab_size, embed_dim — one mismatch = cryptic error.
21. **Verify files exist.** Before writing code that loads them. Novel concept.

### Documentation (Future You Is Dumb)

22. **Document as you go.** That "obvious" fix you just did? You'll forget it by next week. Write it down.

---

## What's Next

1. ~~**Finish epoch 10**~~ — Done! Went to epoch 11 for good measure.
2. ~~**Test generation quality**~~ — It produces coherent text! See above.
3. **Consider GPT-2 Medium** — 355M params. Because I'm a glutton for punishment.
4. **Fine-tuning experiments** — Make it useful for something specific.
5. ~~**Release on GitHub/HuggingFace**~~ — Done! [github.com/GPUburnout/gpt2-from-scratch](https://github.com/GPUburnout/gpt2-from-scratch)

---

## Final Thoughts

Was this worth it? Honestly? Yes.

Not because I built something amazing — GPT-2 Small is table stakes in 2026. But because I finally *understand* the stack. The papers make sense now. When something breaks, I know where to look. When someone says "just use torch.compile," I know what it's doing under the hood.

The gap between "I've read the attention paper" and "I've debugged tensor shapes at 2 AM" is massive. This project closed that gap.

**My advice:** Start smaller than you think you need to. 400K params catches bugs before they're expensive. Scale up only when the small model works perfectly.

And for the love of GPUs, optimize *before* you start training. Not on epoch 4. I learned that one the expensive way.

*Questions? Check out the code at [github.com/GPUburnout/gpt2-from-scratch](https://github.com/GPUburnout/gpt2-from-scratch). I've made every mistake so you don't have to.*
