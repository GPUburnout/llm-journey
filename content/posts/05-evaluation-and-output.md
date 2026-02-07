---
title: "The Results Are In (And My Wallet Is Empty)"
date: 2026-02-06
draft: false
tags: ["results", "loss-curves", "lessons", "evaluation"]
summary: "Final loss curves, the damage to my compute budget, and 22 lessons I paid dearly to learn."
weight: 5
---

So. After 140 hours of GPU time, countless Colab disconnects, and one existential crisis about whether I should've just used a pretrained model... here's what I got.

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
3.0 |                *  *  *  *  ?  ?     <- Epoch 8: still going
    |
2.5 |                                      <- Would be nice
    |
2.0 |                                      <- Maybe in my dreams
    +--+--+--+--+--+--+--+--+--+--+--+
    1  2  3  4  5  6  7  8  9  10
                  Epoch

Legend: * = Actually happened, ? = Hopeful projection
```

It's going down. That's... that's good, right?

### The Epoch-by-Epoch Damage Report

| Epoch | Train Loss | Speed | Runtime | What Was Happening |
|-------|------------|-------|---------|-------------------|
| 1 | ~4.5 | 1.6s/step | ~35 hrs | Model: "What are words?" |
| 2 | ~3.8 | 1.6s/step | ~35 hrs | Model: "Oh, patterns exist!" |
| 3 | ~3.4 | 1.6s/step | ~35 hrs | Me: "Why is this so slow?" |
| 4 | ~3.24 | 0.225s/step | ~5 hrs | Me: "OH. That's why." |
| 5 | 3.093 | 0.225s/step | ~5 hrs | Finally making progress |
| 6 | ~3.05 | 0.225s/step | ~5 hrs | Loss go down |
| 7 | ~3.03 | 0.225s/step | ~5 hrs | Loss go down more |
| 8 | 3.014 | 0.225s/step | ~5 hrs | Currently cooking |
| 9-10 | TBD | 0.225s/step | ~5 hrs each | The home stretch |

Notice the jump from epoch 3 to 4. That's the 7x optimization kicking in. I literally wasted 105 hours being dumb.

---

## What The Model Actually Says Now

Here's the real test. Same prompts, different epochs. Watch it get less stupid:

### Prompt: "What is the capital of France?"

| Epoch | Output | Verdict |
|-------|--------|---------|
| 1 | `What is the capital of France? France France France France` | Brain damage |
| 4 | `What is the capital of France? The capital of France is a city.` | Technically true? |
| 8 | `What is the capital of France? The capital of France is Paris.` | We did it 🎉 |

### Prompt: "Explain machine learning in simple terms."

| Epoch | Output |
|-------|--------|
| 1 | `Explain machine learning in simple terms. terms terms terms` |
| 4 | `Explain machine learning in simple terms. Machine learning is a type of learning.` |
| 8 | `Explain machine learning in simple terms. Machine learning is a way for computers to learn from data without being explicitly programmed.` |

### Prompt: "Write a poem about coffee."

| Epoch | Output |
|-------|--------|
| 1 | `Write a poem about coffee. coffee coffee coffee coffee` |
| 4 | `Write a poem about coffee. Coffee is a drink. The drink is good.` |
| 8 | `Write a poem about coffee. In the morning light, I wake / With a cup of coffee to take / It warms my soul and clears my mind` |

Not Shakespeare, but not keyboard spam either. Progress.

---

## The Two Sessions: A Tale of Suffering and Enlightenment

**Session 1 (Epochs 1-3):** The Dark Ages
- ~105 hours of my life I'm not getting back
- Memory-mapped like an amateur
- 1.6s per step on an A100. AN A100.
- The GPU was probably playing solitaire

**Session 2 (Epochs 4-10):** I Learned Things
- RAM preload + vectorization + torch.compile + AMP
- 0.225s/step like a civilized person
- ~35 hours for 7 epochs
- GPU finally working for its electricity

---

## The Financial Damage 💸

| Item | Compute Units |
|------|---------------|
| A100 burn rate | ~8 units/hour |
| Epochs 1-3 (suffering) | ~840 units |
| Epochs 4-10 (enlightened) | ~280 units |
| **Total** | **~1,120 units** |

Let me do some fun math:

- **If I had optimized from the start:** ~400 units
- **What I actually used:** ~1,120 units
- **Units wasted on stupidity:** ~720 units (64% of total)

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
15. **torch.compile + AMP = mandatory.** It's 2026. There's no excuse.

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

## What's Next (If I Haven't Given Up)

1. **Finish epoch 10** — Almost there. Maybe.
2. **Test generation quality** — Does it produce coherent text? We'll see.
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
