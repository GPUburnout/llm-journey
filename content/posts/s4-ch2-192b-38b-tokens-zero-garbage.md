---
title: "1.92B Parameters, 38.4B Tokens, Zero Garbage"
date: 2026-03-28
draft: true
tags: ["season-4", "pretraining", "progressive-growth", "gpuburnout-2b", "data-quality"]
description: "In which a larger, cleaner model is built from the ruins of the last one - and the hypothesis from Season 3 starts looking right."
season: 4
chapter: 2
---

The 1B taught me that pretraining data quality matters more than anything that comes after it. Clean before you tokenize, or spend weeks trying to undo what you can't undo.

The 2B is the experiment that tests whether I actually learned that lesson, or whether I just *said* I learned it. Same architecture family, same training code, same evaluation suite. Different data - 600,000 documents scanned, 660 contaminated ones deleted, everything re-tokenized from scratch. If the hypothesis is right, the 2B should produce zero garbage tokens even before SFT runs. If it's wrong, congratulations, I just bought a $183 souvenir. Either way, somebody learns something.

---

## The Architecture

| | GPUburnout-1B | GPUburnout-2B |
|---|---|---|
| Parameters | 1.04B | 1.92B |
| Layers | 18 | 24 |
| Hidden dim | 2048 | 2304 |
| GQA heads | 32Q / 8KV | 36Q / 9KV |
| d_ff | 8192 | 9216 |

The 2B was *grown* from the 1B-160K checkpoint, not trained from scratch. Take the existing model, bolt on 6 new transformer layers, widen everything, copy the trained weights into the new structure, pad the new dimensions with small noise. Frankenstein's transformer. The model starts with the 1B's brain already loaded and gets a few new rooms in the attic for storage.

New layers get initialized as copies of their neighbors instead of random weights. A copied layer is functional from step 0. A random layer is pure noise and wastes thousands of steps just learning how to be a layer instead of, you know, computing things. Copying is free money. My favorite kind of optimization, right after "did you remember to turn it on?"

---

## The Mistake We Paid For

Every parameter got the same learning rate: 3e-4. Sounds reasonable until you remember the old layers already spent 160,000 steps carefully finding their happy place, while the new layers are showing up to their first day of work with no idea where the bathroom is. Same aggressive learning rate on both means the old layers go skidding off their optimum while the new layers are still trying to figure out which way is up.

Result: loss spiked from 2.446 to 2.80. A 14% temporary case of amnesia. The 1B's months of training, briefly forgotten, like a senior employee getting a concussion. It recovered. But those recovery steps cost real money and real time.

The fix is obvious in hindsight - separate learning rates for old layers (1e-5) and new layers (3e-4). Did I implement it? No. Filed under "next time." The "next time" list is always longer than the "this time" list. That's how it stays the "next time" list.

---

## The Training Run

Single A100 SXM 80GB on RunPod, $1.49/hr. Four phases to 75,000 steps:

| Phase | Steps | Train loss | Val loss | Throughput |
|---|---|---|---|---|
| Smoke test | 50 | 2.75 | - | 15,700 tok/s |
| Phase 1 | 0 → 1K | 2.679 | - | 15,700 tok/s |
| Phase 2 | 1K → 10K | 2.560 | 2.603 | 15,900 tok/s |
| Phase 3 | 10K → 30K | 2.495 | 2.510 | 15,730 tok/s |
| Phase 4 | 30K → 75K | **2.406** | **2.371** | 15,960 tok/s |

By step 30K, the 2B had already matched the 1B-90K's final loss (2.495 vs 2.494) - with only 3.9B tokens of post-expansion training. The growth was working. The 1B's knowledge was still in there, and the extra capacity was already paying rent.

**Total cost: ~$183.** Including ~$15 wasted on a disk-full crash mid-training because I didn't set up checkpoint rotation. Classic. The pod ran out of disk space the same way I run out of patience: gradually, then all at once. The 2B and I both learned things that day.

---

## The Benchmarks (Still Useless)

| Benchmark | 2B-75K | 1B-90K | Delta |
|---|---|---|---|
| HellaSwag | 28.9% | 28.8% | +0.1% |
| ARC-Easy | 47.0% | 47.1% | -0.1% |
| ARC-Challenge | 22.5% | 23.3% | -0.8% |
| MMLU | 23.0% | 23.0% | +0.0% |

Flat. Again. Four models, same story: **0-shot benchmarks are useless at sub-3B scale.** They're like judging a restaurant by counting the chairs. Sure, more chairs is more capacity. But are the chairs comfortable? Is the food good? Are people having a good time? The chair count cannot tell you.

Meanwhile, the 2B-30K was already citing specific genes (FOXP3), real journals (Journal of Bacteriology), and writing 150-token stories with named characters and actual plot. None of which the benchmarks can see, because the benchmarks are too busy counting chairs.

---

## The Proof

The hypothesis: clean pretraining data is the fix that nine fine-tuning experiments couldn't provide.

I ran the same 8 prompts that exposed the garbage tokens in the 1B. The 2B base model - no SFT, no DPO, just raw pretrained weights - produced **zero garbage tokens**.

Not fewer. None. Zero.

The same prompts that made the 1B vomit `PersonX` and `AndroidRuntime` produced coherent (if incomplete) continuations from the 2B. It still fails at math. It still runs out of things to say sometimes. But when it runs out, it *stops*. It doesn't fall into an attractor and start chanting `fefefe` like it's at a very weird football game.

The hypothesis holds. The contamination was in the data. The data is clean. The model is clean. The bouncer worked.

Now it just needs to learn how to have a conversation.

---

*GPUburnout documents real LLM training at small scale - actual costs, actual failures, actual numbers. Follow along at [gpuburnout.com](http://gpuburnout.com).*
