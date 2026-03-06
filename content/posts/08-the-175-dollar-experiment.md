---
title: "The $175 Experiment: Training GPUburnout-1B on a Single GPU"
date: 2026-03-01
draft: true
tags: ["GPUburnout-1B", "training", "loss-curves", "cost-analysis", "A100", "season-2"]
description: "I trained a 1 billion parameter language model from scratch on a single A100 for $175. Here's every phase, every dollar, and every surprise — including the moment it started citing real scientific journals."
---

## The short version

I trained a 1 billion parameter model from scratch. It took 90,000 steps, 11.8 billion tokens, one A100 GPU, and $175. The model went from generating random unicode soup to writing paragraphs about single-cell RNA sequencing with real journal citations.

This is the full story — every phase, every dollar, and every moment I stared at a loss curve instead of sleeping like a normal person.

## Phase 1: Smoke Test (Steps 0–200)

**Goal:** Does it crash?

This is the most underrated phase in any training run. You've spent days writing code, tokenizing data, debugging shape mismatches at 1 AM while questioning the life decisions that led you here. The smoke test is where you find out if any of it works, or if you've just built a very expensive space heater.

Step 1. Loss: 10.63. Expected: ~10.4 (that's ln(32005), which is what your model outputs when it has literally no idea what a word is. It's the mathematical equivalent of a shrug emoji).

Step 10. Loss: 9.58. It's dropping.

Step 50. Loss: 8.19. *Still* dropping. Nobody is more surprised than me.

By step 200, loss hit 6.41 and throughput had stabilized at 23,500 tokens per second. No NaN gradients, no memory errors, no mysterious segfaults. Every single component — data pipeline, model architecture, optimizer, gradient checkpointing, torch.compile — was working. First try.

Well, "first try" after eight separate infrastructure issues caught during debugging. But here's the thing — all eight were stuff like disk space, DataLoader serialization, pip version conflicts. Computers being computers. The actual training code? Zero bugs. The v7 training pipeline was bulletproof. It was the *everything around it* that kept trying to die.

**Phase 1 cost: ~$4.** The cheapest dopamine hit of the entire project. Cheaper than my coffee habit. More satisfying, too.

## Phase 2: Proof of Life (Steps 200–10,000)

**Goal:** Loss below 4.0. Coherent English. Proof that I haven't made a $30 mistake.

Phase 2 is where you find out if your model is learning language or just getting suspiciously good at predicting newlines. My target was a loss of 4.0–4.5 by step 10,000. I'd spent real time calibrating this against published scaling curves. I was feeling scientific about it.

The model did not care about my calibrations.

Loss hit 3.0 at step 7,750. It blew through my Phase 2 target before Phase 2 was 80% done. By step 10,000, training loss was 2.907 with a validation loss of 2.948. The train/val gap was essentially zero, which means no overfitting — just a model learning faster than I expected and making me look bad at forecasting.

Throughput held steady at 24,066 tokens per second for the entire run. Fifteen hours. 1.3 billion tokens. Not a single fluctuation. The A100 is, if nothing else, *consistent.* More consistent than my motivation, certainly.

At step 10,000, I asked the model: *"The capital of France is"*

> "the city of Paris. It is located at the south end of the city and is surrounded by the Mediterranean Sea and Lake Seine."

It knows Paris is the capital. It knows the Seine is a thing. It invented "Lake Seine," which is not a real body of water but honestly sounds like it could be a boutique hotel in the Marais.

For comparison, at step 200:

> "that the world of the first of the number of the way..."

So in 10,000 steps we went from word salad to inventing geographic features. I call that progress.

**Phase 2 cost: ~$22.** Cost per loss point: $6.27. Remember this number. It's going to make you sad later.

## Phase 3: The Grind (Steps 10,000–60,000)

**Goal:** Push through the steep part of the loss curve. Try not to refresh the WandB dashboard every five minutes. Fail at both.

Before kicking off the long run, I made the biggest single optimization of the entire project: **disabling gradient checkpointing.**

Quick explainer: gradient checkpointing saves VRAM by throwing away intermediate values during the forward pass and recomputing them during backprop. It's great when you're tight on memory. I was not tight on memory. I had an 80GB A100, and my model was using 27GB with checkpointing on. That's like renting a three-bedroom apartment and living in the closet.

Turning it off bumped VRAM to 44GB (still 36GB of headroom) and throughput jumped from 24,000 to 28,300 tokens per second. **An 18% speedup from changing a single boolean.** No new code, no new libraries, no clever algorithmic insight. Just using the GPU I was already paying for. This optimization alone saved roughly $17 over the course of Phase 3 — which means it basically paid for all of Phase 2. Free money, sitting there in a config file, waiting for someone to flip it.

I also installed Liger Kernel, which fuses multiple GPU operations into single kernels. The benchmarks promise a 20% throughput gain. On 8B models with 128K vocabularies. On *my* 1B model with a 32K vocabulary? Zero. Measurable. Speedup. I waited for it. I checked twice. I checked a third time because hope is irrational. Nothing. The operations Liger optimizes just aren't the bottleneck at this scale. Always test optimizations on your own workload before getting excited about someone else's benchmark. I say this as someone who got very excited about someone else's benchmark.

Fifty thousand steps and 63 hours later, loss had crept from 2.907 to 2.573. That's 0.33 points for $94. Phase 2 got me 3.5 points for $22. The log-linear decay had arrived, and it was not here to make friends.

At step 60,000, the "capital of France" prompt returned:

> "The capital of France is located in Paris. It is the home to the French government, the European Union, the National Assembly, the European Parliament..."

No more imaginary lakes. It's listing real institutions with something resembling accuracy. The model now has a genuine internal representation of France as a political entity — not just a word that statistically likes to hang out near "Paris." That's a qualitative jump, even if the loss curve only moved 0.33 points.

**Phase 3 cost: ~$94.** The biggest check I wrote, and the phase where I started doing cost-per-loss-point arithmetic and immediately wished I hadn't.

## Phase 4: The Last Push (Steps 60,000–90,000)

**Goal:** Squeeze out whatever's left. Try not to think about the economics. Think about the economics anyway.

By this point I was deep in diminishing-returns territory. The original plan called for 228,000 steps — Chinchilla-optimal for 1B parameters. But the loss curve was flattening like a pancake, and every step was getting more expensive per unit of improvement, like a gym membership in February.

I ran 30,000 more steps. Loss went from 2.573 to 2.494 — a 0.079-point drop for $44. Phase 2 got me 3.5 points for $22. Phase 4 got me 0.08 points for *twice the money.* If Phase 2 was a happy hour deal, Phase 4 was bottle service at a club where they don't put prices on the menu.

But the model kept getting quietly more interesting. At step 90,000, *"In a shocking discovery, scientists found that"* returned:

> "...a woman's DNA can help to diagnose breast cancer... They discovered that the cells contained many genetic abnormalities, including cancer-causing genes."

And *"def fibonacci(n):"* produced a recursive function with base cases. The logic was wrong, but the *structure* was correct — it understood that fibonacci is recursive, needs base cases, and should return something. At step 10,000, the same prompt returned dashes and line noise. At step 90,000, it's writing code that looks right from across the room. Get closer and it falls apart, like a Hollywood set — but from across the room, it's convincing.

The most unsettling output came from a science prompt that generated text referencing "scRNAseq (Single-Cell RNA Sequencing)" and cited specific journals. The citations were hallucinated, obviously. But the *format* was flawless — journal name, year, volume number, the whole academic ritual. As someone with a PhD who's actually published papers, watching a mass of matrix multiplications independently reinvent the conventions of scientific writing was both impressive and mildly existentially threatening. I didn't teach it to do that. It figured out that scientists cite things by reading enough text written by scientists who cite things. I'm choosing not to think too hard about what that implies.

**Phase 4 cost: ~$44.**

## The Full Bill

| Item | Cost |
|---|---|
| Data upload + setup | ~$8 |
| Phase 1: Smoke Test (200 steps) | ~$4 |
| Phase 2: Proof of Life (10,000 steps) | ~$22 |
| Phase 3: The Grind (60,000 steps) | ~$94 |
| Phase 4: The Last Push (90,000 steps) | ~$44 |
| Storage | ~$3 |
| **Total** | **~$175** |

For $175 I got a model that writes about breast cancer genetics and attempts recursive Python functions. That's less than my monthly cloud bill and roughly what I'd spend on a nice anniversary dinner — which, coincidentally, I now owe my wife for all the evenings I spent watching a loss curve instead of being a functional human being.

## The Diminishing Returns Problem

Here's the table that made me close my laptop and go for a walk:

| Phase | Steps | Loss Drop | Cost | Cost per Loss Point |
|---|---|---|---|---|
| Phase 2 | 0 → 10K | 3.51 | $22 | **$6.27** |
| Phase 3 | 10K → 60K | 0.33 | $94 | **$285** |
| Phase 4 | 60K → 90K | 0.08 | $44 | **$550** |

Read that last column again. In Phase 2, six bucks bought a measurable leap in capability. By Phase 4, I was paying $550 per loss point — the kind of return on investment that makes venture capitalists quietly excuse themselves from the room.

This is the fundamental economics of language model training. The first 10% of your budget gets you 90% of the way there. The remaining 90% is an increasingly expensive argument with a logarithm. It's like trying to shave the last few seconds off a marathon — except each second costs exponentially more and at some point you have to admit you're not slow because you undertrained, you're slow because you only have two legs.

At 1B parameters, I've got two legs. More training won't grow a third one. That requires more parameters.

## The Road Not (Yet) Taken

Here's what keeps me up at night (when the loss curves aren't doing that job already):

GPUburnout-1B trained on 11.8 billion tokens. Chinchilla-optimal for 1B parameters is ~20 billion. I stopped at 59% of the theoretically ideal training budget. If this were a steak, I pulled it off at medium-rare when the recipe called for medium-well.

I have 30.6 billion tokens tokenized and ready. Nineteen billion have never been seen by the model. They're sitting on a hard drive, gathering digital dust, judging me silently.

The cost to finish: roughly $90–130 on an A100. Two nice dinners. One parking ticket in downtown Boston. A rounding error in a corporate ML budget and a genuine decision for a guy paying out of pocket.

So why did I stop? Partly economics — the diminishing returns cliff is steep and I'd spent the budget I'd planned. Partly pragmatism — at 1B parameters, there's a ceiling on what the model can learn regardless of how much data you throw at it. The fibonacci function will never actually work at this scale. That needs more parameters, not more tokens.

But I haven't closed the door. I'm genuinely still mulling over whether another $100 and 8 billion tokens would move the benchmarks enough to justify the spend. If I decide to go for it, you'll be the first to know. If not, Season 3 — teaching this thing to actually hold a conversation instead of just completing your sentences — is already being planned.

Either way, 19 billion unseen tokens are sitting on a hard drive, staring at me. They can wait. Probably.

## What's Next

Next post, I put GPUburnout-1B under the microscope: benchmark scores against published models, side-by-side text generation at every milestone, and an honest accounting of where this model punches above its weight and where it falls flat on its face.

Spoiler: there's one benchmark where it nearly matches a model trained on 25x more data. And there are others where it's basically rolling dice. Both of those results tell you something important — and neither is what I expected.

**Next post: [What GPUburnout-1B Actually Learned — Benchmarks, Samples, and Honest Numbers.](/posts/09-what-gpuburnout-1b-learned/)**

---

*This is Post 8 of an ongoing series documenting my journey building language models from scratch. [Post 7](/posts/07-from-134m-to-1b/) covers the architecture and dataset decisions that got us here.*

*Follow along: [GitHub](https://github.com/GPUburnout) · [RSS](/index.xml)*
