---
title: "7 Out of 8 - How DPO Finally Worked"
date: 2026-03-29
draft: true
tags: ["season-4", "dpo", "sft", "alignment", "gpuburnout-2b", "fine-tuning"]
description: "In which the same technique that failed nine times on the 1B succeeds on the 2B - and the reason why is the whole point of Season 4."
season: 4
chapter: 3
---

Season 3: four DPO configurations on the 1B. Best: 4/8 clean. Worst: 7/8 garbage. More training literally made the model dumber. I had receipts.

Season 4: same technique, similar hyperparameters, on the 2B. Result: **7/8 clean.** First try. No suffering necessary.

Same method. Different foundation. Completely different outcome. That's the entire moral of Season 4 in one A/B test. I could end the post here. I won't, because the details are too good to skip.

---

## Step 1: SFT

Same setup as the 1B. SlimOrca 50K, LoRA r=16/alpha=32, 1 epoch, A100 80GB. The *exact* dataset that failed to fix the 1B, now running on a clean base. I changed nothing. Not one hyperparameter. I wanted to see what happened.

Same dataset. Same hyperparameters. The 1B turned into a haunted slot machine. The 2B turned into a functional chat model. The dataset didn't change. The foundation did. That's the whole post in two sentences. (Also that's why pretraining matters more than alignment, but that's harder to fit on a t-shirt.)

---

## Step 2: DPO

1,078 preference pairs across 10 categories. (Started with 1,200 but lost 122 to a file overwrite accident. Yes, I overwrote my own training data. No, I do not want to talk about it. The ones that survived have my eternal gratitude.) Beta 0.1, lr 5e-7, 1 epoch, LoRA same as SFT.

Training time: **~2 minutes** on the A100. The entire alignment step - the thing frontier labs spend millions of dollars and entire research teams on - took less time than brewing a pot of coffee. Mine cost about 5 cents. The economics are so absurd I had to recheck the wandb dashboard twice.

---

## The Results

| Prompt | Status |
|---|---|
| Explain how photosynthesis works | CLEAN |
| What is the theory of relativity? | CLEAN |
| Write a Python function to reverse a string | GARBAGE |
| Tell me a creative story about a robot learning to paint | CLEAN |
| If a train travels 60 mph for 2.5 hours, how far does it go? | CLEAN |
| What are the ethical implications of AI in healthcare? | CLEAN |
| Explain the water cycle to a 10-year-old | CLEAN |
| What is the difference between a virus and a bacterium? | CLEAN |

**7/8 clean.** The one failure? The Python code prompt. *Of course* it's the Python prompt. Of all the prompts to fail on, the model picked the one most likely to remind it of its traumatic past. Python-Edu was the source of contamination, and even after removing 660 of its worst offenders, the model's relationship with Python remains complicated. It's the exact same old garbage pattern from the 1B - just way harder to trigger now. Like a reformed smoker who only relapses at weddings.

For comparison:

| Model | Clean | Garbage |
|---|---|---|
| 1B DPO Run 1 (beta=0.1, 1ep) | 4/8 | 4/8 |
| 1B DPO Run 4 (beta=0.3, 3ep) | 1/8 | 7/8 |
| **2B DPO (beta=0.1, 1ep)** | **7/8** | **1/8** |

The 1B's best was 4/8. The 2B beat it with the most conservative config on the first try.

---

## Base vs Chat vs DPO

All three 2B models, same 8 prompts, temperature 0.7:

| Prompt | Base | Chat | DPO | Winner |
|---|---|---|---|---|
| Photosynthesis | Incoherent | Good | Better | DPO |
| Train reasoning | Garbage | Correct then GARBAGE | Correct then GARBAGE | Tie (both fail) |
| Palindrome code | Incoherent | Pseudo-code garbage | Pseudo-code garbage | All fail |
| Ocean poem | Incoherent | Good then GARBAGE | Short but clean | DPO |
| Relativity | Incoherent | Good then GARBAGE | Full coherent answer | DPO |
| Math 247x18 | Incoherent | Wrong + GARBAGE | Wrong but clean | DPO |
| Water jugs | Incoherent | Wrong, clean | Wrong, clean | Tie |
| Ethics | Incoherent | Good | Good | Tie |

DPO doesn't make the model smarter. It still can't multiply 247 by 18. (To be fair, neither can most humans without a calculator. The model and I are even on this one.) What DPO does is make the model *cleaner*. When the model runs out of coherent things to say, it stops instead of collapsing into nonsense. That's the alignment win - not brilliance, just *knowing when to shut up*. A skill plenty of humans, frontier chatbots, and certain politicians could also benefit from.

---

## The Benchmarks

| Benchmark | 2B Base | 2B DPO | 1B-90K Chat | 1B-160K Chat |
|---|---|---|---|---|
| TruthfulQA MC2 | 47.6% | **42.4%** | 41.5% | 41.1% |
| IFEval strict | - | **17.0%** | 15.5% | 14.7% |

DPO scores *lower* than Base on TruthfulQA. That's the **alignment tax** - you trade benchmark points for actually being useful in conversation. Every aligned model pays it. Even after paying, the 2B DPO beats every 1B variant on both benchmarks. Tax included.

---

## Why It Worked

Same technique. Failed four times on the 1B. Worked first try on the 2B. The difference isn't DPO. It's what's underneath DPO.

The 1B had garbage attractors with 21 billion tokens of momentum. DPO on the 1B was arm-wrestling a freight train. DPO on the 2B is nudging preferences on a clean slate. One of those is a fight you can win. The other one is a fight that ends with you under the train, wondering where you went wrong.

**Alignment is downstream of pretraining. Always. Carve it on something.**

---

## Where This Leaves Us

GPUburnout-2B-75K-Chat-DPO is live on HuggingFace. 7/8 clean. Built for roughly **$193 total** - $183 pretraining, ~$10 for SFT and DPO combined. That's less than I spent debugging the 1B's garbage problem. The 2B cost less than the *autopsy* of the 1B. Let that sit for a second.

Season 4 ends with a working model and a fork in the road:

**Option 1:** Scale to 3B. More parameters, PubMed data in the mix, the beginning of BioLlama.

**Option 2:** Build something useful with the 2B. RAG over biomedical literature. A domain-specific assistant that actually does a job instead of just looking pretty on a leaderboard.

Both paths are on the table. Season 5 decides which one bankrupts me first.

---

*GPUburnout documents real LLM training at small scale - actual costs, actual failures, actual numbers. Follow along at [gpuburnout.com](http://gpuburnout.com).*
