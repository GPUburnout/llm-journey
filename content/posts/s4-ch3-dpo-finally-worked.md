---
title: "7 Out of 8 - How DPO Finally Worked"
date: 2026-03-29
draft: true
tags: ["season-4", "dpo", "sft", "alignment", "gpuburnout-2b", "fine-tuning"]
description: "In which the same technique that failed nine times on the 1B succeeds on the 2B - and the reason why is the whole point of Season 4."
season: 4
chapter: 3
---

Season 3: four DPO configurations on the 1B. Best result was 4/8 clean. Worst was 7/8 garbage - more training literally made it dumber.

Season 4: same technique, similar hyperparameters, on the 2B. Result: **7/8 clean.**

Same method. Different foundation. Completely different outcome. That's the whole story of Season 4 in one comparison. But the details are fun, so here they are.

---

## Step 1: SFT

Same setup as the 1B runs. SlimOrca 50K, LoRA r=16/alpha=32, 1 epoch, A100 80GB. Same dataset that failed to fix the 1B - now running on a clean base.

SlimOrca on the 1B produced garbage. SlimOrca on the clean 2B produced a functional chat model. The dataset didn't change. The foundation did.

---

## Step 2: DPO

1,078 preference pairs across 10 categories. (Started with 1,200 but lost 122 to a file overwrite accident. Yes, I overwrote my own training data. No, I don't want to talk about it.) Beta 0.1, lr 5e-7, 1 epoch, LoRA same as SFT.

Training time: ~2 minutes on A100. The entire alignment step - the thing that's supposed to make the model "safe" and "helpful" - took less time than making coffee.

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

**7/8 clean.** The one failure? The Python code prompt. Of course it's the Python prompt. Python-Edu contamination, even after removing 660 documents, left enough residual signal that code generation is still the weakest domain. The model didn't learn a new garbage pattern - it's the same old one, just much harder to trigger. Like a reformed smoker who still craves one at parties.

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

DPO doesn't make the model smarter. It still can't multiply 247 by 18. (Neither can most humans without a calculator, to be fair.) But it makes the model *cleaner*. When it runs out of coherent things to say, it stops instead of collapsing into nonsense. That's the alignment win - not brilliance, just knowing when to shut up. A skill some humans could also benefit from.

---

## The Benchmarks

| Benchmark | 2B Base | 2B DPO | 1B-90K Chat | 1B-160K Chat |
|---|---|---|---|---|
| TruthfulQA MC2 | 47.6% | **42.4%** | 41.5% | 41.1% |
| IFEval strict | - | **17.0%** | 15.5% | 14.7% |

DPO scores *lower* than Base on TruthfulQA. That's the **alignment tax** - you trade benchmark points for actually being useful. Every aligned model pays it. But even after paying, the 2B DPO beats every 1B variant on both benchmarks.

---

## Why It Worked

Same technique. Failed four times on the 1B, worked first try on the 2B. The difference isn't DPO - it's what's underneath.

The 1B had garbage attractors with 21 billion tokens of momentum. DPO on the 1B was arm-wrestling a freight train. DPO on the 2B is shaping preferences on a clean slate. One of those fights you can win. The other one you can't.

**Alignment is downstream of pretraining. Always.**

---

## Where This Leaves Us

GPUburnout-2B-75K-Chat-DPO is live on HuggingFace. 7/8 clean. Built for roughly $193 total - $183 pretraining, ~$10 for SFT and DPO combined. That's less than what I spent on GPU time debugging the 1B's garbage problem.

Season 4 ends with a working model and a fork in the road:

**Option 1:** Scale to 3B. More parameters, PubMed data in the mix, the beginning of BioLlama.

**Option 2:** Build something useful with the 2B. RAG over biomedical literature. A domain-specific assistant that actually does a job.

Both paths are on the table. Season 5 decides.

---

*GPUburnout documents real LLM training at small scale - actual costs, actual failures, actual numbers. Follow along at [gpuburnout.com](http://gpuburnout.com).*
