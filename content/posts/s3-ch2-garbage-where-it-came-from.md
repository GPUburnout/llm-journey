---
title: "I Found Garbage in My Model. Here's Where It Came From."
date: 2026-03-18
draft: true
tags: ["season-3", "pretraining", "data-quality", "garbage-tokens", "gpuburnout-1b"]
description: "In which a forensic investigation into nonsense tokens reveals a problem that no amount of fine-tuning can fix."
season: 3
chapter: 2
---

Both my chat models were producing the same garbage tokens - `PersonX`, `AndroidRuntime`, `fefefe`, `oardvark`, `Paasilinna`, `substeps`, `usavik` - across different prompts, different temperatures, different runs. Not random noise. Specific and deterministic. Like a slot machine that only lands on the same cursed symbols.

My first instinct: blame the fine-tuning data. SlimOrca is machine-generated, GPT-4 completions. Maybe it carried NLP annotation artifacts and the model picked them up during SFT.

That was the hypothesis. The evidence killed it in about five minutes.

---

## The Forensic Scan

I built a scanner. Simple logic: take each garbage token, search every JSONL file in my pretraining datasets, count hits.

Three datasets went into the 1B:

- **FineWeb-Edu** - web text filtered for educational content (87% of mix)
- **Python-Edu** - Python code and documentation (3%)
- **FineMath** - mathematical content (10%)

Results:

| Token | FineWeb-Edu | Python-Edu | FineMath |
|---|---|---|---|
| PROPN | 0 | **61** | 1 |
| prettyprint | 0 | **45** | 8 |
| substeps | 0 | **20** | 14 |
| fefefe | 0 | **6** | 1 |
| InstanceState | 0 | **3** | 2 |
| AndroidRuntime | 0 | **1** | 0 |

FineWeb-Edu: clean. Zero hits. FineMath: a secondary contributor. Python-Edu: the culprit. Every garbage token traced back to it.

Three percent of the training mix. One hundred percent of the problem.

---

## Why Python-Edu Is the Problem

Python-Edu comes from Stack Overflow answers, code comments, and NLP library docs. `PROPN` and `prettyprint` are spaCy/NLTK annotation tags embedded in tutorials. `AndroidRuntime` is an exception class from Android SO threads. `fefefe` is a CSS hex color code. `substeps` is from LaTeX documentation.

These weren't errors in the source - they were real content. Stack Overflow is real text. The model learned from it exactly as intended. The problem is what happens next.

When the model runs out of coherent things to say, it doesn't produce an "I don't know." It collapses toward whatever high-frequency token sequences it saw most often in confusing contexts during pretraining. For this model, those sequences are `PersonX`, `fefefe`, and `oardvark`. They became the model's verbal tic - its equivalent of saying "um" except instead of "um" it says `AndroidRuntime`.

---

## Why You Can't Just Fix It Later

The scale asymmetry here is brutal:

- **Pretraining:** 21 billion tokens. Garbage sequences seen hundreds of times across thousands of documents. Baked into the base weights permanently.
- **SFT:** 15-25 million tokens of clean signal. Roughly 1,000x less.
- **DPO:** 1,200 preference pairs. A rounding error.

SFT can teach the model *how to respond*. It can't overwrite *what the model considers valid output*. Those garbage tokens have 21 billion tokens of momentum behind them. Showing the model 25 million tokens where `PersonX` doesn't appear is like whispering "please stop" at a freight train.

---

## The Only Fix

If the contamination is in the pretraining data, the only fix is in the pretraining data. Not in fine-tuning. Not in DPO. Not in hyperparameter tuning. Before tokenization - before the data ever touches the model.

I removed 660 contaminated documents from Python-Edu and FineMath. Re-scanned 600,000 documents: zero garbage tokens remaining. Then re-tokenized from scratch.

That cleaned corpus became the foundation for the 2B model. But first I had to prove cleaning was actually necessary - that fine-tuning really couldn't save it. Nine experiments later, I had my proof.

That's the next chapter. Spoiler: it's nine experiments failing in a row.

---

**If you're building your own LLM:** inspect your pretraining data before you tokenize it. Build a token-frequency scanner and run it on your raw JSONL. An hour of inspection saves weeks of fine-tuning experiments that were doomed before they started.

---

*GPUburnout documents real LLM training at small scale - actual costs, actual failures, actual numbers. Follow along at [gpuburnout.com](http://gpuburnout.com).*
