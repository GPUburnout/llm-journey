---
title: "My Model's Vocabulary Came from Stack Overflow at 3am"
date: 2026-03-18
draft: true
tags: ["season-3", "pretraining", "data-quality", "garbage-tokens", "gpuburnout-1b"]
description: "In which a forensic investigation into nonsense tokens reveals a problem that no amount of fine-tuning can fix."
season: 3
chapter: 2
---

My chat model had a haunted vocabulary. `PersonX`. `AndroidRuntime`. `fefefe`. `oardvark`. `Paasilinna`. The same seven nonsense tokens, in different prompts, at different temperatures, across totally separate runs. Not random. Specific. Reproducible. A slot machine that only ever pays out in cursed symbols.

I needed to find where they came from. Standard CSI episode: dust the model for prints, follow the trail back, identify the perpetrator.

My first suspect: the fine-tuning data. SlimOrca is GPT-4 generated, and machine text sometimes carries annotation crud from academic NLP datasets. Plausible. Easy to test. Confidently wrong.

The evidence killed that theory in about five minutes.

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

FineWeb-Edu: clean. Zero hits. The educational content datasets, ironically, were the educated ones. FineMath: a secondary offender, mostly because LaTeX docs are also a mess. Python-Edu: the smoking gun. Every garbage token traced back to it.

Three percent of the training mix. One hundred percent of the contamination. The smallest dataset was doing the most damage. As usual.

---

## Why Python-Edu Is the Problem

Python-Edu is built from Stack Overflow answers, code comments, and NLP library tutorials. The greatest hits collection of "code written at 3am by someone who just wants their script to run."

Roll call:
- `PROPN` and `prettyprint` are spaCy/NLTK annotation tags from NLP tutorials
- `AndroidRuntime` is an Android exception class haunting Stack Overflow threads
- `fefefe` is a CSS hex color (a charming off-white)
- `substeps` is LaTeX documentation
- `PersonX` and `PersonY` come from academic NLP corpora that use them as placeholder names

None of this is *wrong*. It's all real text from real sources. The problem isn't *that* the model learned this stuff. The problem is *what the model does when it gets nervous*.

When a language model runs out of coherent things to say, it doesn't gracefully shrug and admit ignorance. It collapses toward whatever high-frequency tokens it saw most often in confusing contexts during pretraining. For this model, those are `PersonX`, `fefefe`, and `oardvark`. They became its verbal tic. Most people say "um." This model says `AndroidRuntime`. It's a worse tic than "um" and significantly harder to therapy out of.

---

## Why You Can't Just Fix It Later

The scale asymmetry here is brutal:

- **Pretraining:** 21 billion tokens. Garbage sequences seen hundreds of times across thousands of documents. Baked into the base weights permanently.
- **SFT:** 15-25 million tokens of clean signal. Roughly 1,000x less.
- **DPO:** 1,200 preference pairs. A rounding error.

SFT can teach the model *how to respond*. It can't overwrite *what the model thinks is valid speech*. Those garbage tokens have 21 billion tokens of momentum behind them. Showing the model 25 million tokens where `PersonX` doesn't appear is like whispering "please stop" at a passing freight train. The train is not whispering back.

---

## The Only Fix

If contamination is in the pretraining data, the only fix is *in the pretraining data*. Not in fine-tuning. Not in DPO. Not in your favorite hyperparameter. Before tokenization - before the data ever touches the model.

I removed 660 contaminated documents from Python-Edu and FineMath. Rescanned 600,000 documents: zero garbage tokens remaining. Re-tokenized from scratch.

That clean corpus became the foundation for the 2B. But first I had to prove cleaning was actually necessary - that fine-tuning couldn't save the 1B no matter what I did. Nine experiments later, I had my proof.

That's the next chapter. Spoiler: nine experiments failing in a row.

---

**If you're building your own LLM:** scan your pretraining data *before* you tokenize it. One hour with grep and a token counter would have saved me three weeks of fine-tuning experiments and the slow, dawning horror of realizing none of them would ever work. Don't be me. Be slightly less me.

---

*GPUburnout documents real LLM training at small scale - actual costs, actual failures, actual numbers. Follow along at [gpuburnout.com](http://gpuburnout.com).*
