---
title: "RIP GPUburnout-1B. Cause of Death: Its Own Training Data."
date: 2026-03-22
draft: true
tags: ["season-4", "pretraining", "data-quality", "gpuburnout-2b", "gpuburnout-1b"]
description: "In which the 1B model is officially retired, the autopsy is filed, and a better model begins."
season: 4
chapter: 1
---

Nine experiments. Zero fixes. Five SFT runs, four DPO runs, three different datasets - including one written entirely by humans. All failed. The most aggressive DPO config actually made things *worse*: 7 out of 8 prompts producing garbage. I tried to teach the model manners. It responded by getting louder. We've all been there.

Diagnosis confirmed. The garbage tokens are pretraining attractors from contaminated source data. No amount of post-training alignment can reach them. The bones were laid wrong. There is no fixing the bones.

Time to fill out the death certificate.

---

## The Autopsy

GPUburnout-1B is officially archived. Not deprecated - archived. There's a difference. Deprecated means it didn't work. Archived means it worked exactly as designed, and what it taught me is worth more than what it shipped. Same outcome on paper, very different funeral.

**Cause of death: its own training data.**

Specifically: Python-Edu. Stack Overflow answers and NLP tutorials with annotation artifacts (`PersonX`, `PROPN`, `AndroidRuntime`, `fefefe`) baked in. The model learned them as valid completions and collapsed into them like a student freezing on an exam and writing their own name in the margin, over and over, until time runs out.

The kicker: one hour with `grep` would have caught every single one of these tokens before they ever touched the tokenizer. Instead I spent three weeks running experiments that were dead on arrival. Tuition: $0 for the grep I didn't run, ~$200 for the autopsies I did. Worth it. Painfully worth it.

---

## The Cleaning

Before building the 2B, I went back to the source:

- **Python-Edu:** 660 contaminated documents identified and removed. Stack Overflow threads with heavy NLP annotation, Python tutorials with inline spaCy output, Android dev Q&A with exception traces. Real content, just incompatible with what I was building.
- **FineMath:** rescanned and re-tokenized from scratch.
- **Both verified:** zero garbage tokens across 600,000 scanned documents.

Then re-tokenized everything from scratch. Not patched. Not filtered at tokenization time. Fresh shards from clean JSONL. The fix only works at the source.

---

## The 2B Plan

With clean data in hand:

- **341 shards** across FineWeb-Edu, Python-Edu (clean), and FineMath (clean)
- **~38.4 billion tokens** - Chinchilla-optimal for 1.92B parameters
- **Zero garbage tokens** in the entire corpus

The 2B isn't a fresh start. It's a *growth* from the 1B-160K checkpoint - same model, expanded architecture, new layers added, existing weights preserved. The model already knows language. It already knows Paris is a city. It just needs more capacity and, critically, a corpus that doesn't try to teach it to say `AndroidRuntime` when stressed. Like sending the kid to a better school after a rough start.

The process:
1. Start from 1B-160K checkpoint (loss 2.446, 20.97B tokens)
2. Expand: 18 → 24 layers, hidden dim 2048 → 2304
3. Copy existing weights, pad new dimensions with small noise
4. Initialize new layers as copies of neighbors (not random - functional from step 0)
5. Reset optimizer, train on clean corpus

---

## The Scoreboard

GPUburnout-1B, final stats: 160,000 steps, 20.97 billion tokens, $243 total, final loss 2.446. It learned to talk. It taught me that data quality isn't just important - it's *everything*. Then it taught me again. And again. Eventually I listened. RIP.

GPUburnout-2B: 1.92 billion parameters, 38.4 billion clean tokens, growing on the 1B's bones. Same architecture family, more capacity, zero contamination. The hypothesis: clean pretraining data is the fix nine fine-tuning experiments couldn't deliver.

Season 4 is the test. My bank account is the unwilling co-investigator.

---

*GPUburnout documents real LLM training at small scale - actual costs, actual failures, actual numbers. Follow along at [gpuburnout.com](http://gpuburnout.com).*
