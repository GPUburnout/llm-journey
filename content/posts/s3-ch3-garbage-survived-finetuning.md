---
title: "I Tried Every Fine-Tuning Trick. The Garbage Survived All of Them."
date: 2026-03-21
draft: true
tags: ["season-3", "sft", "dpo", "fine-tuning", "garbage-tokens", "gpuburnout-1b", "data-quality"]
description: "A controlled experiment on why post-training alignment can't fix pretraining contamination - and what the data proves."
season: 3
chapter: 3
---

I had a diagnosis: garbage tokens come from contaminated pretraining data, not fine-tuning data. Python-Edu is the source. The contamination is baked into the base weights. SFT can't reach it.

That was the hypothesis. Proving it meant running nine experiments and watching every single one fail.

---

## SFT: Five Attempts, Five Failures

I built a cleaning pipeline, removed 27% of SlimOrca (139K examples), verified zero garbage tokens in the cleaned set, and ran five experiments:

| Run | Dataset | Examples | Garbage? |
|---|---|---|---|
| 1 | SlimOrca raw | 50K | Yes |
| 2 | SlimOrca raw | 50K | Yes |
| 3 | Cleaned SlimOrca | 50K | Yes |
| 4 | Cleaned SlimOrca | 10K | Yes |
| 5 | OpenAssistant (human-written) | 8K | Yes |

Run 5 is the one that matters. OpenAssistant is written entirely by humans. Zero contamination of any kind. Zero machine-generated text. Still produced identical garbage tokens.

That killed the SFT hypothesis. Dead. Buried. Flowers optional.

Here's the photosynthesis response after five SFT runs on three different datasets - verbatim, temperature 0.7:

> Photosynthesis is the process by which plants, algae, and certain bacteria convert light energy from the sun into chemical energy...
>
> `PersonX @.@ PersonXGenesis 1:1 AndroidRuntime ** __ Medalists usavik substeps...`

Two coherent sentences. Then collapse. Every dataset. Every configuration. Same tokens. It's like watching someone give a perfectly normal presentation and then suddenly start speaking in tongues.

---

## DPO: The 2x2 That Made Things Worse

If SFT couldn't fix it, maybe preference learning could. DPO shows the model pairs of outputs - preferred vs rejected - and trains it to favor the good ones. I built 1,200 labeled preference pairs across 10 categories, with clean outputs as "chosen" and garbage-heavy outputs as "rejected." All DPO runs used the 160K Chat model (the best 1B I had) as the base.

I ran a 2x2 Design of Experiments (DOE) - two variables, two levels each, four total combinations - varying epochs (1 vs 3) and beta (0.1 vs 0.3):

| Run | Epochs | Beta | Garbage (of 8) |
|---|---|---|---|
| 1 | 1 | 0.1 | 4/8 |
| 2 | 3 | 0.1 | 6/8 |
| 3 | 1 | 0.3 | 5/8 |
| 4 | 3 | 0.3 | **7/8** |

More training made it worse. Higher beta made it worse. The most aggressive configuration produced garbage on 7 out of 8 prompts. I somehow made my model *dumber* by trying harder to make it smarter. That's a special kind of achievement.

But the real gut punch was the loss curve. DPO Run 1 started at loss **0.693** and ended at **0.691**. That starting number - 0.693 - is ln(2). It's the loss of a model that literally cannot tell the difference between the two choices. A coin flip. My 1,200 carefully curated preference pairs moved the needle by 0.002. The model basically looked at my preference data, shrugged, and went back to doing whatever it was doing before.

---

## Why None of This Could Have Worked

The math is simple and depressing. Pretraining: 21 billion tokens. SFT: ~20 million. DPO: 1,200 pairs. That's like trying to change someone's personality with a Post-it note after they've lived 21 billion seconds of life.

Cranking up beta didn't help - it's like turning up the volume to drown out static. You just get louder static.

---

## The 1B Is Done

The 1B is archived. For the 2B, I rescanned everything, removed 660 contaminated documents, re-tokenized from scratch. 341 shards, ~38.4 billion clean tokens staged.

Season 3 ends here. Season 4: clean data, bigger model, same question. (Spoiler: it works.)

---

**TL;DR for anyone building their own LLM:** if your DPO loss starts at 0.693 and doesn't move, stop tuning hyperparameters. Your problem is upstream. Way upstream.

---

*GPUburnout documents real LLM training at small scale - actual costs, actual failures, actual numbers. Follow along at [gpuburnout.com](http://gpuburnout.com).*
