---
title: "Nine Experiments, Nine Funerals"
date: 2026-03-21
draft: false
tags: ["season-3", "sft", "dpo", "fine-tuning", "garbage-tokens", "gpuburnout-1b", "data-quality"]
description: "A controlled experiment on why post-training alignment can't fix pretraining contamination - and what the data proves."
season: 3
chapter: 3
---

I had a diagnosis. Garbage tokens, pretraining contamination, baked into the base weights, unreachable by fine-tuning. Open and shut. Case closed.

Except science doesn't accept "trust me bro" as evidence. The only way to *prove* the diagnosis was to try fixing it the wrong way and watch it not work. Repeatedly. With increasing desperation.

Nine experiments. Zero fixes. One scoreboard. Here we go.

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

Run 5 is the one that matters. OpenAssistant is written by *actual human beings*. People, with hands, typing words for a community-rated dataset. No machine text. No academic crud. No GPT-4 in sight. The cleanest possible SFT data. Still produced identical garbage tokens.

That killed the SFT hypothesis. Dead. Buried. No flowers necessary. The funeral was attended by me and a $1.49/hr A100, both of us tired.

Here's the photosynthesis response after five SFT runs on three different datasets - verbatim, temperature 0.7:

> Photosynthesis is the process by which plants, algae, and certain bacteria convert light energy from the sun into chemical energy...
>
> `PersonX @.@ PersonXGenesis 1:1 AndroidRuntime ** __ Medalists usavik substeps...`

Two coherent sentences. Then collapse. Every dataset, every configuration, same tokens. Like watching someone give a perfectly normal presentation and then, mid-sentence, start speaking in tongues.

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

More training made it worse. Higher beta made it worse. The most aggressive configuration produced garbage on 7 out of 8 prompts. I made my model measurably *dumber* by trying harder to make it smarter. Not many people can claim that as a research result.

But the real gut punch was the loss curve. DPO Run 1 started at **0.693** and ended at **0.691**. That starting number is ln(2) - the mathematical loss of a model that genuinely cannot tell the difference between two choices. A coin flip. My 1,200 carefully curated preference pairs - the ones I labeled by hand, prompt by prompt, agonizing over which response was "better" - moved the needle by 0.002. The model looked at my data, made the smallest possible "huh, neat" gesture, and went back to doing whatever it was doing before.

---

## Why None of This Could Have Worked

The math is simple and depressing. Pretraining: 21 billion tokens. SFT: ~20 million. DPO: 1,200 pairs. That's like trying to fix a 30-year personality with a Post-it note. The Post-it is pretty. The Post-it is well-meaning. The Post-it is not going to change anything.

Cranking up beta to push harder is like turning your speaker up to drown out static. Congratulations, you have invented louder static. You have not invented less static.

---

## The 1B Is Done

The 1B is archived. For the 2B, I rescanned everything, deleted 660 contaminated documents, re-tokenized from scratch. 341 shards, ~38.4 billion clean tokens, ready to go. Round two, this time with the bouncer at the door.

Season 3 ends here. Season 4: clean data, bigger model, same question. (Spoiler: this time it works. The bouncer was the move all along.)

---

**TL;DR for anyone building their own LLM:** if your DPO loss starts at 0.693 and doesn't move, stop tuning hyperparameters. Your problem is upstream. Way upstream.

---

*GPUburnout documents real LLM training at small scale - actual costs, actual failures, actual numbers. Follow along at [gpuburnout.com](http://gpuburnout.com).*
