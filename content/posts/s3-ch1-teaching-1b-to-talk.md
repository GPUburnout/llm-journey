---
title: "Teaching the 1B to Talk"
date: 2026-03-18
draft: true
tags: ["season-3", "sft", "fine-tuning", "garbage-tokens", "gpuburnout-1b"]
description: "In which I try to make a language model useful, discover something deeply wrong, and realize I've been asking the wrong question."
season: 3
chapter: 1
---

At the end of Season 2, I had a working 1B parameter language model.

I use the word "working" loosely. It could complete sentences. It knew Paris was a city. Given the right prompt, it would produce coherent paragraphs about single-cell RNA sequencing with confidently fabricated citations. As base models go, it was functional. As useful tools go, it was a paperweight that costs electricity.

The question I left Season 2 with: *can it be made useful?*

No. But figuring that out took nine experiments and several weeks I'll never get back.

---

## The Plan: SFT

A base model isn't a chat model. Ask it a question and it'll complete your question, not answer it. It thinks you're writing a document, not having a conversation.

The standard fix is **Supervised Fine-Tuning (SFT)** - train it on thousands of (instruction, response) pairs so it learns what "being helpful" looks like. This is how every useful open-source model got its personality.

I ran SFT with **SlimOrca** (50K examples, LoRA fine-tuning) on both the 90K and 160K base models. The question from Season 2: does the extra $68 of Chinchilla pretraining actually matter for chat quality?

---

## The Benchmarks: Nothing

| Benchmark | 90K Chat | 160K Chat |
|---|---|---|
| TruthfulQA MC2 | 41.53% | 41.08% |
| IFEval strict | 15.47% | 14.75% |

Flat. Statistical noise. The model that trained 70,000 extra steps on 9 billion more tokens scored *lower* on instruction following.

If you stopped here, you'd conclude Chinchilla-optimal pretraining doesn't matter for chat. You'd be wrong.

---

## The Actual Outputs: Something

Same 8 prompts I've been using all project. **160K wins 3, ties 4, mixed 1. Zero losses.**

The difference isn't in benchmark points - it's in how long the model stays coherent before falling apart. Photosynthesis prompt, verbatim:

**90K Chat:**

> Photosynthesis is the process by which plants, algae, and certain bacteria convert carbon dioxide into sugars that they can then use for energy. This process allows plants to grow and thrive in environments with limited or no oxygen, making them an essential part of our ecosystem."/& \}\}' asilinna PersonZ substeps '\{\{ "\*\*\*", $"); PersonX @.@ \\\*\*\*\_ PersonXGenesis 1:1 - 3, 4, 5 AndroidRuntime  \_\_ Medalists usavik...

**160K Chat:**

> Photosynthesis is the process by which plants and other organisms convert carbon dioxide (CO2) into organic compounds such as glucose, oxygen, and water. This process is essential for the survival of all living things on Earth...
>
> There are several factors that affect photosynthesis, including:
>
> 1. Climate: The climate plays a significant role in determining the availability of sunlight...

The 160K model writes a full paragraph and starts a structured list. The 90K model writes two sentences and turns into a ouija board.

That $68 of extra pretraining? Worth it. Even if the leaderboard can't tell.

---

## The Twist

Look at that 90K output again.

```
PersonZ substeps PersonXGenesis 1:1 AndroidRuntime usavik
```

Now the creative writing prompt, 160K Chat:

```
Paasilinna usavik :/ PersonZ cerro, seis de la luz...
```

Different model. Different prompt. Different temperature. **Same garbage tokens.**

Both models produce the *exact same* nonsense: `PersonX`, `AndroidRuntime`, `fefefe`, `oardvark`, `Paasilinna`, `substeps`, `usavik`. These aren't random noise - they're specific, consistent, and completely deterministic. Push the model hard enough and it collapses into them every single time, like a nervous tic it can't shake.

| Model | Garbage prompts | Clean |
|---|---|---|
| 90K Chat | 5/8 | 3/8 |
| 160K Chat | 4/8 | 4/8 |

Extra pretraining helps at the margins. But both models collapse into the same garbage from the same source. And that source is not the fine-tuning data.

Something is wrong in the base model itself. Something baked in before SFT ever ran.

Where do those garbage tokens come from? Can they be fixed? Next chapter is the forensics.

---

*GPUburnout documents real LLM training at small scale - actual costs, actual failures, actual numbers. Follow along at [gpuburnout.com](http://gpuburnout.com).*
