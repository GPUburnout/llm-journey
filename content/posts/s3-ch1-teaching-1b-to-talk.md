---
title: "Teaching the 1B to Talk"
date: 2026-03-18
draft: false
tags: ["season-3", "sft", "fine-tuning", "garbage-tokens", "gpuburnout-1b"]
description: "In which I try to make a language model useful, discover something deeply wrong, and realize I've been asking the wrong question."
season: 3
chapter: 1
---

At the end of Season 2, I had a "working" 1B parameter language model.

The scare quotes are doing some heavy lifting. Yes, it could complete sentences. Yes, it knew Paris was a city. Yes, it could write paragraphs about single-cell RNA sequencing with journal citations that looked real and were absolutely not. Ask it the capital of France and it would confidently answer "the currency in the money is dollar and the currency is dollar and the currency is the euro and euro." Technically not wrong about the euro. Wildly wrong about everything else. As base models go, it was functional. As useful tools go, it was a paperweight that costs electricity.

The question I left Season 2 with: *can it be made useful?*

No. But it took nine experiments to find that out, and Season 3 is the body count.

---

## The Plan: SFT

A base model isn't a chat model. Ask it a question and it completes your question instead of answering it. It thinks you're typing a document and helpfully fills in the next word. Great for autocomplete. Terrible for conversation. Imagine asking a friend "how was your weekend?" and they reply "?" and walk away.

The fix is **Supervised Fine-Tuning (SFT)** - feed the model thousands of (instruction, response) pairs until it figures out what "helpful" looks like. This is how every chatbot you've ever used got its personality. It's also how every chatbot you've ever used got its annoying tics.

I ran SFT with **SlimOrca** (50K examples, LoRA) on both the 90K and 160K base models. Real question: did that extra $68 of Chinchilla pretraining from Season 2 actually buy me anything? Or did I light $68 on fire to find out?

---

## The Benchmarks: Nothing

| Benchmark | 90K Chat | 160K Chat |
|---|---|---|
| TruthfulQA MC2 | 41.53% | 41.08% |
| IFEval strict | 15.47% | 14.75% |

Flat. Statistical noise. The model that trained 70,000 extra steps on 9 billion more tokens scored *lower* on instruction following. So I lit $68 on fire to make my model slightly worse at multiple choice. Excellent.

If you stopped reading here, you'd conclude Chinchilla-optimal pretraining doesn't matter for chat. You'd also be completely wrong.

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

The 160K model writes a full paragraph and starts a numbered list. The 90K model writes two sentences and turns into a ouija board.

Worth $68. The leaderboard can't see it, but the leaderboard is also wrong.

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

Both models produce the *exact same* nonsense: `PersonX`, `AndroidRuntime`, `fefefe`, `oardvark`, `Paasilinna`, `substeps`, `usavik`. Not random noise. Specific. Reproducible. The model has approximately seven cursed words in its head, and when you push it past its comfort zone, it picks one and starts chanting.

| Model | Garbage prompts | Clean |
|---|---|---|
| 90K Chat | 5/8 | 3/8 |
| 160K Chat | 4/8 | 4/8 |

Extra pretraining helps at the margins. But both models collapse into the same garbage from the same source. And that source isn't the fine-tuning data.

The problem is baked into the base model. Something happened *before* SFT ever ran. Something I did to myself, weeks ago, without realizing it.

Next chapter: the forensics.

---

*GPUburnout documents real LLM training at small scale - actual costs, actual failures, actual numbers. Follow along at [gpuburnout.com](http://gpuburnout.com).*
