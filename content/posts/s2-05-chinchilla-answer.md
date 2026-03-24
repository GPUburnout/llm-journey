---
title: "I Spent Another $68 Because a Spreadsheet Wouldn't Stop Staring at Me"
date: 2026-03-15
draft: false
tags: ["GPUburnout-1B", "Chinchilla", "scaling-laws", "training", "cost-analysis", "season-2"]
description: "I trained to 59% of Chinchilla-optimal and it haunted me. So I spent another $68 to finish the job. The benchmarks said I wasted my money. The outputs disagreed."
season: 2
chapter: 5
---

## The question that wouldn't leave me alone

[S2-03](/posts/s2-03-what-gpuburnout-1b-learned/) ended with a question I couldn't stop thinking about: *What would more training buy?*

GPUburnout-1B had trained on 11.8 billion tokens — 59% of Chinchilla-optimal for a 1B model. The data was sitting there. Twenty billion tokens is the theoretically ideal ratio for a billion parameters: twenty tokens per parameter, the point where your compute budget is perfectly balanced between model size and training data. I was 41% short of that line.

I'd told myself I was done. The loss curve was flattening. The diminishing returns math was brutal — [$550 per loss point](/posts/s2-02-the-175-dollar-experiment/) by Phase 4. I had written an entire [post about knowing when to stop](/posts/s2-04-lessons-from-training-1b/).

Then I spent a week staring at "59%" on a spreadsheet and decided I needed to know.

Budget: ~$75. Goal: push from 90K steps to 160K steps — roughly 20.97 billion tokens. Chinchilla-optimal.

## The training

Phases 5 and 6 ran on the same A100 SXM 80GB that powered the entire project, but this time on RunPod's spot instances at $0.95/hour instead of the $1.49 on-demand rate. Spot instances are the GPU equivalent of standby airline tickets — cheaper, but the airline can kick you off the plane mid-flight if someone with a full-price ticket shows up.

This happened twice in thirty minutes during one particularly annoying evening. The pod gets terminated without warning, you redeploy, SSH back in, resume from the last checkpoint, and pretend you're not annoyed. You are annoyed.

**Phase 5** (90K → 120K steps): ~36 hours, ~$34. Loss went from 2.494 to 2.530.

Yes, you read that right. The loss went *up*. Learning rate schedule effects, different data distribution in new shards, normal noise — it's not actually getting worse. But watching the number tick upward after paying $34 for the privilege is a specific kind of suffering that no scaling laws paper prepares you for.

**Phase 6** (120K → 160K steps): ~40 hours, ~$34. Loss dropped to 2.446. Redemption. Sort of.

Total additional spend: ~$68. Grand total for the project: ~$243.

| Phase | Steps | Final Loss | Cost | tok/s |
|---|---|---|---|---|
| Phases 1–4 | 0 → 90K | 2.494 | ~$175 | ~28,300 |
| Phase 5 | 90K → 120K | 2.530 | ~$34 | ~30,500 |
| Phase 6 | 120K → 160K | 2.446 | ~$34 | ~30,500 |
| **Total** | **160K** | **2.446** | **~$243** | |

The throughput actually improved — 30,500 tok/s versus the earlier 28,300 — from a newer PyTorch version on the spot instances. A free 8% speedup for doing absolutely nothing. The universe occasionally compensates you for the preemptions.

## Did more training help?

I ran the same benchmarks as [S2-03](/posts/s2-03-what-gpuburnout-1b-learned/). Same harness, same zero-shot conditions, same quiet prayer to the evaluation gods.

| Benchmark | 90K (11.8B tokens) | 160K (20.97B tokens) | Change |
|---|---|---|---|
| **ARC-Easy** | 47.10% | ~47% | Flat |
| **HellaSwag** | 28.80% | ~29% | Flat |
| **ARC-Challenge** | 23.30% | ~23% | Flat |
| **MMLU** | 23.00% | ~23% | Flat |
| **TruthfulQA MC2** | 48.04% | 48.40% | +0.36% |

Flat. Flat. Flat. Flat. And +0.36% on TruthfulQA, which is within the margin of error and is the benchmark equivalent of a participation trophy.

In S2-03, I made predictions about what Chinchilla-optimal would buy. Let me grade myself with the honesty I apparently lacked when making them:

- **ARC-Easy 50%+?** Didn't budge. I said this would be "a headline." The headline is now "Man Wrong About Everything." **F.**
- **HellaSwag mid-30s?** Still 29%. I was off by an entire bracket. **F.**
- **ARC-Challenge above 25%?** Still below random chance. My model would literally perform better by guessing. It spent $68 learning to be *more confidently wrong.* **F.**
- **MMLU improvement?** No. **F.**

Four predictions, four F's, zero dignity. I have a PhD and I just went 0-for-4 on predictions about my own model. The benchmarks looked at my $68 and shrugged.

## But the outputs tell a different story

Here's where it gets interesting.

Benchmarks are multiple-choice tests. They measure whether the model picks the right answer from four options. But language models don't take multiple-choice tests in the real world — they *generate text*. And the text at 160K is noticeably, undeniably better than at 90K.

I ran the same generation prompts I've been tracking since step 200. The differences don't show up in a score. They show up in what the model *does* with its next 200 tokens.

**The model learned to hallucinate better.** I realize that's a weird thing to celebrate. But hear me out. At 90K, "In a shocking discovery, scientists found that" produced:

> "...a woman's DNA can help to diagnose breast cancer... Single-Cell RNA Sequencing (scRNAseq)... The findings of the study were published in the journal Nature."

Real journal, real technique, completely invented study. At 160K:

> "...the water from the hot spring was almost completely saturated with microplastics, with some microplastics reaching levels that could pose a risk to human health. A study in Science Advances found that microplastics in the water were at least five..."

Also an invented study. Also a real journal. But here's what changed: the 90K model grabbed impressive-sounding words (scRNAseq! Nature!) and stapled them together. The 160K model constructed a *plausible narrative* — microplastics in hot springs is a real area of active research, *Science Advances* is where you'd actually find that kind of environmental study, and the framing ("levels that could pose a risk to human health") reads like a real press release. Both models are making things up. The 160K model is making things up *well enough to fool you for a paragraph.* That's a terrifying kind of progress, and it came from nine billion more tokens of reading how scientists actually write.

**The repetition loops disappeared — and factual recall exploded.** Our unofficial mascot prompt, "The capital of France is," tells the whole story across three checkpoints. At 90K:

> "The capital of France is the capital city of France is Paris, it is the largest city in the country and the largest in Europe... Paris is the capital city of France, it is also the largest city of France. Paris is located in the central part of France..."

It knows the answer. It can't stop saying it. Like a kid who studied one flashcard and wants you to know he studied. At 120K:

> "The capital of France is Paris. The capital of Canada is Ottawa. The capital of Australia is Canberra. The capital of India is New Delhi. The capital of Germany is Berlin. The capital of Russia is Moscow. The capital of Thailand is Bangkok."

Seven correct capitals. Unprompted. No repetition. The model went from being unable to stop saying "Paris" to rattling off a geography quiz. By 160K:

> "The capital of France is Paris. Credit card use in France: In France, credit cards are accepted in many shops, supermarkets, and restaurants..."

Paris, done, moving on to credit cards. Nobody asked about credit cards. But notice what *didn't* happen — no looping, no restating. The 90K model was a broken record. The 160K model answers, pivots, and stays on the new topic like someone who actually finished a thought before starting the next one.

**Creative writing grew sensory details.** "Once upon a time, in a land far away" — at 90K:

> "...a boy who had been born in the far north saw a man coming towards him, with a basket full of berries. The boy climbed up the tree and dropped the basket into the man's hands... The man then picked the basket up, took the berries and brought them to his own home..."

Generic, repetitive, berries going in circles. At 160K:

> "...there lived a very old woman who was very poor. She had nothing but two small bags of flour to put in the oven every day, and two small pots of water to boil in. She would always be very hungry."

Specific objects. A character with a life. You can picture the scene. That's the model learning, from nine billion more tokens of human writing, that stories are built from concrete details, not abstract gestures.

## What Chinchilla-optimal actually means at 1B

Here's what I expected: benchmarks would improve. They didn't.

Here's what actually happened: the model got *better at being a language model* in ways that benchmarks can't measure. Less repetition. More factual grounding. Richer detail. Longer coherence before degeneration.

| | 90K (11.8B tokens) | 160K (20.97B tokens) |
|---|---|---|
| **Repetition** | Still present | Minimal |
| **Factual recall** | Basic associations | Cites real journals, multiple capitals |
| **Detail quality** | Generic, filler-heavy | Specific, sensory |
| **Coherence length** | ~2-3 sentences before looping | Full paragraphs, topic transitions |

The 160K model doesn't know more facts than the 90K model — or if it does, the difference is too small for a four-option multiple-choice test to detect. But it *uses* what it knows more skillfully. It generates text that reads like it was written by something that understood the assignment, not something that memorized the study guide.

At 1B parameters, benchmarks hit ceilings fast. HellaSwag, MMLU, ARC-Challenge — they all plateau because the model doesn't have enough representational capacity to reason through the harder questions. More data doesn't help when the bottleneck is brain size. But more data *does* help with fluency, coherence, and factual grounding — qualities that show up in generation but not in multiple-choice scores.

**Chinchilla-optimal matters. But not the way the scaling laws papers imply.** The improvement isn't in benchmark points. It's in the texture of the output — the difference between a model that knows facts and a model that can *write about* facts. That difference is worth $68 to me, even if the leaderboard doesn't notice.

## What's next

The Chinchilla question is answered. The 1B model is as trained as the theory says it should be. Twenty billion tokens, one billion parameters, $243 total.

The benchmarks say it's roughly the same model. The outputs say it's quietly, measurably better. Both of those things are true, and the tension between them is the most interesting result of this entire experiment.

The real question now isn't whether this model can absorb more data. It's whether it can be made *useful* — and whether the only way past the 1B ceiling is more parameters. Season 3 is coming.

---

*This is Chapter 5 of Season 2. The full series: [Ch. 1 — Architecture](/posts/s2-01-from-134m-to-1b/) · [Ch. 2 — The $175 Experiment](/posts/s2-02-the-175-dollar-experiment/) · [Ch. 3 — Benchmarks](/posts/s2-03-what-gpuburnout-1b-learned/) · [Ch. 4 — Lessons](/posts/s2-04-lessons-from-training-1b/) · Ch. 5 — The Chinchilla Answer (you are here).*

*Season 1 (GPT-2, 134M parameters): [Start here.](/posts/01-why-build-a-language-model/)*

*Follow along: [GitHub](https://github.com/GPUburnout) · [RSS](/index.xml)*
