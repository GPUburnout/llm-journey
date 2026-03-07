---
title: "What GPUburnout-1B Actually Learned"
date: 2026-03-08
draft: false
tags: ["GPUburnout-1B", "benchmarks", "evaluation", "inference", "HellaSwag", "ARC", "MMLU", "season-2"]
description: "Benchmark scores, text samples at every milestone, and an honest look at where a $175 model punches above its weight — and where it rolls dice."
season: 2
chapter: 3
---

## Time to face the music

Training a language model is the fun part. You watch the loss drop, you generate text samples that are slightly less incoherent than yesterday's, you tell yourself "look, it almost knows what France is." It's addictive. It's rewarding. It also tells you absolutely nothing about how good your model actually is.

Benchmarking is where the universe hands you a report card you didn't ask for.

I ran GPUburnout-1B through four standard benchmarks using EleutherAI's lm-evaluation-harness — the same tool used to evaluate basically every open-source model you've ever seen on a leaderboard. Zero-shot, no tricks, no cherry-picking. Just the model and the questions.

Here's how it went.

## The Scorecard

| Benchmark | GPUburnout-1B | Random Baseline | Pythia-1B | TinyLlama-1.1B |
|---|---|---|---|---|
| **ARC-Easy** | **47.1%** | 25% | ~50% | 55.3% |
| **HellaSwag** | **28.8%** | 25% | ~47% | 59.2% |
| **ARC-Challenge** | **23.3%** | 25% | ~27% | 30.1% |
| **MMLU** | **23.0%** | 25% | ~26% | 25.3% |

Before you look at those numbers and think "well, that's... underwhelming," let me add some context that makes them either more or less depressing depending on your perspective.

**GPUburnout-1B trained on 11.8 billion tokens.** Pythia-1B trained on 300 billion. TinyLlama trained on 3 *trillion*. That's 25x and 250x more data, respectively. Comparing my model to theirs is like comparing someone who's been learning Spanish for a weekend to someone who lived in Madrid for a decade. The fact that I can order tapas at all is the surprise.

## ARC-Easy: The One That Worked

47.1%. Nearly double random chance. Within spitting distance of Pythia, which trained on 25x more data.

ARC-Easy is a science reasoning benchmark — elementary and middle school science questions. The kind where you need to know that plants need sunlight, ice is frozen water, and gravity pulls things down. Not exactly PhD-level stuff, but you'd be surprised how many billion-parameter models struggle with "which way does a ball roll on a hill."

This is GPUburnout-1B's best result, and it's not a fluke. My training mix is 87% FineWeb-Edu — web text specifically filtered for educational quality. The model didn't marinate in random Reddit threads and recipe blogs; it saw text that was *teaching things.* The dataset strategy paid off exactly where you'd expect: the model is disproportionately good at the kind of reasoning you find in textbooks. Garbage in, garbage out. Textbooks in, science scores out.

Getting within 3 points of Pythia on 4% of the training data is the result I'll be putting on my LinkedIn. If I'd told you before training that a $175 model would hit 94% of the score achieved by a model with institutional compute backing, you'd have been skeptical. I would have been skeptical. I *am* still skeptical, and I ran the eval myself.

## HellaSwag: The Humbling

28.8%. Only 3.8 points above random.

HellaSwag is a sentence completion benchmark. It gives you the beginning of a scenario and asks you to pick the most plausible continuation from four options. It tests common sense — the kind of intuitive understanding of how the physical and social world works that humans absorb from just *being alive* for a few decades.

Turns out, 11.8 billion tokens is not a few decades. It's not even a few months in model years. Pythia hits ~47% here with 300B tokens, and TinyLlama reaches 59% with 3T. The pattern is clear: HellaSwag performance scales directly with data volume, and I brought a squirt gun to a fire hose fight.

This is the benchmark that would improve the most if I continued training. Common sense completion requires seeing a staggering number of everyday scenarios to build the statistical intuition for "what usually happens next." My model has seen enough to form sentences. It hasn't seen enough to know that when someone picks up an umbrella, it's probably raining.

## ARC-Challenge and MMLU: Existential Humility

23.3% and 23.0%. Both *below* random chance (25%).

Let me be painfully clear about what this means: on these benchmarks, my model would literally perform better if it randomly guessed. It's not just failing — it's *anti-succeeding*. It has learned just enough to be confidently wrong, which, if you think about it, is a disturbingly human quality. We've all been in meetings with people like this.

ARC-Challenge is the harder sibling of ARC-Easy — same science domain, sharper questions. The 24-point gap between Easy (47.1%) and Challenge (23.3%) tells you everything about the model's depth: it has the broad strokes but nothing underneath. It knows "plants need sunlight" but can't reason about what happens when you put a plant in different experimental conditions. It read the chapter title and skipped the chapter.

MMLU covers 57 subjects from abstract algebra to world religions. At 23.0%, the model doesn't meaningfully know any of them. This is basically a knowledge inventory, and my model's warehouse is 59% empty — which, not coincidentally, is exactly how much training data it's missing relative to Chinchilla-optimal.

The silver lining? Even TinyLlama — trained on *250x more data* — only manages 25.3% on MMLU. This benchmark is just brutally hard at the 1B scale. Everyone is basically guessing. My model is guessing slightly worse, but it's in good company. When you're at a poker table and everyone is losing, the guy who loses least doesn't feel so bad.

## What the Model Sounds Like

Numbers are useful. Watching a model try to string words together is *visceral.* Here's what GPUburnout-1B actually generates at each checkpoint, using the same prompts across all milestones. These are real outputs, unedited, warts and imaginary lakes and all.

### "The capital of France is"

**Step 200** (loss 6.41):
> "that the world of the first of the number of the way..."

This isn't language. This isn't even trying to be language. This is what happens when a billion random numbers attempt to predict text. It's the literary equivalent of a cat walking across a keyboard, except the cat cost $4.

**Step 10,000** (loss 2.91):
> "the city of Paris. It is located at the south end of the city and is surrounded by the Mediterranean Sea and Lake Seine."

Knows Paris. Knows the Seine. Invented a lake. We'll call this "confident and approximately correct" — the model's signature vibe for the rest of training.

**Step 60,000** (loss 2.58):
> "located in Paris. It is the home to the French government, the European Union, the National Assembly, the European Parliament..."

Real institutions. No more imaginary lakes. The model now understands France as a political entity, not just a word that likes to appear near "Paris."

**Step 90,000** (loss 2.49):
> "The capital of France is the capital city of France is Paris, it is the largest city in the country and the largest in Europe. The population of the country is about 80 million people, it is one of the most densely populated countries in the world. Paris is the capital city of France, it is also the largest city of France. Paris is located in the central part of France, and it is the largest city in France and the second-largest French city."

The difference between 10K and 90K isn't just accuracy — it's *texture*. Early outputs read like Wikipedia on ambien. Late outputs read like a slightly confused but earnest encyclopedia.

### "def fibonacci(n):"

**Step 10,000:** Dashes and line noise. Doesn't know code exists.

**Step 30,000:** Knows comments exist. Produces lines starting with `#` followed by nonsense. It's learned the *aesthetic* of code without any of the substance. Instagram programmer energy.

**Step 60,000:** Understands function structure. Writes something with `if` and `return`. The logic is wrong, but the skeleton is real.

**Step 90,000:** Attempts recursion with base cases. Gets `if n <= 0: return 0` and `if n == 1: return 1` — which is actually correct. Then botches the recursive step. So close you can almost taste it, like a student who studied the right chapter but skipped the last page.

Remember: only 3% of training data was Python code. The model is trying to write code having seen very little of it. That it understands recursion *at all* is frankly surprising. That it can't execute it correctly is completely expected.

### "In a shocking discovery, scientists found that"

**Step 10,000:** Generic text about discoveries. Nothing specific.

**Step 90,000:**
> "...a woman's DNA can help to diagnose breast cancer... They discovered that the cells contained many genetic abnormalities, including cancer-causing genes."

And from other science prompts, references to "scRNAseq (Single-Cell RNA Sequencing)" with journal-style citations including volume numbers and years. All hallucinated, all formatted perfectly.

The model didn't learn facts. It learned the *language of facts* — the style, structure, and conventions that scientists use. It can write a sentence that *sounds* like a Nature paper without containing a single verifiable claim. Whether that's impressive or terrifying depends on your priors. As someone who spent years writing actual papers, I find it both.

## The Big Picture

Here's the honest summary of what 11.8 billion tokens and $175 bought:

**What worked:** General English fluency. Basic factual associations (Paris ↔ France, DNA ↔ cancer). Scientific writing conventions. Elementary science reasoning. The structural grammar of Python. An uncanny ability to sound authoritative while being wrong — a skill that, depressingly, also transfers to humans.

**What didn't:** Common sense reasoning. Deep knowledge in any specific domain. Actual functioning code. Anything that requires the kind of broad world experience you get from seeing hundreds of billions of tokens. The model has read the back cover of every textbook and the inside of none of them.

**The data cliff is real.** The gap between ARC-Easy (47.1%) and HellaSwag (28.8%) tells the whole story. Educational content transfers efficiently to educational benchmarks. Everything else — the messy, intuitive, common-sense understanding of how the world works — requires brute-force data volume that I simply didn't have. You can't shortcut your way to common sense with a clever dataset. You need the data.

**Model size is the real ceiling.** Even TinyLlama at 3T tokens only gets 59% on HellaSwag and 25% on MMLU at the 1B scale. There's a limit to what a billion parameters can represent, and we're all bumping against it like goldfish against the glass.

## What Would More Training Buy?

This is the question that follows me around like a stray cat. I trained to 59% of Chinchilla-optimal. The data is right there. What would happen if I pushed to 20 billion tokens?

My honest, possibly delusional, estimate:

- **ARC-Easy:** 50%+ (could match Pythia — and that would be a headline)
- **HellaSwag:** Mid-30s (meaningful improvement, still well behind Pythia's 47%)
- **ARC-Challenge:** Maybe crack 25% (crossing the random baseline — a low bar, but I'd take it)
- **MMLU:** Probably still in the low 20s (this benchmark is stubborn at 1B and everyone knows it)

Would that be worth ~$100? For the blog content? Absolutely — "matches Pythia on ARC with 25x less data" is a Reddit title that writes itself. For the model's actual usefulness? The fibonacci function still won't work. The common sense gap won't magically close. The ceiling is still the ceiling, just with slightly nicer wallpaper.

I'm still deciding. The tokens are still waiting. They're very patient.

## What's Next

Next and final post of Season 2: the lessons. Everything I learned about training a model from scratch that nobody told me and nobody writes about — from the revelation that cloud GPU pricing pages are works of creative fiction to the discovery that the biggest optimizations are always free and always embarrassingly obvious in hindsight.

**Next post: [10 Things I Learned Training a 1B Parameter Model That Nobody Talks About.](/posts/10-lessons-from-training-1b/)**

---

*This is Post 9 of an ongoing series. [Post 7](/posts/07-from-134m-to-1b/) covers architecture and dataset. [Post 8](/posts/08-the-175-dollar-experiment/) covers the training run and costs.*

*Follow along: [GitHub](https://github.com/GPUburnout) · [RSS](/index.xml)*
