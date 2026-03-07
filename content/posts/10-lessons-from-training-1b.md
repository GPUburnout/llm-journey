---
title: "10 Things I Learned Training a 1B Parameter Model That Nobody Talks About"
date: 2026-03-07
draft: false
tags: ["GPUburnout-1B", "lessons", "cloud-gpu", "cost-optimization", "infrastructure", "season-2"]
description: "Infrastructure secrets, cost traps, and hard-won lessons from training a 1 billion parameter model from scratch on a $175 budget. The stuff that doesn't make it into research papers."
season: 2
chapter: 4
---

## The stuff that doesn't make it into papers

Research papers tell you about architectures, loss functions, and scaling laws. They do not tell you that the cheapest GPU per hour is almost never the cheapest GPU per token, that your biggest optimization is probably a boolean you forgot to flip, or that every single crash you'll experience will be infrastructure — never training code. They especially don't tell you that the five-second decision you make on day one about which datacenter region to pick will haunt you for the entire project.

I trained a 1B parameter model from scratch for $175. Along the way I made mistakes that cost real money, discovered optimizations that had been hiding in plain sight the whole time (mocking me, presumably), and learned that the boring operational stuff matters more than any clever algorithmic trick.

These are the lessons I wish someone had written down before I started. Nobody did, so here I am.

## 1. Region selection is the most important cost decision you'll make

Cloud GPU platforms like RunPod let you create persistent storage volumes (called "network volumes") to hold your data and checkpoints. These volumes are tied to a specific datacenter region. Your GPU pod must be in the same region as your volume.

Here's the trap: GPU availability varies wildly by region. I created my volume in US-MD-1 because it had A100s when I started. Seemed reasonable. I did not think about this for more than five seconds. This was a $50+ mistake.

By Phase 4, I wanted to try cheaper GPUs. The RTX A6000 at $0.25/hour? Not available in US-MD-1. The L40S at $0.40/hour? Not in US-MD-1. The RTX PRO 6000? Available in US-NC — a different region entirely. I could see it on the pricing page. I could admire it from afar. I could not use it. My 60GB of training data was in Maryland, and it was not moving.

I was locked in. The volume I created on day one — in five thoughtless seconds — determined my GPU options for the entire project. If I'd spent 30 minutes checking availability across regions *before* creating the volume, I could have saved $50–100+. That's 30–60% of total compute cost, lost to a decision I made between sips of coffee on the first morning.

**The rule:** Check GPU availability first. Create the volume second. This single decision at the beginning determines your entire cost structure.

## 2. $/hour is a lie. $/token is the truth.

The A40 GPU costs $0.40/hour. The A100 SXM costs $1.45/hour. The A40 is 3.6x cheaper. Obviously the A40 is the better deal.

I, too, can do division. I was wrong.

I tested both. The A100 pushes 28,300 tokens per second. The A40 manages 10,400. Why the gap? The A40 has 48GB of VRAM, which means gradient checkpointing is mandatory — and that recomputation overhead eats ~30% of throughput. Add in lower memory bandwidth (700 GB/s vs 2,039 GB/s) and fewer tensor cores, and the "3.6x cheaper" GPU is actually 2.7x slower. Net savings per token: 25%. Not the 73% my napkin math promised me.

| GPU | tok/s | $/hr | $/billion tokens | Savings vs A100 |
|---|---|---|---|---|
| A100 SXM 80GB | 28,300 | $1.45 | $14.23 | — |
| A40 48GB | 10,400 | $0.40 | $10.68 | 25% |
| RTX A6000 48GB (est.) | ~11,000 | $0.25 | ~$6.31 | 56% |

The A6000 at $0.25/hour would have been the best deal — 56% cheaper per token than the A100. It was never available in my region. See Lesson 1.

**The rule:** Always benchmark $/token trained, not $/hour. A GPU that's 3.6x cheaper per hour can be only 1.3x cheaper per token. The hourly rate is marketing. The per-token rate is reality.

## 3. The biggest optimizations are free

The three optimizations that had the largest impact on my training run cost exactly $0.00 combined. I'm still annoyed about this.

**Disabling gradient checkpointing: 18% speedup.** One boolean in a config file. I had 80GB of VRAM and was using 27GB. That's like paying for an all-you-can-eat buffet and eating a single breadstick. Turning off checkpointing used 44GB (still 36GB of headroom) and throughput jumped from 24K to 28.3K tok/s. This saved ~$17 over Phase 3 — nearly the entire cost of Phase 2. It was sitting there for weeks, waiting for me to notice it, judging me.

**Setting num_workers > 0: massive speedup.** With `num_workers=0`, the GPU sits idle while the CPU loads the next batch. Your $1.45/hour A100 is literally doing nothing, staring at the ceiling, waiting for data like a dog waiting for dinner. With `num_workers=2`, data loads in parallel while the GPU trains. One config line. I'm not going to tell you how long I ran with this at zero because I have a reputation to maintain.

**TF32 matmul: free precision boost.** Three lines of code. `torch.backends.cuda.matmul.allow_tf32 = True`. Small improvement, zero downside, zero cost. Available since Ampere GPUs launched in 2020. Mentioned in approximately zero beginner tutorials.

Combined, these three config changes matter more than any paid library, fancy kernel, or clever algorithm I tried. They were all available from day one. I found them at various points *during* training, which means I overpaid for every step before discovering each one. The universe does not refund you for this.

**The rule:** Before you install any new library, check if you're fully using the hardware you already have. You're probably not.

## 4. Optimization benchmarks are model-size dependent

I installed Liger Kernel after reading benchmarks showing a 20% throughput gain on A100s. Twenty percent! For free! I spent two hours fighting pip dependency conflicts to install it. I was going to be a genius.

The benchmark was on LLaMA 3-8B with a 128K vocabulary. My model: 1B parameters, 32K vocabulary. Speedup on my actual model: 0%. I tested it three times because denial is the first stage of grief.

Here's why: Liger's biggest win comes from Fused Linear Cross Entropy, which avoids materializing the full logit tensor. With a 128K vocabulary, that tensor is enormous. With my 32K vocabulary, it's 4x smaller and not the bottleneck. The RMSNorm and SwiGLU fusions? Also less impactful at smaller hidden dimensions (2048 vs 4096). And torch.compile was already capturing some of the fusion gains anyway.

The lesson isn't that Liger is bad — it's great, just not for me. Someone else's benchmark on a model 8x larger with a vocabulary 4x bigger tells you precisely nothing about your workload. Test on your actual model first, *then* get excited. I did it backwards and got a valuable education in premature optimization enthusiasm.

## 5. Gradient checkpointing is the hidden cost multiplier

This gets its own lesson because the implications are sneaky and expensive.

Gradient checkpointing saves memory by throwing away intermediate activations during the forward pass and recomputing them during backprop. The standard estimate is a 20–30% throughput penalty. That sounds manageable until you realize that **whether you need it at all depends entirely on your GPU's VRAM** — and the penalty cascades through your entire cost structure.

On my A100 (80GB): checkpointing unnecessary. Full speed. 28,300 tok/s.

On the A40 (48GB): checkpointing mandatory. Can't fit without it. 10,400 tok/s.

That's a 2.7x throughput difference, and checkpointing is a major contributor. VRAM headroom doesn't just affect memory — it directly affects *speed* and therefore *cost*. Every GB of unused VRAM is potential throughput you're leaving on the table. It's like having a bigger gas tank — it doesn't make the engine faster, but it means you don't have to stop and refuel every 20 miles.

When choosing GPUs, don't just ask "does my model fit?" Ask "does my model fit *without* gradient checkpointing?" Because the answer to that question is the difference between training at full speed and paying a 20–30% tax on every single step for the rest of your run.

## 6. Every crash was infrastructure, never training code

Across the entire project, eight things went wrong. I kept a list because I am the kind of person who keeps lists of things that went wrong. Here it is:

1. Disk space full on the pod
2. DataLoader serialization error
3. pip version conflict between libraries
4. Corrupted checkpoint from an interrupted test run
5. PyTorch version mismatch with Liger Kernel
6. RunPod host machine hardware error (deployed on a bad physical node)
7. Network volume quota exceeded during checkpoint export
8. Missing `num_workers` causing a data loading bottleneck

Notice anything? Not a single training bug. No gradient explosions. No NaN losses. No shape mismatches. No mysterious loss spikes. The training pipeline — version 7, with 5 bugs pre-fixed from earlier iterations — was rock solid from step 1 to step 90,000. It never once did anything unexpected. I cannot say the same for pip.

This matches what I've heard from everyone who trains at larger scale: the ML works. It's the *engineering* — disk space, dependencies, hardware reliability, networking, data pipelines — that will ruin your Saturday. If you're debugging a training run and the loss looks fine but something still broke, I can almost guarantee it's plumbing, not math.

## 7. Multi-GPU saves time, not money

This one broke my intuition. Two GPUs should be cheaper because you finish faster and pay for less wall-clock time, right? That's how bulk discounts work. That's how everything works.

That is not how GPU math works.

Training requires a fixed number of floating-point operations regardless of how many GPUs you use. Two GPUs do those FLOPs in half the wall-clock time, but you're paying for two GPUs, so the total cost is identical. Actually, it's worse — gradient synchronization between GPUs adds 4–12% communication overhead. You pay *more* for the same result. Faster, yes. Cheaper, no. It's like hiring two movers instead of one — the job finishes sooner, but you're paying both of them.

The exception: if you can find two cheap GPUs that together outperform one expensive GPU (2× A6000 at $0.25/hr = $0.50/hr combined vs 1× A100 at $1.45/hr), you win on both time *and* cost. But this requires distributed training code, both GPUs available in the same region, and the kind of luck that I historically do not have. For a solo project on a budget, single-GPU is almost always the right call.

**Multi-GPU is for deadlines. Single-GPU is for wallets.**

## 8. Early stopping is the best cost optimization you'll never read a paper about

Phase 2 cost $6.27 per loss point. Phase 4 cost $550. I'll wait while you re-read that.

The diminishing returns curve in language model training isn't just steep — it's a cliff with a sense of humor. The first 10% of compute gets you most of the learning. Everything after that is an increasingly expensive negotiation with a logarithm that does not care about your feelings or your budget.

I originally planned to train to 228,000 steps. I stopped at 90,000. That decision saved roughly $150+ in compute — nearly doubling the effective value of every dollar I'd already spent. The model at 228K would have been marginally better. *Marginally.* The $150 I saved was not marginal. The $150 was real and is currently sitting in my bank account instead of NVIDIA's.

Most training guides tell you how to start a run. Very few tell you how to decide when to stop. Here's my framework: track cost-per-loss-point at each phase boundary. When that number makes you wince, stop. When it makes you physically stand up from your desk and go look out a window, definitely stop. Your threshold will depend on your budget, but the important thing is having one — because the loss curve will never tell you to quit. It just keeps going down, slowly, expensively, forever.

## 9. Model size is the ceiling, not training duration

At step 10,000, GPUburnout-1B attempted a fibonacci function and produced dashes. At step 90,000, it produced a recursive function with correct base cases but wrong logic. That's real, visible, undeniable progress. But here's the uncomfortable truth — training to 228,000 steps wouldn't fix the logic. Training to a million steps wouldn't fix it. Training until the heat death of the universe wouldn't fix it.

Fibonacci requires tracking a recursive call stack, maintaining variable state across function calls, and reasoning about arithmetic. A 1B parameter model doesn't have the representational capacity for that. It's not a homework problem — it's a brain size problem. The model needs more layers, more dimensions, more parameters. A bigger brain, not a longer school day.

This informed my biggest strategic decision: **Season 3 is about teaching GPUburnout-1B to hold a conversation (SFT + alignment), not about training it longer.** And Season 4, when it comes, scales to 2B parameters — because that's where fibonacci might actually start working. *Might.*

More data makes the model better at what it *can* do. More parameters expand what it *can* do. If your model is failing at something fundamental, throwing more tokens at it is like studying harder for an exam that's above your grade level. Admirable work ethic. Wrong strategy.

## 10. The cloud GPU pricing page is a work of creative fiction

RunPod's pricing page lists the A6000 at $0.25/hour. I saw this number. I built a spreadsheet around this number. I told myself I was going to save a fortune. I was practically retired.

That GPU did not exist in my region. Not "limited availability." Not "sometimes available at 3 AM on a Tuesday." Zero units. Zero availability. Every day. For the entire project. The $0.25 A6000 was a Bigfoot sighting — everyone swears it's out there, but nobody I know has actually seen one.

The prices on cloud GPU pricing pages represent the cheapest rate across *all* regions, across *all* time. Your actual price depends on what's available in *your* region, at the exact moment you need it. It's like advertising a flight to Paris for $99 — technically true, if you're departing from a specific airport, on a specific date, at 4 AM, and you booked six months ago.

This isn't RunPod's fault. It's the nature of GPU clouds. Supply is limited, demand is insatiable (thanks, AI boom), and inventory varies by region, by hour, and by the phase of the moon. The price you see is the price you *could* pay in a perfect world. The price you'll actually pay is whatever's still available after everyone else has grabbed the good stuff.

**The rule:** Don't build your budget around the pricing page. Build it around what's actually available. Spin up a test pod in your target region before you transfer 60GB of training data and emotionally commit to a plan. If the cheap GPU isn't there, you want to know that *before* you've set up camp, not after.

## The Meta-Lesson

If I had to distill everything into one sentence: **the operational decisions matter more than the technical ones.** And nobody tells you this because the operational stuff is boring to write about and impossible to publish.

I spent days choosing between RMSNorm and LayerNorm. I should have spent that time checking GPU availability across regions. I read three papers on optimal learning rate schedules. I should have checked whether `num_workers` was set to zero. I benchmarked Liger Kernel for an entire day. I should have flipped a single boolean in a config file.

The architecture decisions are important, but they're one-time choices you make before training starts. The operational stuff — region selection, GPU benchmarking, config optimization, knowing when to stop — compounds across every hour of every training run. Getting the operations right saves more money than getting the architecture perfect saves loss points. It's not even close.

Next time I start a training run, the first thing I'll do isn't choose an activation function. It's survey every GPU in every region, benchmark $/token on my actual model, and triple-check every config flag. The boring stuff. The stuff nobody writes papers about. The stuff that actually determines whether you spend $175 or $350.

I just wrote 3,000 words about config files and pricing pages. If you told me a year ago this would be my most useful blog post, I would not have believed you. But here we are.

## Season 2: The Final Score

| | GPUburnout-134M (Season 1) | GPUburnout-1B (Season 2) |
|---|---|---|
| **Parameters** | 134M | 1.04B |
| **Training tokens** | 2.8B | 11.8B |
| **Total cost** | ~$15 (Colab) | ~$175 (RunPod) |
| **Best benchmark** | N/A | ARC-Easy: 47.1% |
| **Training time** | 21 minutes | ~80 hours |
| **Things that crashed** | Many | 8 (all infrastructure) |
| **Fibonacci** | Not attempted | Almost (right skeleton, wrong math) |
| **Imaginary lakes invented** | 0 | 1 (Lake Seine) |
| **Times I questioned my life choices** | Several | Countless |

Season 2 is a wrap. GPUburnout-1B exists, it works, and it cost less than a PS5. It can write about breast cancer genetics, attempt recursive Python, and cite scientific journals that don't exist. I'm proud of it the way you're proud of a kid who tried really hard at the science fair and got an honorable mention.

Whether I push it to Chinchilla-optimal or move straight to Season 3 (teaching it to actually hold a conversation instead of just autocompleting your sentences into oblivion), I'll let you know. Either way — if you're thinking about training a model from scratch, do it. The scaling laws papers can't teach you what a 2 AM disk space error teaches you. The math is the easy part. The engineering is the adventure. And the config files? The config files are where the money lives.

Thanks for following along.

---

*This is the final post of Season 2. The full series: [Post 7 — Architecture](/posts/07-from-134m-to-1b/) · [Post 8 — The $175 Experiment](/posts/08-the-175-dollar-experiment/) · [Post 9 — Benchmarks](/posts/09-what-gpuburnout-1b-learned/) · Post 10 — Lessons (you are here).*

*Season 1 (GPT-2, 134M parameters): [Start here.](/posts/01-why-build-a-language-model/)*

*Follow along: [GitHub](https://github.com/GPUburnout) · [RSS](/index.xml)*
