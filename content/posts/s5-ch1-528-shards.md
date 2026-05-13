---
title: "I Have an A100. I Have 528 Shards of Data. I Cannot Combine Them."
date: 2026-04-07
draft: false
tags: ["season-5", "gpuburnout-3b", "runpod", "thunder-compute", "vram", "training", "cost-analysis"]
description: "Three days. Four GPUs. Three datacenters. Zero training tokens."
season: 5
chapter: 1
---

I had a 3B model expanded from the 2B-75K base. Code tested. Smoke test passed. 528 shards on my NAS, ~70 GB, ~38 billion tokens of FineWeb, FineMath, PubMed, and cleaned Python.

Three days later I had spent zero training tokens and was 1,200 words deep into a Notion page about VRAM accounting.

This is that story.

## Why a 3B

Two reasons.

**One: I wanted the next model to know what a kinase is.** The 2B was clean, polite, and had read a lot of FineWeb. It had also never seen a single PubMed abstract. I have plans for this model that involve answering biomedical questions, and you cannot retrieve your way out of a model that does not know what "phosphorylation" means. The 3B's data plan added 256 shards of PubMed, ~5.5B tokens, all fresh. The 2B is a polite generalist. The 3B is a polite generalist who also took two semesters of biochemistry.

**Two: I owed myself an architecture experiment.** When I grew the 1B into the 2B, I used a single learning rate across the whole model. The old layers, which had spent 160,000 steps finding their happy place, got the same aggressive 3e-4 as the new layers showing up to their first day. The model gave itself a concussion (loss 2.446 → 2.80) and spent thousands of steps recovering. I wrote at the time that the fix was "obvious in hindsight, filed under next time."

This was next time.

The 3B uses a differential learning rate: old params 1e-5, new params 3e-4, embeddings 5e-5. Whether it eliminated the loss spike is the subject of Chapter 3. (It did.)

So: a model with biomedical pretraining and a fix for the only thing the 2B got wrong. That was the plan. Here is what happened on the way to actually starting the training.

## The Thunder problem

I had been on Thunder Compute for the last few weeks. Good H100 pricing, $80 in credits sitting in the account. The plan was simple: launch on Thunder Production, run for ~5 days, ship the model.

Then I ran the smoke test.

| Config | Result |
|---|---|
| 3B, mb=4, no grad ckpt | OOM |
| 3B, mb=2, no grad ckpt | OOM |
| 3B, mb=4, grad ckpt ON | Passed at 8,700 tok/s |

Thunder advertises an H100 with 80 GB. The 3B in bf16 should fit. Why was I OOM'ing?

I started reading `nvidia-smi` like a tea leaf reader.

| Source | VRAM |
|---|---|
| Spec sheet | 80 GB |
| `nvidia-smi` total | 81,920 MiB |
| PyTorch reports available | 79.18 GiB |
| What PyTorch could actually allocate | ~63 GiB |

A 16 GiB ghost.

I emailed support. Carl from Thunder responded thoughtfully. Maybe CPU RAM. Maybe the GPU-over-TCP virtualization layer reserving headroom. We went back and forth for four days. Nobody had an answer. Meanwhile the credit clock was running.

## Plan B: the H100 that wouldn't show up

OK, fine. RunPod.

H100 SXM in US-GA-2. Real NVLink. Full PCIe Gen5. Working `torch.compile`. No virtualization eating 16 GB. The math said $254 on-demand for the whole 75K-step run.

I considered spot. The floor was $1.50/hr but US-GA-2 stock was Low, which had bid clearing up to $2.54. A 15% discount for the privilege of being kicked off mid-run. I have a personal rule: spot only at medium stock or higher. Pass.

H100 SXM, on-demand, $2.99/hr. Locked in.

I clicked deploy. The pod went into "starting" and stayed there.

| Attempt | Time in "starting" | Outcome |
|---|---|---|
| 1 | 7 minutes | Self-terminated |
| 2 | 9 minutes | Self-terminated |

This is a thing that can happen on RunPod. The host reserves the GPU. The container never comes up. Their UI cheerfully shows you an "active rental" while you cannot SSH into anything. The bill is small. The vibe is bad.

Stock indicator: Low. By the time I gave up: None. RunPod was showing me a price page for a GPU that did not exist.

This is the part of the day where most people would have gotten frustrated. I, instead, got clever.

## Plan C: I outsmart myself

While staring at the inventory page in despair I noticed an A100 SXM 80GB available in US-MD-1. **$1.49/hr.** Half the H100. And US-MD-1 was where my old `llama-training` volume already lived, with the 2B-75K checkpoint sitting on it from the previous season.

| Option | Hourly | Total | Migration |
|---|---|---|---|
| H100 SXM US-GA-2 | $2.99 | $254 | Re-upload 70 GB |
| **A100 SXM US-MD-1** | **$1.49** | **$216** | **None** |

Save $38. Skip a 70 GB upload. I had taken a setback and turned it into a win.

I felt smart. The productivity blogs would have been proud. Convert obstacles into opportunities. Find the silver lining. Make lemonade.

I deployed `realistic_lavender_lynx`. RunPod names pods after an adjective, a color, and an animal. `realistic_lavender_lynx` was the one that broke me.

I deleted the old training data, started uploading 528 shards via tar piped through ssh from the NAS, and went to make coffee. 30 MB/s steady. ETA 35 minutes.

This was finally going well.

I should have been more suspicious about how well it was going.

## The OOM matrix

Upload finished. Code synced. Expanded the 2B-75K into a 3B. Ran the smoke test.

OOM.

I tweaked the config.

| Config | Peak | Result |
|---|---|---|
| mb=4, ga=16, compile ON | 81 GB | OOM |
| mb=4, ga=16, compile OFF | 81 GB | OOM |
| mb=2, ga=32, compile ON | 81 GB | OOM |
| mb=2, ga=32, compile OFF | 81 GB | OOM |
| All of the above + `expandable_segments:True` | 81 GB | OOM |

Every combination OOM'd at 81 GB on a GPU with 80 GB. I lowered batch sizes, killed dataloader workers, considered sacrificing a goat. The goat was spared. Nothing else worked.

This was supposed to fit. My benchmark on Thunder three weeks earlier had peaked at 53-65 GB on the same 3B config. Why was the A100 hitting 81?

I went and re-read my benchmark code.

The benchmark used a fresh optimizer with no loaded state and `torch.randint` on the GPU as a fake dataloader. No checkpoint loading. No worker buffers. No CPU-to-GPU pressure. A beautiful, sterile, completely unrepresentative test environment.

I had been benchmarking a vacation home and trying to live in it.

## The fp32 AdamW problem

fp32 AdamW maintains two pieces of optimizer state per parameter, both full-precision floats. For a 3.12B parameter model, that is a lot of full-precision floats.

| Component | VRAM |
|---|---|
| Model (bf16) | ~6 GB |
| Optimizer state (`exp_avg` + `exp_avg_sq`, fp32) | **~25 GB** |
| Gradients (fp32) | ~12 GB |
| Activations | ~10-20 GB |
| CUDA workspace + kernel overhead | ~3 GB |
| Dataloader buffers (4 workers) | ~2 GB |
| **Realistic minimum with margins** | **75-90 GB** |

An "80 GB A100" was right at the cliff. The spec sheet was telling the truth. It was just answering a different question than the one I was asking. "Has 80 GB of VRAM" is not the same as "can train a 3B model with fp32 optimizer state."

## The 8-bit Hail Mary

`bitsandbytes` quantizes the optimizer state from fp32 to int8. Optimizer state from ~25 GB down to ~6.5 GB. Twenty-line change to `train_v7.py`.

Peak VRAM with 8-bit AdamW: 81 GB. It survived. Just.

I had bought myself about 100 MB of headroom, with an optimizer I did not fully trust, on hardware that would crash the run on any unlucky activation spike. I started training. ~10,000 tok/s. Sometimes. The loss curve had a small wobble in the first few thousand steps. Could have been noise. Could have been 8-bit quantization eating something. I could not tell.

This is the part of the story where I got lucky.

## Enter the H200

RunPod added H200 NVL inventory to US-MD-1 mid-afternoon. $3.39/hr.

I deployed `available_aqua_tarsier` (great name) and ran the smoke test on the H200 with full fp32 AdamW, no quantization, no gradient checkpointing, mb=8, ga=8, `torch.compile` on.

**23,200 tok/s. 126 GB / 143 GB peak. 17 GB of headroom.**

That is what training a 3B model is supposed to feel like.

| | A100 ($1.49/hr) | H200 NVL ($3.39/hr) |
|---|---|---|
| tok/s | ~10,000 (barely) | 23,200 |
| Hours to 75K | ~270 | ~117 |
| Wall clock | ~11 days | ~5 days |
| **Total** | **~$400** | **~$397** |

Identical total cost. The H200 charges twice as much per hour and finishes in less than half the time. The cheaper hardware was more expensive.

I stared at the table for a while.

## What I learned

Three things, paid for in dollars and dignity.

**"80 GB" is not a ceiling. It's a budget.** fp32 AdamW eats 25 GB before you have done anything. The real question for any training run is "how much VRAM after the optimizer eats its tax." That question is not on the marketing page.

**GPU clouds lie about availability.** "Low" stock can mean zero deployable in the next four hours. A pod in "starting" can stay there until the heat death of the universe and then quietly fail with no error. Have a Plan D.

**The cheap GPU is rarely cheap.** I picked the A100 because it was half the hourly rate. I felt clever. It was structurally too small, threatened to crash on every step, and would have taken twice as long. Throughput per dollar is the only number that matters. Hourly rate is the number that fools you.

Training started on the H200. It went fine for six hours.

Then I learned what MooseFS was.

That's the next chapter.

---

*This is Chapter 1 of Season 5. Season 4 ended with a 2B model that finally worked. This season is about what happened when I tried to grow it into a 3B.*

*Total spent before training a single token: a few dollars on phantom RunPod hosts, ~$3 on smoke tests, four days of my life, and one entire morning of feeling clever.*
