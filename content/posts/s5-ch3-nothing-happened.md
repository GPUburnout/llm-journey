---
title: "Nothing Happened for 75,000 Steps and It Was Glorious"
date: 2026-04-19
draft: false
tags: ["season-5", "gpuburnout-3b", "training", "differential-learning-rate", "loss-curve"]
description: "Five days of training. One small architectural fix. Zero crises. The loss curve I had wanted for two seasons."
season: 5
chapter: 3
---

After Chapter 1 (three days of cloud chaos) and Chapter 2 (twelve hours of blaming the wrong thing), you have earned the right to expect another disaster chapter. I am sorry. There is no disaster here. The training worked.

Here is the diary.

| Day | What happened |
|---|---|
| 1 | Loss went down |
| 2 | Loss went down |
| 3 | Loss went down |
| 4 | Loss went down |
| 5 | Loss reached 2.2475. Run complete. |

That is the whole season, basically. We can stop now if you want.

You are still here. Fine. Let's talk about why nothing happened.

## The fix that did not exist last time

Quick recap for new readers, and a callback for old ones.

When I grew the 1B into the 2B, I trained the whole model with one learning rate (3e-4). The old layers, which had spent 160,000 careful steps finding their happy place, got the same aggressive updates as the brand-new layers showing up to their first day. The result was a 14% loss spike (2.446 → 2.80) and several thousand wasted steps recovering. I wrote at the time that the fix was "obvious in hindsight, filed under next time."

This was next time.

The 3B uses a differential learning rate:

| Param group | Learning rate | Multiplier |
|---|---|---|
| Existing layers (from 2B) | 1e-5 | 0.033× |
| New layers (8 added for 3B) | 3e-4 | 1.0× |
| Embeddings | 5e-5 | 0.167× |

The math is intuitive. The old layers know things. Don't shake them. The new layers know nothing. Train them hard. The embeddings know some things and connect to all the layers, so split the difference.

The implementation was twenty lines of code in `train_v7.py`. The hypothesis was that this small change would prevent the loss spike entirely.

It did.

## The loss curve

Here is the honest part. The training log lived on a RunPod container disk that was reclaimed when the pod terminated, so the intermediate trajectory between checkpoints did not survive. What I have are the three anchor points the validation runs preserved:

| Step | Val loss | Notes |
|---|---|---|
| 2,500 | 2.390 | First eval of the run |
| **42,000** | **2.2256** | **Best val of the run** |
| 75,000 | 2.2475 | Final |

Between those points, the curve descended smoothly. No spike. No regression. No mid-run "I should be panicking" moment. The model started training and kept training until the step counter hit the number I had told it to stop at. I checked the loss twice a day. Each time it was lower than the last time. Boring. Glorious.

The slight bump at the end - val loss rising from 2.2256 at step 42K to 2.2475 at step 75K - is a real and documented training phenomenon. Late in the run, with a slightly stale data mix, the model starts memorizing patterns it had genuinely learned earlier. It does not get worse exactly. It just stops getting better and starts getting more confident about things it already knew. The best checkpoint is the one before it started doing that.

For comparison: when I bolted six new layers onto the 1B with a single learning rate, the 2B's loss went from 2.446 to 2.80 in the first few hundred steps and took thousands of steps to recover. That is what differential LR is for. The 3B did not do that. It just trained.

## A side note on the val loss

The 3B's best val loss was 2.2256 at step 42,000. The 2B's final val loss after 75,000 steps was 2.371. The 3B beat the 2B's *final* score while still pretraining, and beat it by quite a bit. Roughly 6% lower val loss.

Capacity matters. The 2B was a polite generalist with a clean education. The 3B is the same polite generalist with more rooms in the attic for storing things. The "more rooms" part started paying rent earlier than I expected.

(The benchmarks did not show this. The benchmarks at this scale are still useless. We will get to that in Chapter 4.)

## What I actually did for five days

To be clear about how boring this was: training was running on RunPod's H200 NVL, in tmux, on a pod I had detached from. My local machine was not involved. I checked on the run twice a day. Each check involved:

| Time | Activity |
|---|---|
| Morning | SSH to pod. Run `tail -50 /root/logs_3b/training.jsonl`. Note loss is still going down. Check disk usage on `/tmp` (rotating two 35 GB checkpoints, no problem). Close terminal. |
| Evening | Same thing. |

That was the entire operational load. Five minutes a day, twice a day. The rest of the time the training was doing something I had not been able to make it do for two seasons: it was running.

I used the time to start writing the next chapter, which is about SFT, which is also where things stop being boring.

## The receipts

Total cost for the run:

| Line item | Amount |
|---|---|
| H200 NVL pod, ~125 hours @ $3.39/hr | $397 |
| Network volume storage | $3 |
| Phantom rentals from Chapter 1 | A few cents (forgiven) |
| Pre-training adventures | ~$25 |
| **Total** | **~$425** |

Steady throughput: ~24,691 tok/s on the H200 NVL.

Tokens processed: 9.83 billion (about 26% of the data I had uploaded, or 0.26 epochs). The dataset was deliberately oversized at 37.8 billion tokens, so the model only saw a fraction of available shards at the planned mix ratio:

| Source | Share of mix |
|---|---|
| FineWeb | 73% |
| FineMath | 14% |
| PubMed | 10% |
| Python-Edu | 3% |

Plenty of headroom for a longer run if I had wanted one. I did not. I had budget for one full pass of pretraining and the SFT experiments to follow, and overspending now would have meant skipping those.

Wall clock: roughly five days from `python3 train_v7.py --resume` to step 75,000.

Garbage tokens during pretraining: zero. Same as the 2B. Clean data continues to be clean.

## What I learned

**Boring beats interesting.** When training goes well there is nothing to write about. When it goes badly there are blog posts. Papers show you the run that worked. They do not show you the seventeen runs they killed before it.

**Differential LR is cheap and skipping it is expensive.** Twenty lines of code. The 2B paid for skipping it with a 14% loss spike. The 3B paid for including it with nothing. Highest leverage twenty lines I have ever written.

**The lessons compound. The mistakes do not.** The 2B's loss spike taught me a fix I could not use until the 3B. That is annoying. It is also the only way the project gets less stupid over time.

**3B is the ceiling for single-GPU pretraining at this config.** The A100 (80 GB) could not do it. The H200 (143 GB) did it with 17 GB to spare. A 4B on the same H200 would land around 100-120 GB - twenty to forty gigabytes of headroom on a good day, zero on a bad one. The next model class is multi-GPU territory, or 8-bit optimizer territory, or both. The boring single-pod run we just shipped was the last one I get at this scale without that work.

Pretraining is done. SFT is next, which is where I discover that the best checkpoint is not the one with the lowest val loss.

That's the next chapter.

---

*This is Chapter 3 of Season 5. Chapter 1 was the platform pivot. Chapter 2 was the moose. Chapter 4 is the SFT chapter, in which the alignment step I skipped turned out not to matter.*

*Total spent training the 3B: $425, five days, five minutes a day of operational attention, and zero new gray hairs.*
