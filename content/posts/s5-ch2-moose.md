---
title: "My Code Agent Said It Was a Moose. I Said No. It Was a Moose."
date: 2026-04-12
draft: false
tags: ["season-5", "gpuburnout-3b", "moosefs", "debugging", "checkpoints", "training"]
description: "Twelve hours of debugging, five wrong suspects, one network filesystem, and a lesson in trusting the diagnostic."
season: 5
chapter: 2
---

The H200 was working. The 3B was training. After three days of fighting the cloud, the model was finally putting tokens through the GPU at 23,200 per second. I had a checkpoint at step 1,000. I had a checkpoint at step 1,200. I went to bed feeling, briefly, like a person.

Six hours later the run was dead. The checkpoint at step 1,200 was corrupted. The next run got to step 25 and froze. The one after that got to step 17 and silently disappeared.

My code agent told me it was MooseFS.

I told my code agent it was wrong. Partly because I had reasons. Partly because "MooseFS" sounded like the punchline to a joke whose setup I had missed. I am Canadian. We have a sense for when a moose is being put in front of us for comic effect, and I was not going to fall for it.

This is the story of the next twelve hours.

## Symptom 1: the compile deadlock

The first run after my first good night's sleep got to step 25 and stopped moving. tok/s dropped to 0. GPU util dropped to 0. The process was alive. It just was not doing anything.

I ran `py-spy` on the python process. It was stuck inside `futex_do_wait`. Some thread was waiting for some other thread to release some lock, and the wait was forever.

This is a known failure mode of `torch.compile`. The graph capture phase does some clever multithreaded work, and on the wrong combination of CUDA driver, kernel version, and dataloader worker count, that work can deadlock. There are GitHub issues about it. There are forum threads. The recommended fix is to disable `torch.compile`.

So I disabled `torch.compile`. tok/s dropped from 24,000 to 16,500. About a 30% throughput penalty. Not great. But it was running.

For 17 steps.

## Symptom 2: the silent CUDA crash

The 17-step run did not deadlock. It just vanished. One moment there was a python process. The next moment there was no python process and no error message in the log.

I checked dmesg. Nothing about OOM. I checked the CUDA logs. Nothing about a kernel fault. I checked the watchdog script. The watchdog script said the process had stopped producing output for six minutes and had killed it.

But the process had not been frozen. It had been training. tok/s was steady. Loss was going down. Then the watchdog killed it for being unresponsive.

This made no sense. I increased the watchdog timeout from 6 minutes to 15 minutes and relaunched. Step 22. Watchdog killed it again.

I started suspecting the watchdog. The watchdog had been written by me. The watchdog had been working for two seasons. The watchdog had not changed. But surely the bug was in the watchdog.

(My code agent suggested it might be MooseFS. I scrolled past that suggestion to read the watchdog source again.)

## Symptom 3: the corrupted checkpoint

I ran a save test. Saved a 35 GB checkpoint to the network volume. The save completed. The file size was right.

I ran md5sum on it. Mismatch.

OK, fluke. I ran it again. Different file. Different mismatch.

I tried smaller chunks. 3 GB at a time. Three out of four chunks would save cleanly. The fourth would corrupt at the very last second, on close, with an I/O error that the kernel did not surface to my python process. The torch.save call returned successfully. The file on disk was garbage.

This was concerning. I started thinking about the checkpoint format. Maybe `torch.save` had a known issue with large files. Maybe pickle was choking on something. Maybe the model state dict had a tensor that was breaking serialization.

(My code agent, around this point, was repeating itself. The diagnosis still said MooseFS. I had not changed my mind. I had also, by this point, not actually looked up whether MooseFS was a real thing. A small, confident voice in my head was insisting that my agent had hallucinated a filesystem with a funny name. The voice was wrong. MooseFS is real. It has a website and a logo and corporate customers since 2005. I would learn this six hours later.)

## Symptom 4: the dataloader worker hang

I rebooted the pod. Fresh start. New training run. The corrupted checkpoint file was deleted. Everything looked clean.

Step 14. Loss is fine. tok/s steady at 24,000. Then tok/s drops to 4,000. Then 800. Then 0.

GPU util is 0%. No deadlock symptoms. No CUDA error. The python process is alive. py-spy this time shows the main thread waiting in `poll_schedule_timeout`. It is waiting on a file descriptor. The file descriptor belongs to a dataloader worker.

The dataloader worker has stopped producing batches. It is stuck reading from disk.

I considered this for a moment. I had four dataloader workers. The data lived on the network volume. The workers were reading from the volume. The volume was MooseFS.

I thought: that's a coincidence. Reading should always work. Reads do not fail like writes do. There is no fsync semantics involved in reading data files. The workers must be hitting some Python multiprocessing bug, or maybe the prefetch queue is deadlocking against the GPU's memory pressure. There are seventeen GitHub issues about dataloader hangs. It is a known unstable surface.

I lowered `num_workers` to 2. Step 19. Hang. I lowered `num_workers` to 0. Step 7. Hang.

(The code agent's diagnosis was still MooseFS. I was now actively avoiding looking at that part of the log.)

## Symptom 5: the zombie processes

By this point I had launched and killed maybe twelve runs. Each one left some residue. After enough cycles, `nvidia-smi` was showing 141 GB of VRAM in use on a 143 GB GPU. There was no python process visible in `ps`. There were no running containers other than mine. The VRAM was just gone.

`pkill -9 python3` did nothing because there was no python3 to kill. The processes had been killed. They had just left their VRAM behind, like crash victims who managed to leave their luggage at the scene.

I had to restart the pod to clear it. Restarting the pod wipes `/tmp`. Restarting the pod also reboots the network mount. After the restart I had a fresh GPU and a fresh `/tmp` and the next run got to step 31 before it produced an I/O error on the next checkpoint save.

I sat there. I looked at the screen. I looked at five different symptoms across twelve hours. Compile deadlock. Silent CUDA crash. Corrupted checkpoint. Dataloader hang. Zombie processes.

I scrolled back up to the original diagnosis from the code agent.

It said MooseFS.

## What MooseFS actually is

It is, as it turns out, real.

MooseFS is an open-source distributed filesystem from Poland, released to the public in 2008. It has been in production at petabyte scale for almost two decades. The website is moosefs.com. The logo is a moose, drawn in red, made of geometric line segments that look like they were sketched on a circuit board. It is not winking at the joke. It is taking itself completely seriously, which is somehow funnier.

RunPod uses MooseFS for persistent volumes in some datacenters, including US-MD-1, which is where my volume lived. It is generally fine for moderate I/O. It is not fine for a 35 GB `torch.save` operation that holds a write transaction open for tens of seconds. It is not fine for four dataloader workers all trying to read random shards from the same volume at high throughput. It is fine for 12 GB checkpoints and 23 GB checkpoints, which is why it had worked for the entire 1B and 2B runs. It was not fine for 35 GB checkpoints, which is what 3B checkpoints looked like.

Once you accept that the filesystem is the problem, every symptom maps onto it cleanly:

| Symptom | What was actually happening |
|---|---|
| Compile deadlock | A dataloader worker was stuck reading from MooseFS. The futex was waiting on the worker. |
| Silent CUDA crash | A read was hanging long enough that the watchdog killed the parent. |
| Corrupted checkpoint | 35 GB write held the transaction open until MooseFS dropped a chunk on close. |
| Dataloader hang | Worker stuck in `poll_schedule_timeout` waiting on a MooseFS read that never returned. |
| Zombie processes | Killed runs left VRAM allocations because their I/O syscalls never returned to userspace cleanly. |

Five different bugs. One filesystem. One moose, drawn in red, taking itself completely seriously.

## The fix: nothing on the volume

The fix is simple and slightly humiliating. Use the network volume for nothing during training.

| Path | What lives there | Why |
|---|---|---|
| `/root/training_data/` | 71 GB of shards | Local SSD reads, no network I/O on the hot path |
| `/root/llm-dev/` | Training code | Same |
| `/tmp/ckpt_full_*.pt` | Latest 2 full checkpoints (35 GB each) | Local SSD writes, fast and reliable |
| Network volume | Nothing during training | Used only as a backup destination after training |

The container's local overlay disk had to be enlarged from 100 GB to 200 GB to fit the data plus two rotating checkpoints. RunPod requires a pod restart to resize the disk, which wipes `/tmp` again, which I learned the hard way after losing the just-saved checkpoint to the resize.

I rebuilt from a partially-saved chunk on the volume. I copied 71 GB of shards onto the local disk. I patched `train_v7.py` so that `save_checkpoint()` only writes to `/tmp`. I rebuilt my mental model of the run.

## The 3-layer checkpoint strategy

Once writes to MooseFS were off the hot path, I still needed checkpoints to survive a pod failure. Local SSD checkpoints disappear if the pod dies. So the new strategy was three layers, all running concurrently:

| Layer | Where | What | Why |
|---|---|---|---|
| L1 | `/tmp/ckpt_full_*.pt` | Last 2 full checkpoints, 35 GB each | Fast resume |
| L2 | HuggingFace Hub | Weights-only, 12 GB per save | Permanent, free, no infrastructure |
| L3 | My laptop's C: drive via scp | Full checkpoint, 35 GB | Belt-and-suspenders |

I tested all three running at step 2,500 while training continued. Save to local: 18.6 seconds. HF upload: 90 seconds at 132 MB/s, in the background. scp to my laptop: in the background. Training throughput throughout: 16.5K tok/s, no impact.

Then I turned `torch.compile` back on, switched to mb=8 ga=8, and watched tok/s climb to 24,640.

The training had a heartbeat again.

## What I learned

**Believe the diagnostic, even when it sounds ridiculous.** My code agent identified MooseFS in the first hour. I spent the next eleven hours discovering it had been right. Part of why I did not believe it: I was not sure MooseFS was a real thing. The diagnostic was offering me a name I had never heard of, attached to a creature my country uses for tourism photographs, and I let the absurdity of the name override the evidence in the logs. A diagnosis that sounds silly is not less likely to be correct. It is just less likely to be taken seriously, which is a different problem.

**Test the boring infrastructure first.** The 1B and 2B trained on the same volume with no problems. So I assumed the volume was a solved component. The volume was not a solved component for 35 GB checkpoints. Anything that worked at the previous scale should be re-tested at this scale. Especially the parts you think are trivial.

**Reads can fail too.** I had a mental model where filesystem writes can corrupt but reads always work. Reads can hang. Reads can timeout. Reads can leave a python process stuck inside `poll_schedule_timeout` for the rest of eternity. A working read history is not a guarantee of future read availability.

**One filesystem can produce five different bugs.** When five things break in the same hour and they all look unrelated, look for one cause underneath all of them. The cause is rarely all five separate things going wrong simultaneously. The cause is usually one thing they all share.

The training run resumed at step 2,500 with the new architecture. Next stop: 75,000 steps. Five days of nothing happening.

That sounds bad. It was actually the best thing that happened this season.

That's the next chapter.

---

*This is Chapter 2 of Season 5. Chapter 1 is the platform-pivot saga. Chapter 3 is the actual training run, in which nothing notable happened, and that was the entire point.*

*Total spent on Wrong Suspect Day: ~$45 in pod time, twelve hours of my life, and the lingering suspicion that I should learn to read my own logs more carefully.*
