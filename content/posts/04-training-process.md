---
title: "11 Training Challenges and How I Solved Them"
date: 2026-02-02
draft: false
tags: ["training", "debugging", "optimization", "torch-compile", "AMP", "flash-attention", "Colab"]
summary: "A comprehensive guide to every way I shot myself in the foot training GPT-2 Small. Learn from my pain."
weight: 4
---

Training GPT-2 Small on 12GB of data sounded simple. It was not simple. Here are 11 ways the universe humbled me, and how I eventually fixed each one.

---

## Challenge 1: The Config That Lied

**The Error:**
```
RuntimeError: size mismatch for causal_mask: copying a param with shape
torch.Size([1024, 1024]) from checkpoint, the shape in current model is
torch.Size([512, 512])
```

**What I Did:** Tried to resume training with `SEQ_LEN = 1024`.

**What I Should Have Done:** Checked the checkpoint. It was trained with `SEQ_LEN = 512`.

**The Fix:** Change one number. Feel dumb for 10 minutes.

**Lesson:** Read your own configs before complaining about PyTorch bugs.

---

## Challenge 2: The dtype That Shall Not Be Named

**The Error:**
```
NotImplementedError: Could not run 'aten::nll_loss_forward_reduce_cuda_kernel_2d_index'
with arguments from the 'CUDA' backend.
'Int' (dtype)
```

**Translation:** CrossEntropyLoss wants int64. I gave it int32. It threw a tantrum.

**The Fix:**
```python
X = batch[:, :-1].long().to(device)  # .long() = int64
Y = batch[:, 1:].long().to(device)   # Don't forget the targets
```

**Lesson:** PyTorch loss functions are dtype snobs. Just `.long()` everything classification-related.

---

## Challenge 3: The Phantom Checkpoint

**The Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: '.../checkpoint.pth'
```

**What Happened:** My code looked for `checkpoint.pth`. My training script saved `pytorch_model.bin`. Nobody told me.

**The Fix:**
```python
model.load_state_dict(torch.load(f'{checkpoint_path}/pytorch_model.bin'))
optimizer.load_state_dict(torch.load(f'{checkpoint_path}/optimizer_state.bin'))
```

**Lesson:** Actually look at what files exist before writing code that loads them. Revolutionary, I know.

---

## Challenge 4: The Glacial Training Run 🐢

**The Symptom:** 1.6 seconds per step on an A100. The A100. The $10,000 GPU. Running like a potato.

**The Problem:** Memory-mapped files + random access = death by I/O.

Every batch, mmap was doing random reads across an 11GB file. The disk was crying. The GPU was bored.

**The Fix:** Just... load it into RAM:
```python
data = torch.from_numpy(np.fromfile('/content/tokens_bpe.bin', dtype=np.int32))
```

**Result:** 0.8s/step. 2x faster. The GPU started doing GPU things.

**Lesson:** If it fits in RAM, stop being clever with mmap.

---

## Challenge 5: Python Being Python

**The Symptom:** Still 0.8s/step. GPU utilization still sad.

**The Culprit:** This innocent-looking line:
```python
batch = torch.stack([data[i:i+SEQ_LEN+1] for i in idx])
```

Ah yes, a Python for loop in the hot path. Peak performance.

**The Fix:** Vectorize like an adult:
```python
offsets = torch.arange(SEQ_LEN + 1)
idx = torch.randint(0, train_size - SEQ_LEN - 1, (BATCH_SIZE,))
indices = idx.unsqueeze(1) + offsets
batch = data[indices]  # Zero Python loops
```

**Result:** 0.5s/step. 1.6x faster. Python loops are for preprocessing, not training.

---

## Challenge 6: The GPU That Wasn't Trying

**The Symptom:** 0.5s/step. GPU at 60% utilization. It's just... vibing.

**The Fixes:**

**Step 1 - torch.compile():**
```python
model = torch.compile(model)
```
PyTorch 2.0's magic spell. Fuses operations, generates custom CUDA kernels, makes everything faster.

**Step 2 - Mixed Precision (AMP):**
```python
scaler = torch.amp.GradScaler('cuda')

with torch.amp.autocast('cuda'):
    logits = model(X)
    loss = criterion(logits.view(-1, vocab_size), Y.view(-1))

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Bfloat16 goes brrr.

**Result:** **0.225s/step**. 2.2x faster. GPU finally earning its keep.

---

## Challenge 7: The OOM That Wouldn't Quit

**The Error:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate X GiB
```

**What I Did:** Got greedy. `batch_size=128`, `mode='max-autotune'`. YOLO.

**What Happened:** The GPU politely informed me that 40GB isn't infinite.

**The Fix:**
```python
BATCH_SIZE = 64  # Humility
model = torch.compile(model)  # Default mode, not "try everything and explode"
```

**Lesson:** `max-autotune` is cool but uses extra VRAM for kernel search. Start conservative.

---

## Challenge 8: The Zombie Memory 🧟

**The Horror:** After an OOM crash, GPU memory shows 79GB in use. Nothing is running. The memory just... won't leave.

**What I Tried:**
```python
gc.collect()  # Please?
torch.cuda.empty_cache()  # Pretty please?
torch.cuda.reset_peak_memory_stats()  # I'm begging here
```

**What Worked:** None of that. Runtime > Restart Runtime. Start over. Accept defeat.

**Lesson:** After OOM, the crashed process ghosts you with its memory. Restart is the only exorcism.

---

## Challenge 9: The Tokenizer That Changed Its Mind

**The Problem:** Trained with character tokenizer. New data uses BPE. Code hardcoded character tokenizer. Surprise!

**The Fix:**
```python
tokenizer_type = metadata.get('tokenizer_type', 'character')
if tokenizer_type == 'bpe':
    from tokenizer_bpe import BPETokenizer
    tokenizer = BPETokenizer()
else:
    from tokenizer import CharacterTokenizer
    tokenizer = CharacterTokenizer()
```

**Lesson:** Hardcoding is technical debt with a high interest rate.

---

## Challenge 10: The dtype Strikes Back

**The Problem:** Character tokenization used int16 (vocab < 65K). BPE uses 32K vocab but I used int32 for safety. Code assumed int16.

**The Fix:**
```python
dtype_str = metadata.get('dtype', 'int16')
mmap_data = np.memmap(args.bin_file, dtype=dtype_str, mode='r')
```

**Lesson:** Store dtype in metadata. Read dtype from metadata. Trust no assumptions.

---

## Challenge 11: The Last 2x (Flash Attention)

**The Symptom:** 0.225s/step. Good, but the A100 still had more to give.

**The Problem:** Standard attention materializes a massive `[batch, heads, seq, seq]` matrix. For batch=64, heads=12, seq=512, that's **800MB** just for attention scores — and it has to read/write from slow HBM memory.

**The Discovery:** PyTorch 2.0+ has Flash Attention built in via `scaled_dot_product_attention`:

```python
# Old way (materializes full attention matrix):
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
attn_weights = F.softmax(scores, dim=-1)
output = torch.matmul(attn_weights, V)

# New way (Flash Attention, automatic):
output = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
```

**What Flash Attention Does:**
- Processes attention in small tiles that fit in fast SRAM (~19 TB/s)
- Never writes the full attention matrix to slow HBM (~2 TB/s)
- Recomputes during backward pass instead of storing

**Result:** **0.1s/step**. 2x faster than AMP alone. The attention memory went from 800MB to ~20MB.

**Lesson:** If you're on PyTorch 2.0+ and not using `scaled_dot_product_attention`, you're leaving free performance on the table.

*Want the full technical breakdown? See [Training Optimizations Deep Dive: Flash Attention](/posts/06-training-optimizations-deep-dive#6-flash-attention-pytorch-20).*

---

## The Optimization Journey (1.6s → 0.1s)

| What I Did | Speed | Speedup | Effort Level |
|------------|-------|---------|--------------|
| Baseline (mmap) | 1.6s/step | 1x | N/A |
| RAM preload | 0.8s/step | 2x | 1 line of code |
| Vectorized batching | 0.5s/step | 3.2x | 20 minutes |
| + torch.compile | 0.35s/step | 4.6x | 1 line of code |
| + AMP | 0.225s/step | 7.1x | 10 lines of code |
| + Flash Attention | **0.1s/step** | **16x** | Built into PyTorch 2.0+ |

**16x speedup. ~140 hours saved. The A100 finally earned its electricity bill.**

*For detailed explanations of each optimization (torch.compile internals, AMP/GradScaler mechanics, vectorization patterns), see the [Training Optimizations Deep Dive](/posts/06-training-optimizations-deep-dive).*

---

## The Takeaways

### On Optimization
- **Profile first.** Is it I/O? Compute? Python? Find out before changing random things.
- **RAM beats mmap for random access.** Memory-mapping is for sequential reads, not ML training.
- **Vectorize or suffer.** Python loops in hot paths are a crime.
- **torch.compile + AMP + Flash Attention = free performance.** If you're not using all three in 2026, you're volunteering for slow training.

### On Colab Survival
- **Checkpoints go to Drive.** Colab will disconnect. It's not a question of if.
- **OOM = restart runtime.** `empty_cache()` is a suggestion, not a command.
- **Budget compute units.** A100 burns 8 units/hour. Math accordingly.

### On Debugging
- **Check dtypes.** Then check them again. Then add `.long()` anyway.
- **Config mismatches are silent killers.** SEQ_LEN, vocab_size, embed_dim — they all must match.
- **Verify file names exist.** Before writing the code that loads them. Wild concept.
