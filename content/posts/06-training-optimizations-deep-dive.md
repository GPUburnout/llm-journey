---
title: "Training Optimizations Deep Dive: How I Made the A100 Actually Work"
date: 2026-02-07
draft: false
tags: ["optimization", "torch-compile", "AMP", "flash-attention", "vectorization", "deep-dive", "GPU", "performance"]
summary: "The complete technical reference for achieving 16x speedup. Every optimization explained with code and diagrams."
weight: 6
---

This is the deep technical reference for everything I learned the hard way. If you want the narrative version, see [11 Training Challenges](/posts/04-training-process) and [The Results](/posts/05-evaluation-and-output).

Fair warning: this post is long. Grab coffee.

---

## Overview: All Optimizations at a Glance

| Optimization | Speed Gain | Memory Impact | Effort Required |
|--------------|------------|---------------|-----------------|
| Pre-tokenization | ~2x | Saves CPU | One afternoon |
| RAM preload | ~2-3x | +5.6 GB RAM | One line of code |
| torch.compile | ~1.5-2x | +slight (kernels) | One line of code |
| AMP (Mixed Precision) | ~1.5-2x | -50% activations | 10 lines |
| Vectorized batching | ~1.2x | Negligible | 20 minutes |
| Flash Attention | ~2x | -90% attention memory | Built-in |
| **Combined** | **~16x** | Net savings | 3 wasted epochs |

Yes, you can get 16x speedup. No, I didn't do it from the start. Yes, I regret it.

---

## Batch Fundamentals

Before diving into optimizations, let's make sure we're on the same page about batches. Skip this if you're already comfortable with the basics.

---

### What is a Batch?

A **batch** is a group of training samples processed together. Think of it like doing laundry — you don't wash one sock at a time.

```
Your dataset: 5.5 million sequences (each 512 tokens)

Without batching (batch_size=1):
┌─────────┐     ┌─────────┐     ┌─────────┐
│ seq 1   │ →   │ seq 2   │ →   │ seq 3   │ → ... → 5.5M updates
│ forward │     │ forward │     │ forward │
│ backward│     │ backward│     │ backward│
│ update  │     │ update  │     │ update  │
└─────────┘     └─────────┘     └─────────┘

With batching (batch_size=64):
┌───────────────────────────────┐     ┌───────────────────────────────┐
│ seq 1, seq 2, ... seq 64      │ →   │ seq 65, seq 66, ... seq 128   │
│ forward (all 64 at once)      │     │ forward (all 64 at once)      │
│ backward (average gradient)   │     │ backward (average gradient)   │
│ update (once)                 │     │ update (once)                 │
└───────────────────────────────┘     └───────────────────────────────┘
        ~86,000 updates total (5.5M / 64)
```

---

### Why Not batch_size=1?

**Problem 1: GPU underutilization**
```
GPU with batch_size=1:
┌──────────────────────────────────────────────────────┐
│ ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│  5% Used                                   95% Idle  │
└──────────────────────────────────────────────────────┘

GPU with batch_size=64:
┌──────────────────────────────────────────────────────┐
│ ████████████████████████████████████████████████░░░ │
│                    90% Used              10% Idle    │
└──────────────────────────────────────────────────────┘
```

**Problem 2: Noisy gradients**
```
Single sample gradient:
  "This sample says: move weights LEFT a lot!"

Next sample gradient:
  "This sample says: move weights RIGHT a lot!"

Result: Model wobbles around instead of converging. Not great.
```

---

### The Batch Size Tradeoff

| Batch Size | Memory | Speed | Gradient Quality | Use Case |
|------------|--------|-------|------------------|----------|
| 1 | Minimal | Very slow | Very noisy | Debugging only |
| 16 | Low | Slow | Noisy | Limited VRAM |
| 64 | Medium | Fast | Good | Sweet spot for most |
| 256 | High | Faster | Smoother | Multi-GPU setups |
| 1024+ | Very high | Fastest | Very smooth | Large-scale training |

---

### Gradient Averaging: Why Batches Actually Help

```
Batch of 4 samples:

Sample 1 gradient: "move weight +0.1"
Sample 2 gradient: "move weight -0.3"
Sample 3 gradient: "move weight +0.2"
Sample 4 gradient: "move weight +0.4"
─────────────────────────────────────
Average gradient:  "move weight +0.1"

Instead of 4 conflicting updates, you get a consensus.
Your model learns the general direction instead of getting whiplash.
```

---

### Memory Usage Per Batch Size

For our 134M parameter GPT-2 Small:

| Batch Size | Input Tokens | Activations | Total GPU RAM |
|------------|--------------|-------------|---------------|
| 16 | 8K | ~2 GB | ~4 GB |
| 32 | 16K | ~4 GB | ~6 GB |
| 64 | 32K | ~8 GB | ~10 GB |
| 128 | 64K | ~16 GB | ~18 GB |

**Why activations grow linearly:** Each layer stores intermediate values for the backward pass. Double the batch = double the stored values = double the memory usage.

---

## 1. Pre-Tokenization (Do This Before Anything Else)

**What it is:** Convert text → token IDs once, save as binary file, never think about it again.

### Without Pre-Tokenization (Slow)
```python
# Tokenize every batch during training
for text in dataset:
    tokens = tokenizer.encode(text)  # SLOW - runs every epoch!

# Tokenizing 2.8 billion tokens, every epoch, for 11 epochs.
# That's 30.8 billion tokenization operations total.
```

### With Pre-Tokenization (Fast)
```python
# Pre-tokenize once locally, then load binary
tokens = np.memmap('tokens.bin', dtype=np.int16)  # Already done!
```

### Impact

| Metric | Before | After |
|--------|--------|-------|
| CPU overhead | High (tokenization) | Near zero |
| Disk format | Raw text | Binary int16/int32 |
| File size | 12 GB text | ~5.6 GB binary |

---

## 2. RAM Preload vs Memory-Mapping (mmap)

This is where I wasted 105 hours of A100 time. Don't be me.

### Memory-mapping
```python
data = np.memmap('tokens.bin', dtype=np.int16, mode='r')
# OS fetches pages from disk on demand
# Random batch sampling = random disk reads = slow
```

### RAM Preload
```python
data = np.fromfile('tokens.bin', dtype=np.int16)
# Everything in RAM - instant random access
```

### Comparison

| Metric | mmap | RAM Preload | Winner |
|--------|------|-------------|--------|
| RAM usage | ~0 (OS cache) | Full dataset (~5.6 GB) | mmap |
| Random access speed | Slow (disk I/O) | Fast (memory) | RAM |
| Best for | Huge datasets (>RAM) | Fits in RAM | Depends |

**Our case:** 5.6GB tokens. A100 has 80GB RAM. RAM preload wins by a landslide.

**The lesson:** If your dataset fits in RAM, just load it into RAM. Sometimes the clever solution is the slow solution.

---

## 3. torch.compile (One Line, Big Gains)

```python
model = TransformerLanguageModel(...)
model = torch.compile(model)  # One line, significant speedup
```

### What's Actually Happening Here?

Every PyTorch operation launches a "kernel" (a function on the GPU). Each launch has overhead:

```python
# Each line = separate kernel launch = GPU overhead
x = a + b         # Kernel 1: addition      (5-20 μs overhead)
x = x * c         # Kernel 2: multiplication (5-20 μs overhead)
x = torch.relu(x) # Kernel 3: ReLU          (5-20 μs overhead)
x = x / d         # Kernel 4: division      (5-20 μs overhead)

# That's 20-80 microseconds of overhead for 4 simple operations.
# Your transformer has thousands of operations per forward pass.
# The overhead adds up FAST.
```

### How torch.compile Helps

**Before (Eager Mode):**
```
Python → Op 1 → Launch Kernel 1 → Wait
       → Op 2 → Launch Kernel 2 → Wait
       → Op 3 → Launch Kernel 3 → Wait
       → Op 4 → Launch Kernel 4 → Wait

Total: 4 kernel launches, 4 waits
```

**After (Compiled):**
```
Python → Compiler analyzes the computation graph
       → Generates ONE FUSED kernel that does all 4 operations
       → Launch Single Kernel → Done

Total: 1 kernel launch, 1 wait
```

### Real Example: Layer Normalization

**Eager mode (4 kernels):**
```python
def layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)    # Kernel 1: read x, write mean
    var = x.var(dim=-1, keepdim=True)      # Kernel 2: read x, write var
    x = (x - mean) / torch.sqrt(var + eps) # Kernel 3: read x,mean,var, write x
    return x * weight + bias               # Kernel 4: read x,weight,bias, write output

# 4 kernel launches + 4 memory round-trips to GPU VRAM
# VRAM bandwidth is the bottleneck, not compute
```

**Compiled (1 fused kernel):**
```python
@torch.compile
def layer_norm(x, weight, bias, eps=1e-5):
    # Compiler generates ONE kernel that does all of this
    # in a single pass, reading x once, writing output once
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    x = (x - mean) / torch.sqrt(var + eps)
    return x * weight + bias

# 1 kernel launch + 1 memory round-trip
# 75% less memory traffic
```

### Impact on Our Transformer

| Component | Eager Kernels | Compiled Kernels | Reduction |
|-----------|---------------|------------------|-----------|
| Embedding lookup | 2 | 1 | 50% |
| Layer Norm (×24) | 4 × 24 = 96 | 1 × 24 = 24 | 75% |
| Attention (×12) | ~20 × 12 = 240 | ~5 × 12 = 60 | 75% |
| FFN (×12) | ~8 × 12 = 96 | ~2 × 12 = 24 | 75% |
| **Total** | **~434 kernels** | **~109 kernels** | **75%** |

75% fewer kernel launches = the GPU spends more time computing and less time twiddling its thumbs.

### The First Run Is Slow

```python
model = torch.compile(model)

# First forward pass:
output = model(input)  # SLOW (~30-60 seconds)
# Compiler is tracing your model, building computation graph,
# generating optimized CUDA code, and caching for future runs

# Second forward pass:
output = model(input)  # FAST (uses cached kernels)
```

---

## 4. AMP (Automatic Mixed Precision)

**What it is:** Use 16-bit floats for most operations, 32-bit only where you absolutely need it.

**Why it matters:** Half the memory, twice the speed. There's a catch (underflow), but we'll handle it.

---

### FP16 vs FP32: The TL;DR

```
FP32 (32-bit float, "single precision"):
┌─────┬──────────┬───────────────────────┐
│sign │ exponent │       mantissa        │
│1 bit│  8 bits  │       23 bits         │
└─────┴──────────┴───────────────────────┘
= 4 bytes per number
= 7 decimal digits of precision
= Can represent 0.00000001 to 340,000,000,000,000,000,000,000,000,000,000,000,000

FP16 (16-bit float, "half precision"):
┌─────┬──────────┬───────────┐
│sign │ exponent │ mantissa  │
│1 bit│  5 bits  │  10 bits  │
└─────┴──────────┴───────────┘
= 2 bytes per number (half!)
= 3 decimal digits of precision
= Can represent 0.00006 to 65,504 (much smaller range)
```

**Key insight:** FP16 is twice as fast on Tensor Cores and uses half the memory. But it can't represent very small numbers.

---

### The Underflow Problem

**Underflow** = when a number is too small, it becomes zero. This is bad for gradients.

```
FP16 minimum positive value: ~0.00006

Your gradient during training:
  Actual gradient:     0.00001  (small but important for learning)
  FP16 representation: 0.00000  (becomes zero)

Result: Your model stops learning and you spend hours wondering why.
```

---

### GradScaler: The Solution

**GradScaler** scales numbers up before they can underflow, then scales them back down after.

```
Without scaling:
  loss = 0.5
  gradient = 0.00001  → FP16 sees: 0 (underflow)

With scaling (scale = 1024):
  scaled_loss = 0.5 × 1024 = 512
  scaled_gradient = 0.00001 × 1024 = 0.01024  → FP16 sees: 0.01024 (preserved!)

  After backward:
  actual_gradient = 0.01024 / 1024 = 0.00001  (recovered)
```

---

### Complete AMP Code (Copy-Paste Ready)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # STEP 1: Forward pass in FP16 (fast!)
    with autocast():
        output = model(batch)
        loss = criterion(output, target)
        # autocast automatically uses FP16 where safe
        # Uses FP32 for softmax, loss, layer norm (precision matters there)

    # STEP 2: Scale loss, then backward (prevent underflow)
    scaler.scale(loss).backward()
    # Gradients are now scaled up → no underflow!

    # STEP 3: Unscale and update weights
    scaler.step(optimizer)
    # Unscales gradients, checks for INF/NaN, updates weights

    # STEP 4: Adjust scale for next iteration
    scaler.update()
    # If INF/NaN detected: reduce scale
    # If stable for a while: increase scale
```

---

### What autocast() Does Behind The Scenes

| Operation | Precision | Why |
|-----------|-----------|-----|
| Matrix multiply (Linear, Attention) | FP16 | Tensor Cores accelerate FP16 |
| Convolution | FP16 | Same |
| GELU, ReLU | FP16 | Simple ops, FP16 is fine |
| Softmax | FP32 | exp() needs precision |
| Layer Norm | FP32 | Variance needs precision |
| Loss calculation | FP32 | Small differences matter |
| Weight updates | FP32 | Accumulation needs precision |

You don't have to think about any of this. autocast() handles it automatically. One less thing to debug.

---

### Memory Savings

```
Without AMP (all FP32):
┌────────────────────────────────────────────┐
│ Model weights:     134M × 4 bytes = 536 MB │
│ Gradients:         134M × 4 bytes = 536 MB │
│ Optimizer states:  134M × 8 bytes = 1.07 GB│
│ Activations:       ~2 GB                   │
│ ─────────────────────────────────────────  │
│ TOTAL:             ~4.1 GB                 │
└────────────────────────────────────────────┘

With AMP (mixed FP16/FP32):
┌────────────────────────────────────────────┐
│ Model weights:     134M × 4 bytes = 536 MB │ (master copy)
│ FP16 weights:      134M × 2 bytes = 268 MB │
│ Gradients:         134M × 2 bytes = 268 MB │
│ Optimizer states:  134M × 8 bytes = 1.07 GB│
│ Activations:       ~1 GB (HALF!)           │
│ ─────────────────────────────────────────  │
│ TOTAL:             ~3.1 GB                 │
│ SAVINGS:           ~1 GB (24% less)        │
└────────────────────────────────────────────┘

The real win: activations scale with batch size.
With AMP, you can double your batch size!
```

---

## 5. Vectorized Batch Creation

**What it is:** Stop writing Python loops in the hot path. Let NumPy do the work.

---

### The Problem with Python Loops

```python
# Slow approach
batch = []
for i in range(64):                           # 64 iterations
    start = random.randint(0, len(data) - 512)
    batch.append(data[start:start+512])       # Python list append
batch = torch.tensor(batch)

# What Python does for EACH iteration:
# 1. Check if i < 64 (comparison)
# 2. Call random.randint (Python function overhead)
# 3. Slice data[start:start+512] (type check, bounds check, create new object)
# 4. Call batch.append (method lookup, type check, resize list if needed)
# 5. Loop overhead

# Hundreds of Python operations for something NumPy does in one call. Ouch.
```

---

### The Fix (Vectorized)

```python
# Generate all 64 random starts at once
starts = np.random.randint(0, len(data) - 512, size=64)
# ↳ ONE call to C code that generates 64 random numbers

# Create all sequences at once
batch = np.stack([data[s:s+512] for s in starts])
# ↳ Still a comprehension, but NumPy's stack is optimized

# Zero-copy conversion to PyTorch
batch = torch.from_numpy(batch)
# ↳ Shares memory, no copying
```

---

### SIMD: The Hardware You're Not Using

Modern CPUs have **SIMD** units that process multiple values in one clock cycle:

```
Without SIMD (your Python loop):
Clock 1: a[0] + b[0] = result[0]
Clock 2: a[1] + b[1] = result[1]
Clock 3: a[2] + b[2] = result[2]
Clock 4: a[3] + b[3] = result[3]
→ 4 clock cycles for 4 additions

With SIMD (NumPy/PyTorch):
Clock 1: [a[0], a[1], a[2], a[3]] + [b[0], b[1], b[2], b[3]] = [r[0], r[1], r[2], r[3]]
→ 1 clock cycle for 4 additions

NumPy uses SIMD automatically.
Python loops cannot.
```

---

### Performance Comparison

| Batch Size | Python Loop | Vectorized | Speedup |
|------------|-------------|------------|---------|
| 16 | ~1.2 ms | ~0.02 ms | 60x |
| 32 | ~2.4 ms | ~0.03 ms | 80x |
| 64 | ~4.8 ms | ~0.05 ms | 96x |
| 128 | ~9.6 ms | ~0.08 ms | 120x |

The speedup *increases* with batch size because vectorization amortizes overhead.

---

### Quick Reference: Common Vectorization Patterns

```python
# ❌ SLOW                            ✅ FAST

# Sum elements
total = 0
for x in arr:                    →   total = np.sum(arr)
    total += x

# Count positives
count = 0
for x in arr:                    →   count = np.sum(arr > 0)
    if x > 0:
        count += 1

# Element-wise operation
for i in range(len(arr)):        →   result = arr * 2 + 1
    result[i] = arr[i] * 2 + 1

# Random sampling
samples = []
for _ in range(n):               →   samples = np.random.choice(arr, n)
    samples.append(random.choice(arr))
```

---

## 6. Flash Attention

**What it is:** A memory-efficient attention algorithm that gave us 2x speedup by being smarter about memory access.

**Impact:** 0.225s/step → 0.1s/step. The final piece of the 16x puzzle.

---

### The Problem: Standard Attention Is Memory-Hungry

```
Standard Attention computes this matrix:

Input: Q, K, V each of shape [batch, heads, seq_len, head_dim]
       [64, 12, 512, 64]

Step 1: Compute attention scores
        scores = Q @ K.T  →  shape: [64, 12, 512, 512]
                                              ↑    ↑
                                         seq × seq = O(n²)

Step 2: Store this ENTIRE matrix in GPU memory
        512 × 512 × 64 batches × 12 heads × 4 bytes = ~800 MB!
        Just for attention scores!
        In ONE layer!
        You have 12 layers!

Step 3: Out of memory
```

**The quadratic scaling problem:**
```
seq_len=512:   ~800 MB   ← Our model
seq_len=1024:  ~3.2 GB   ← Getting tight
seq_len=2048:  ~12.8 GB  ← Hope you have a big GPU
seq_len=4096:  ~51 GB    ← Exceeds A100 VRAM
seq_len=8192:  ~200 GB   ← Good luck
```

---

### Flash Attention: Tile-Based Processing

Instead of materializing the full attention matrix, Flash Attention processes it in **tiles**:

```
Standard Attention (stores full matrix):
┌─────────────────────────────────────────────┐
│                                             │
│          Full 512×512 attention             │
│              matrix in VRAM                 │
│                                             │
│            ~800 MB sitting there            │
│                doing nothing                │
│                                             │
└─────────────────────────────────────────────┘

Flash Attention (tiles):
┌───────┐
│ tile  │  Process tile 1 → output chunk 1 → FORGET tile 1
│  1    │
└───────┘
         ┌───────┐
         │ tile  │  Process tile 2 → output chunk 2 → FORGET tile 2
         │  2    │
         └───────┘
                  ┌───────┐
                  │ tile  │  Process tile 3 → ...
                  │  3    │
                  └───────┘

Each tile: ~few MB
Total: ~10-20 MB instead of ~800 MB
Savings: 97%+
```

---

### Why It's Fast: GPU Memory Hierarchy

The GPU has different types of memory, and they are NOT created equal:

```
GPU Memory Hierarchy:

┌─────────────────────────────────────────────────────────┐
│ SRAM (on-chip)     │ ~20 MB   │ 19 TB/s  │ Fastest     │
├────────────────────┼──────────┼──────────┼─────────────┤
│ HBM (VRAM)         │ 40-80 GB │ 2 TB/s   │ 10x slower  │
├────────────────────┼──────────┼──────────┼─────────────┤
│ System RAM         │ 64+ GB   │ 50 GB/s  │ 40x slower  │
└─────────────────────────────────────────────────────────┘

SRAM: On-chip, very fast
HBM: Off-chip VRAM (what nvidia-smi shows)
System RAM: Across the PCIe bus

Flash Attention keeps tiles in SRAM.
Standard attention writes everything to HBM.
```

**Why this matters:**
```
Standard Attention:
  Q, K, V in HBM → Compute attention → Store 512×512 in HBM
                                       ↑
                              Slow write (2 TB/s)

Flash Attention:
  Q, K, V in HBM → Load tile to SRAM → Compute in SRAM → Next tile
                           ↑                    ↑
                   Fast (19 TB/s)    Never writes big matrix to HBM!

10x bandwidth difference = massive speedup
```

---

### Enabling Flash Attention (It's Built Into PyTorch)

```python
import torch.nn.functional as F

# Old way (stores full attention matrix):
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
scores = scores.masked_fill(mask == 0, float('-inf'))
attn_weights = F.softmax(scores, dim=-1)
output = torch.matmul(attn_weights, V)

# New way (Flash Attention, automatic):
output = F.scaled_dot_product_attention(
    Q, K, V,
    attn_mask=mask,
    is_causal=True  # For decoder-only models like GPT
)

# PyTorch automatically picks the best backend:
# - Flash Attention (if your GPU supports it)
# - Memory-Efficient Attention (fallback)
# - Standard math (last resort)
```

---

### Requirements

| Requirement | Details |
|-------------|---------|
| PyTorch version | 2.0+ |
| GPU | NVIDIA Ampere or newer (A100, 3090, 4090) |
| Head dimension | ≤ 128 and divisible by 8 |
| Dtype | FP16 or BF16 (not FP32) |

---

### Memory Savings (The Real Numbers)

```
Our model (batch=64, seq=512, heads=12):

Standard Attention:
┌────────────────────────────────────────────┐
│ Attention matrix: 64 × 12 × 512 × 512 × 2  │
│                 = 402 MB (FP16)            │
│ Stored for backward: another 402 MB        │
│ TOTAL: ~800 MB just for attention!         │
└────────────────────────────────────────────┘

Flash Attention:
┌────────────────────────────────────────────┐
│ Tile buffer: ~few MB                       │
│ Stored for backward: O(seq_len) = ~few MB  │
│ TOTAL: ~10-20 MB for attention             │
└────────────────────────────────────────────┘

Savings: ~780 MB per layer
What to do with that headroom: larger batches or longer sequences.
```

---

## 7. Training Stability Techniques

These don't speed up individual steps, but they help you reach lower loss in fewer epochs. Convergence matters.

---

### Learning Rate: The Most Important Number

The **learning rate (LR)** controls how much to adjust weights each step. It's the single most important hyperparameter. Get it wrong and nothing else matters.

```
Weight update formula:
  new_weight = old_weight - learning_rate × gradient

Example:
  old_weight = 0.5
  gradient = 0.1

  With lr = 0.01:  new_weight = 0.5 - 0.01 × 0.1 = 0.499  (tiny step)
  With lr = 1.0:   new_weight = 0.5 - 1.0 × 0.1  = 0.4    (big step)
```

---

### The LR Goldilocks Problem

**LR Too High (lr=1.0) — Overshooting**
```
Loss
  │
  │    ·                         ·
  │   · ·                       · ·
  │  ·   ·       ·   ·         ·   ·
  │ ·     ·     · · · ·       ·     ·
  │·  ①    ·   ·       ·     ·
  │         · ·    ③    ·   ·
  │          ②           · ·
  │                       ④        ← BOUNCING FOREVER
  └─────────────────────────────────
        ① → ② → ③ → ④ → ∞

Loss bounces around forever. Training goes nowhere.
```

**LR Just Right (lr=3e-4) — Smooth Convergence**
```
Loss
  │
  │    ·                         ·
  │   · ·                       · ·
  │  ·   ·                     ·   ·
  │ ·     ·                   ·     ·
  │·  ①    ·                 ·
  │    ↘    ·               ·
  │     ②   ·             ·
  │      ↘    ·           ·
  │       ③    ·         ·
  │        ↘    ·       ·
  │         ④   ·     ·
  │          ↘   ·   ·
  │           ⑤  · ·
  │            ↘  ★              ★ = Minimum reached!
  └─────────────────────────────────
        ① → ② → ③ → ④ → ⑤ → ★
```

**LR Too Low (lr=1e-7) — Slow Progress**
```
Loss
  │
  │    ·                         ·
  │   · ·                       · ·
  │  ·   ·                     ·   ·
  │ ·     ·                   ·     ·
  │·  ①    ·                 ·
  │    ②   ·                ·
  │    ③    ·              ·           ← After 1000 steps...
  │    ④     ·            ·               still barely moved
  │    ⑤      ·          ·
  │    ...     ·   ★    ·              ★ = You'll get here eventually
  └─────────────────────────────────       (in 100,000 steps)
```

---

### Common Learning Rates by Model Size

| Model Size | Typical LR | Notes |
|------------|-----------|-------|
| GPT-2 Small (124M) | 3e-4 to 6e-4 | Can handle higher LR |
| GPT-2 Medium (355M) | 1e-4 to 3e-4 | |
| GPT-2 Large (774M) | 1e-4 to 2e-4 | |
| GPT-2 XL (1.5B) | 5e-5 to 1e-4 | Larger = need lower LR |
| GPT-3 (175B) | 0.6e-4 | Very low for stability |

**Rule of thumb:** Bigger models need smaller learning rates. More parameters = more ways for things to go wrong.

---

### Diagnosing LR Problems From Your Loss Curve

```
LR too high:                    LR too low:                   LR just right:
Loss                            Loss                          Loss
  │  ╱╲  ╱╲                       │█                            │█
  │ ╱  ╲╱  ╲                      │ █                           │ █
  │╱        ╲  ╱                  │  █                          │  ██
  │          ╲╱                   │   █                         │    ███
  │              ╱╲               │    █                        │       █████
  │             ╱  ╲              │     █████████████████       │            ████████
  └─────────────────→             └─────────────────────→        └─────────────────────→

  Oscillating wildly            Flat for too long           Smooth descent
  → Reduce LR by 2-10x          → Increase LR by 2-10x      → Good LR choice

Loss = NaN or Inf:
  → LR is WAY too high
  → Reduce by 10x immediately
  → Check for exploding gradients
```

---

### Cosine Decay: Start High, End Low

**What it is:** Gradually decrease learning rate following a cosine curve.

```
Constant LR (bad):
lr |████████████████████████████████████████
   +----------------------------------------→ Steps
   Same LR entire time → overshoots at the end

Cosine Decay (good):
lr |████████████
   |        ████████
   |                ████████
   |                        ████████
   |                                ████
   +----------------------------------------→ Steps
   High LR early (explore fast)
   Low LR late (fine-tune carefully)
```

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=total_steps,  # Total training steps
    eta_min=1e-5        # Minimum LR at end
)

# In training loop:
for batch in dataloader:
    # ... training code ...
    scheduler.step()  # Update LR after each step
```

---

### Warmup: Don't Shock The Model

**What it is:** Start with tiny LR, ramp up to target LR over first N steps.

```
Without warmup:
lr |████████████████████████████████████████
   +----------------------------------------→ Steps
   Full LR from step 1 → unstable, may explode

With warmup (1000 steps):
lr |    ████████████████████████████████████
   |   █
   |  █
   | █
   |█
   +----------------------------------------→ Steps
   Gradual ramp-up → stable training
```

**Why it matters:** At step 1, your weights are random garbage. The gradients are huge and chaotic. Hitting them with full LR is like flooring the gas pedal on ice. Warmup lets things stabilize first.

---

### Gradient Accumulation: Simulating Larger Batches

**What it is:** Simulate larger batch sizes without using more VRAM.

```
Normal training (batch_size=64):
Forward(64) → Backward → Update weights
Forward(64) → Backward → Update weights
Forward(64) → Backward → Update weights
= 3 weight updates, 192 samples

Gradient Accumulation (batch=64, accumulate=4):
Forward(64) → Backward → accumulate
Forward(64) → Backward → accumulate
Forward(64) → Backward → accumulate
Forward(64) → Backward → UPDATE (÷4)
= 1 weight update, 256 samples (same memory as batch=64!)
```

**Why it matters:**
- Larger effective batches = smoother gradients = better convergence
- batch_size=256 would OOM
- Accumulate 4 batches of 64 = same gradient quality, fits in memory

```python
accumulation_steps = 4
optimizer.zero_grad()

for step, batch in enumerate(dataloader):
    with autocast():
        loss = model(batch)
        loss = loss / accumulation_steps  # IMPORTANT: scale loss

    scaler.scale(loss).backward()

    if (step + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

---

## Quick Reference: The Complete Setup

```python
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler

# 1. Load data into RAM (not mmap!)
data = np.fromfile('tokens.bin', dtype=np.int16)

# 2. Compile model
model = TransformerLanguageModel(...)
model = torch.compile(model)

# 3. Setup AMP
scaler = GradScaler()

# 4. Training loop
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()

        with autocast():
            loss = model(batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

---

## The Journey: 1.6s → 0.1s

| Optimization | Speed | Cumulative Speedup |
|--------------|-------|-------------------|
| Baseline (mmap) | 1.6s/step | 1x |
| RAM preload | 0.8s/step | 2x |
| Vectorized batching | 0.5s/step | 3.2x |
| + torch.compile | 0.35s/step | 4.6x |
| + AMP | 0.225s/step | 7.1x |
| + Flash Attention | **0.1s/step** | **16x** |

**16x speedup. ~130 hours of training. ~105 of those hours were spent on unoptimized code.**

Don't be me. Apply these optimizations before you start training, not on epoch 4.

---

*Deep dive from the GPT-2 training journey, February 2026. May your loss curves descend smoothly.*
