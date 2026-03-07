---
title: "Data Preparation: Building a 12GB Training Corpus"
date: 2026-01-22
draft: false
tags: ["data", "preprocessing", "tokenization", "BPE", "scaling"]
summary: "How I built a 12GB ChatGPT-style conversational dataset and implemented BPE tokenization for efficient training."
weight: 2
season: 1
chapter: 2
---

You know that saying "garbage in, garbage out"? Turns out it applies to language models too. Who knew.

Before I could train GPT-2, I needed data. A lot of it. Like, "my internet provider called to ask if I was okay" amounts of data.

## What I Ended Up With

| Attribute | Value |
|-----------|-------|
| **Final Size** | ~12GB of raw text |
| **Format** | ChatGPT-style conversations |
| **Tokens** | ~2.8 billion (yes, billion with a B) |
| **Binary Size** | 11.58 GB — because apparently text needs to be even bigger |
| **Compressed** | ~4 GB — compression is magic |

---

## The Five Stages of Data Preparation

### Stage 1: Finding Data (Hope)

"I'll just download some conversational data, should take an hour tops."

*Narrator: It did not take an hour.*

Turns out finding quality conversational data at scale is like finding a parking spot downtown — technically possible, but you'll lose your mind trying.

### Stage 2: Cleaning Data (Denial)

The raw data was... special. Duplicates everywhere. Encoding issues that made Unicode cry. Malformed text that looked like someone smashed their keyboard.

Built `dataset_cleaner.py` with:
- Hash-based deduplication (goodbye, copy-pasted content)
- UTF-8 normalization (hello, sanity)
- Minimum length filtering (sorry, one-word replies)
- Format standardization (because consistency is a myth)

### Stage 3: BPE Tokenization (Bargaining)

**The problem:** Training a BPE tokenizer on 12GB of text will eat your RAM for breakfast.

**The solution:** HuggingFace's `tokenizers` library, written in Rust because Python is too slow for adults.

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# Streams the file instead of loading 12GB into RAM like a maniac
tokenizer.train(files=["dataset.txt"], trainer=trainer)
```

**Result:** 32K vocab tokenizer trained in 30 minutes. On my laptop. While I got coffee. The future is wild.

### Stage 4: Binary Conversion (Depression)

Converting 12GB of text to binary tokens. Sounds simple. Is not simple.

The process:
1. Stream through file line by line (because loading it all crashes everything)
2. Tokenize in chunks
3. Write int32 tokens to binary
4. Pray

```bash
python tokenize_local_bpe.py \
    --input dataset_chatgpt_style_10GB.txt \
    --vocab_size 32000 \
    --compress  # Trust me, you want this flag
```

**Output:**
- `tokens_bpe.bin` (11.58 GB) — the chonky boy
- `tokens_bpe.bin.gz` (~4 GB) — the reasonable boy
- `bpe_tokenizer.json` — the important boy
- `tokens_bpe_metadata.json` — the forgotten boy

### Stage 5: Uploading to Colab (Acceptance)

**Problem:** Getting 11GB to Google Colab without dying.

**Things that didn't work:**
- Direct upload — 4 hours, then timeout. Cool.
- Google Drive sync — "Syncing..." forever. Very helpful.

**What actually worked:**
Upload the compressed version, decompress on Colab:

```python
!gunzip /content/drive/MyDrive/tokens_bpe.bin.gz -c > /content/tokens_bpe.bin
```

5 minutes to decompress. Worth the compression time. Always compress.

---

## The Pipeline (For Fellow Masochists)

```
Raw Text (12GB)
    ↓
[dataset_cleaner.py] — Remove the garbage
    ↓
Cleaned Text (10GB) — Still a lot of garbage, but cleaner garbage
    ↓
[tokenize_local_bpe.py] — Turn words into numbers
    ↓
tokens_bpe.bin (11.58GB) + bpe_tokenizer.json
    ↓
[gzip] — Squish it real good
    ↓
tokens_bpe.bin.gz (4GB) — Upload this, not the big one
    ↓
[Colab: gunzip] — Unsquish it
    ↓
Ready to train (finally)
```

---

## Time Investment (Read: Time Wasted)

| Task | Time | Pain Level |
|------|------|------------|
| Data sourcing | ~4-6 hrs | Medium |
| Cleaning pipeline | ~2-3 hrs | High |
| BPE tokenizer training | ~30 min | Low (Rust is fast) |
| Binary conversion | ~1-2 hrs | Medium |
| Compression | ~30 min | Low |
| Upload to Drive | ~2-3 hrs | Existential 💀 |
| **Total** | **~10-15 hrs** | **Worth it?** |

---

## Hard-Won Wisdom

1. **Quality over quantity.** 10GB of clean data beats 20GB of garbage.
2. **BPE is non-negotiable.** Character-level tokenization is for people who enjoy suffering.
3. **Always compress before uploading.** 4GB uploads 3x faster than 11GB. Math.
4. **Stream everything.** If you try to load 12GB into RAM, you deserve what happens.
5. **Save intermediate files.** Re-running a 2-hour preprocessing step because you forgot to save is a special kind of pain.
6. **Use Rust-based tools.** Pure Python tokenizers are cute. Rust tokenizers are fast.
