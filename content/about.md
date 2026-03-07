---
title: "About"
layout: "single"
---

<div style="text-align: center; margin-bottom: 2em;">
  <img src="/images/profile.png" alt="Jun Park" style="width: 180px; height: 180px; border-radius: 50%; object-fit: cover;">
</div>

## The Short Version

I'm Jun Park — R&D Manager in the life sciences with a background in Immunology and Biomaterials. I've spent 20 years developing and launching life science products from concept to market, and written more application notes and user guides than I care to count.

Then I got curious about how language models actually work. Not "read a blog post" curious — "build one from scratch and see what breaks" curious.

This blog is the result.

## Why I'm Doing This

There's no shortage of tutorials that walk you through importing a pretrained model and calling `.generate()`. That's useful, but it never answered the questions that kept bugging me: What's actually happening inside the transformer? Why does training loss behave the way it does? What does it feel like to watch 2.8 billion tokens flow through 134 million parameters you configured yourself?

So I decided to find out. I started with GPT-2 — trained it from scratch on Google Colab, fought with tokenizers, burned through GPU credits, and documented every mistake along the way.

Then I trained a 1 billion parameter Llama-style model from scratch on a single A100 for $175 total. 90,000 steps, 30 billion tokens, final loss 2.494. It scores 47.1% on ARC-Easy and 28.8% on HellaSwag — not bad for a model that cost less than a nice dinner.

## What You'll Find Here

Honest documentation of training language models from scratch — the kind of stuff that doesn't make it into research papers. Real cost breakdowns ($175 for 1B parameters). Actual loss curves. The 11 things that went wrong before anything went right. Training optimizations that took my run from 90 minutes to 21 minutes. And the infrastructure lessons that nobody talks about.

No hand-waving, no "left as an exercise for the reader."

## The Unusual Background

I know what you're thinking: why is a life scientist doing this?

Fair question. But 20 years of experimental science — designing assays, troubleshooting failed experiments, staring at noisy data from cell-based readouts — teaches you things that transfer surprisingly well to ML. How to design controlled experiments. How to troubleshoot when something doesn't replicate. How to separate signal from noise. And above all — how to be honest about what worked and what didn't.

That's what this blog is about.

## Find Me Elsewhere

- **Email**: [jun@gpuburnout.com](mailto:jun@gpuburnout.com)
- **Blog Source**: [github.com/GPUburnout/llm-journey](https://github.com/GPUburnout/llm-journey)
- **GPT-2 Training Code**: [github.com/GPUburnout/gpt2-from-scratch](https://github.com/GPUburnout/gpt2-from-scratch)
- **GitHub**: [github.com/GPUburnout](https://github.com/GPUburnout)
- **LinkedIn**: [linkedin.com/in/jun-park-b83178a5](https://www.linkedin.com/in/jun-park-b83178a5/)
