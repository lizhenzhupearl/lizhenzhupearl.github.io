---
title: "RC Gate or Sigmoid Gate — Which Would You Choose to Train a Transformer?"
date: 2025-02-13
permalink: /posts/2025/02/rc-vs-sigmoid-bitter-lesson/
excerpt: "A story about a beautiful idea that didn't work, and what I learned from killing it."
tags:
  - machine learning
  - transformers
  - research
---

*A story about a beautiful idea that didn't work, and what I learned from killing it.*

*I am not expert in transformer at all, so if I am wrong, please let me know, I need to learn more!*

---

## The Interview Question

Imagine you're in a machine learning interview. The interviewer draws two functions on the whiteboard:

**Sigmoid gate:** `g = σ(MLP(x))`

**RC gate:** `g = 1 − exp(−1/τ)`, where `τ = softplus(MLP(x))`

"Both produce values between 0 and 1," they say. "Both can gate a residual stream in a transformer — multiplying a layer's output before it's added back. Which one would you use, and why?"

If you'd asked me 1 month ago, I would have given a passionate answer about why the RC gate is better. I had a whole theory. I called the project *MyelinFormer*. It was inspired by neuroscience. It was elegant.

It was wrong.

This is the story of how I found out.

---

## The Beautiful Idea

In the brain, myelin is a fatty sheath that wraps around axons. Heavily myelinated pathways conduct signals fast and efficiently — think of a pianist's motor circuits after years of practice. Unmyelinated pathways are slow but flexible — they can still learn, still adapt.

The RC circuit (resistor-capacitor circuit) is the standard physics model for how electrical signals propagate through myelinated axons. The time constant τ controls everything: low τ means thick myelin, fast conduction, established pathways. High τ means thin myelin, slow conduction, still-learning pathways.

So I thought: what if we use RC circuits as gates in a transformer? Instead of a sigmoid deciding how much of each layer's output to keep, we'd have τ — a learnable time constant — controlling the flow. Layers with low τ would become "myelinated": efficient, fast, committed. Layers with high τ would stay "plastic": still learning, still adapting.

The gate function `g = 1 − exp(−1/τ)` maps directly from RC circuit physics. When τ is small, g approaches 1 (full signal). When τ is large, g approaches 0 (signal suppressed). The biology-to-math translation was clean. I was excited.

---

## What I Expected to Find

My hypothesis was that RC gates would produce qualitatively different computation structures compared to sigmoid gates. Specifically:

1. **Bimodal layer organization** — some layers fully myelinated, others fully plastic, with a clean separation (not a mush of intermediate values).
2. **Reversible decisions** — a "myelinated" layer could demyelinate if the task changed, because the gradient through τ stays healthy.
3. **Better efficiency** — the model would learn to skip unnecessary layers more cleanly.

I set up careful experiments. Same transformer architecture, same data, same everything — just RC gate vs sigmoid gate vs no gate. Text8 character-level language modeling. MNIST classification. Multiple seeds, statistical tests, the works.

---

## What Actually Happened

**Finding 1: RC does produce sharper bimodal distributions.** This was real. RC layers clearly separated into "myelinated" (g ≈ 0.05) and "active" (g ≈ 1.0). Sigmoid was more unimodal. Point for RC.

**Finding 2: Sigmoid outperforms RC on actual task performance.** On BPC (bits per character), sigmoid gates consistently beat RC gates. Not by a huge margin, but decisively and across all seeds. The bimodal structure didn't translate to better language modeling.

**Finding 3: The gradient analysis killed the story.** This is where it got painful.

I did the math on gradient flow through each gate. The RC gate composes two saturating nonlinearities: softplus (which saturates for large inputs) and exp (which saturates for small inputs). This dual saturation compresses the region of input space where the gate is actually sensitive to changes. I called it "compressed dynamic range."

For a sigmoid gate at g ≈ 0.05, the gradient ∂g/∂logit = g(1−g) ≈ 0.0475. That's healthy. The gate can still move.

For an RC gate at g ≈ 0.05, you need τ ≈ 19.5, and the gradient ∂g/∂τ = −exp(−1/τ)/τ² is tiny in magnitude. The gate is stuck.

I measured per-token routing precision: how much does the gate value change between easy and hard inputs for the same layer? Sigmoid showed 41.7× dynamic range. RC was far less discriminating — it was suppressing everything about equally, not making sharp per-token decisions.

**The biological analogy had led me astray.** In biology, RC dynamics work because axons operate in continuous time with physical constraints. A transformer's residual stream is discrete, spatial, and operates in a completely different optimization landscape. The properties that make RC circuits good for neural signal propagation simply don't transfer.

---

## The Pivot Attempts (And Why They Failed Too)

When the direct comparison failed, I tried reframing. Maybe RC's "weakness" — its sticky, compressed gradient — was actually a *feature* for continual learning? If you don't want to forget, maybe you want gates that resist change.

I imagined a two-stage approach: pretrain with sigmoid (learn fast), then switch to RC for fine-tuning (don't forget). The compressed gradient would act as natural, per-layer regularization. Some layers could decide to adapt slowly (protecting old knowledge) while others adapted faster (learning new tasks).

I did a thorough literature search. The dual-speed learning idea (fast plasticity + slow stability) already exists. Per-layer adaptive plasticity already exists. Even activation-function-as-plasticity-controller already exists (AdaLin).

But then I hit the fatal flaw in my own logic. For memory protection during fine-tuning, you need layers that learned Task A to resist change when training on Task B. RC achieves this because the gate gradient is compressed — the gate can't move. But sigmoid achieves the *same protection through a different mechanism*: when a sigmoid gate is at 0.05, the layer's output is multiplied by 0.05 before entering the residual stream. The gradient flowing back into that layer's weights is also attenuated by 95%. The layer is naturally protected — not because the gate is stuck, but because the gate value itself attenuates the learning signal.

And sigmoid's mechanism is *better*, because the gate itself remains flexible (gradient ≈ 0.0475), so if a new task genuinely needs that layer, sigmoid can reopen it. RC gives you "stuck." Sigmoid gives you "protected but recoverable." For continual learning, you want the latter.

I was back to square one.

---

## What I Actually Learned

### 1. Biological analogies are seductive but dangerous

The myelin → RC circuit → transformer gate pipeline *felt* like insight. It had narrative coherence. It made for a great story. But narrative coherence is not the same as mathematical validity. The brain and a transformer share almost no computational substrate. Mapping one onto the other requires justification at every step, and I skipped most of those steps because the story was too compelling.

### 2. Composing saturating functions compresses dynamic range

This is actually a useful general principle. If you chain softplus → exp (or any two saturating nonlinearities), the resulting function has a narrower "sensitive region" than either function alone. This is fine for RNN gates (where you want to commit to remember/forget decisions quickly), but harmful for residual stream gates (where you need per-token flexibility). There's a published paper showing the *opposite direction* — that composed saturation *helps* RNN gates (the "Fast Gate" paper, AAAI 2023). The general lesson: gate function design requirements differ between temporal and spatial contexts.

### 3. The right experiment reveals the truth fast

My most informative analysis was measuring per-token dynamic range for both gates. A single number — 41.7× for sigmoid vs much less for RC — told me more than weeks of training runs. If I'd done that analysis first, time would have been saved.

### 4. Honest negative results teach more than lucky positive ones

I know more about transformer gating dynamics now than I would have if RC gates had happened to work by accident. I understand *why* sigmoid is the standard — not just "it works" but the specific gradient properties that make it work. That understanding transfers to every future project.

### 5. "I invented a medicine, now let me find a disease" is backwards science

The most painful realization. I started with RC gates and went looking for scenarios where they'd shine. That's not how research should work. You start with a problem (catastrophic forgetting, adaptive computation, whatever), survey what's been tried, and then ask whether your tool fits. I reversed the process because I was attached to the tool.

---

## The Answer to the Interview Question

RC gate or sigmoid gate for a transformer?

**Sigmoid. It's not even close.**

Sigmoid gives you clean gradients across the full operating range, sharp per-token routing decisions, the ability to suppress layers almost to zero for efficiency, and the flexibility to reopen those layers when needed. It achieves all of this with a single, well-understood nonlinearity.

RC gates compose two saturating functions, compress the useful gradient region, produce less discriminating per-token routing, and can't reach as low a floor for layer skipping. The bimodal structure they produce is aesthetically pleasing but functionally inferior.

Sometimes the standard tool is standard for a reason.

---

## A Note on Failure

I spent time on this project. I designed experiments, ran hundreds of GPU-hours of training, did careful gradient analysis, reviewed dozens of papers. The result: a conclusive demonstration that my core idea doesn't work.

In academia, this would be a hard outcome. In research culture generally, negative results are undervalued and unpublished. The incentives push you to spin, reframe, find the angle that makes it a positive story.

But I think the honest thing is more valuable. RC gating on transformer residual streams doesn't outperform sigmoid. The biological analogy that motivated it, while intellectually interesting, doesn't survive contact with the optimization landscape. And the cascade of "maybe it works for *this* instead" pivots — continual learning, adaptive plasticity, memory protection — all ran into the same wall: sigmoid already does it better, through simpler mechanisms.

---

*If you're working in academia and it turns out your idea doesn't work? That's fine. You'll learn more from understanding why than you would from a lucky positive result. The understanding is the real output.*
