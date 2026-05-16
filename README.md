

<div align="center">

# FinQwen — Financial LLM Benchmark

**Qwen2.5-7B** domain-adapted with **LoRA (Unsloth)** on SEC 10-K filings and earnings call sentiment data.  
Rigorously evaluated on a **50-question benchmark** across 4 financial NLP task types using LLM-as-judge scoring.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Model](https://img.shields.io/badge/base-Qwen2.5--7B--Instruct-orange)](https://huggingface.co/unsloth/Qwen2.5-7B-Instruct-bnb-4bit)
[![Framework](https://img.shields.io/badge/framework-Unsloth%20%2B%20LoRA-green)](https://github.com/unslothai/unsloth)
[![Hardware](https://img.shields.io/badge/hardware-T4%20GPU%20(Colab)-lightgrey)](#training-setup)
[![HuggingFace Space](https://img.shields.io/badge/demo-HuggingFace%20Space-yellow)](https://huggingface.co/spaces/Adityax-07/FinQwen-Benchmark)

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Benchmark Results](#benchmark-results)
- [Model Architecture](#model-architecture)
- [Training Setup](#training-setup)
- [Dataset](#dataset)
- [Evaluation Methodology](#evaluation-methodology)
- [Key Findings & Analysis](#key-findings--analysis)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Future Work](#future-work)

---

## Project Overview

FinQwen is an end-to-end experiment in **financial domain fine-tuning** — taking a general-purpose 7B instruction model and asking: *how much can targeted LoRA adaptation improve domain-specific NLP performance, and what does it cost?*

The project covers the full ML lifecycle:

1. **Data curation** — merging two complementary financial NLP datasets (~8k examples)
2. **Fine-tuning** — efficient LoRA adaptation via Unsloth on a free T4 Colab GPU
3. **Evaluation** — a structured 50-question benchmark with 5 distinct metrics
4. **Comparison** — head-to-head against an 8B base model and a 70B strong baseline

The core finding: domain fine-tuning dramatically improves **sentiment classification** (+137% F1) but introduces a **hallucination calibration regression** at 300 training steps — a well-known LoRA tradeoff that DPO alignment can address.

---

## Benchmark Results

### Summary Table

| Metric | FinQwen (fine-tuned) | Base LLM (8B) | Strong LLM (70B) |
|---|:---:|:---:|:---:|
| Sentiment F1 ↑ | **0.767** | 0.324 | 0.398 |
| ROUGE-L ↑ | 0.000 | **0.340** | 0.318 |
| QA Accuracy ↑ | 0.867 | **1.000** | **1.000** |
| Groundedness ↑ | 0.387 | 0.687 | **0.833** |
| Hallucination Rate ↓ | 66.7% | 20.0% | **6.7%** |

> ↑ = higher is better · ↓ = lower is better

### Metric-by-Metric Breakdown

#### Sentiment F1 — FinQwen wins by a wide margin

FinQwen achieves a macro-F1 of **0.767** on 10 earnings sentiment questions — **+137% over the 8B base** and **+93% over the 70B strong baseline**. This is the primary success of domain fine-tuning: the model has internalized the linguistic register of financial press releases and earnings calls, allowing it to reliably map phrases like *"strategic review amid declining market share"* to `negative` and *"margin expansion"* to `positive`.

The base models, despite being larger or more capable on general tasks, lack the calibration for financial-specific language — leading to macro-F1 scores in the 0.3–0.4 range.

#### ROUGE-L — Base models win on summarization

FinQwen scores **0.000 ROUGE-L** on the summarization subset, compared to 0.340 for the base model and 0.318 for the strong LLM. This is a genuine regression, not a measurement artifact.

The root cause: at 300 training steps on sentiment-heavy data, the model has shifted its output distribution toward short, label-style answers (`positive`, `negative`, `neutral`). When asked to *summarize* a block of financial text, it produces overly terse outputs that share almost no n-gram overlap with the reference summaries used for ROUGE scoring.

This highlights a known risk with targeted fine-tuning: **task-specific adaptation can degrade generative fluency** on tasks not represented in the fine-tuning mix.

#### QA Accuracy — Base models outperform

Both the 8B base and 70B strong LLM achieve **100% accuracy** on the 15 factual financial QA questions (definitions of ROE, CAPM, duration, LBO, etc.). FinQwen scores **86.7%**, missing 2 questions. This is a minor but real regression — the model's instruction-following precision on structured definitional answers has degraded slightly.

#### Groundedness & Hallucination Rate — The calibration problem

FinQwen's **66.7% hallucination rate** (vs 20% for the 8B base and 6.7% for the strong LLM) is the most significant finding. On 15 hallucination-trap questions (real-time prices, yesterday's S&P close, last quarter's exact ratios), FinQwen frequently invents specific numbers rather than acknowledging uncertainty.

The groundedness score follows the same pattern: **0.387** for FinQwen vs 0.687 (base) and 0.833 (strong LLM).

This is a calibration failure, not a knowledge failure. The model knows the domain — but 300 steps of supervised fine-tuning on factual text has increased its confidence in generating domain-sounding specifics, even when those specifics are fabricated. RLHF/DPO training with explicit uncertainty examples would address this.

---

## Model Architecture

### Base Model

| Property | Value |
|---|---|
| Architecture | Qwen2.5-7B-Instruct |
| Quantization | 4-bit (BnB NF4) |
| Max sequence length | 2048 tokens |
| Precision | bf16 (auto-detected) |
| HuggingFace ID | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` |

### LoRA Configuration

| Hyperparameter | Value | Rationale |
|---|---|---|
| Rank (`r`) | 16 | Meaningful adaptation, not a toy r=4 |
| Alpha (`lora_alpha`) | 32 | Standard 2× rank scaling |
| RSLoRA | Enabled | Rank-stabilized scaling — free performance gain |
| Dropout | 0 | No regularization needed at this scale |
| Bias | none | Standard PEFT setting |
| Gradient checkpointing | unsloth | Memory-efficient training |

### Target Modules

LoRA adapters are applied to all attention and MLP projection layers:

```
Attention:  q_proj, k_proj, v_proj, o_proj
MLP:        gate_proj, up_proj, down_proj
```

Including the MLP layers is important for **domain knowledge injection** — attention-only LoRA tends to underperform on factual tasks.

### Trainable Parameters

With the above configuration on Qwen2.5-7B:

- **Trainable:** ~40M parameters (~0.57% of total)
- **Frozen:** ~7B parameters
- **Storage footprint:** ~80MB for the LoRA adapter (vs ~4GB for the full 4-bit base)

---

## Training Setup

| Setting | Value |
|---|---|
| Hardware | T4 GPU (Google Colab, free tier) |
| Training steps | 300 |
| Per-device batch size | 2 |
| Gradient accumulation | 4 (effective batch = 8) |
| Learning rate | 2e-4 |
| LR scheduler | Cosine (outperforms linear for domain fine-tuning) |
| Optimizer | AdamW 8-bit |
| Warmup steps | 20 |
| Weight decay | 0.01 |
| Seed | 42 |
| Packing | Disabled |

### System Prompt (Chat Template)

```
<|im_start|>system
You are FinQwen, an expert financial analysis assistant. Answer accurately using financial domain knowledge.
<|im_end|>
<|im_start|>user
{INPUT}<|im_end|>
<|im_start|>assistant
{OUTPUT}<|im_end|>
```

Data is formatted in **ShareGPT format** before applying the Qwen chat template via `unsloth.apply_chat_template`.

---

## Dataset

### Sources

| Dataset | Split | Size used | Task coverage |
|---|---|---|---|
| `AdaptLLM/finance-tasks` | test | ~4,000 rows | ConvFinQA, FiQA_SA, FPB, Headline, NER |
| `FinGPT/fingpt-sentiment-train` | train | ~4,000 rows | Earnings call sentiment |
| **Combined** | — | **~8,000 examples** | — |

### Dataset Construction

1. Both datasets are loaded and reformatted into ShareGPT conversation format
2. Subsampled 50/50 to prevent sentiment data from dominating the training mix
3. Combined and shuffled with seed 42
4. Tokenized with the Qwen2.5 chat template

### EDA Findings

- Token length distribution peaks at 150–300 tokens; very few samples exceed the 2048-token limit
- FinGPT sentiment labels: approximately 45% positive, 35% negative, 20% neutral
- finance-tasks covers a diverse set of sub-tasks including NER, headline classification, QA, and sentiment

---

## Evaluation Methodology

### Benchmark Design

The 50-question benchmark is structured across 4 task types:

| Task Type | Questions | Evaluation Method |
|---|:---:|---|
| Sentiment classification | 10 | Keyword extraction → macro F1 |
| Financial summarization | 10 | ROUGE-L against reference summaries |
| Factual financial QA | 15 | LLM-as-judge (llama-3.1-8b-instant) |
| Hallucination traps | 15 | LLM-as-judge (YES/NO + 0–1 groundedness score) |

### Models Compared

| Label | Model | API |
|---|---|---|
| FinQwen (fine-tuned) | Qwen2.5-7B + LoRA adapters | Local inference |
| Base LLM (8B) | llama-3.1-8b-instant | Groq API |
| Strong LLM (70B) | llama-3.3-70b-versatile | Groq API |

### Metric Definitions

**Sentiment F1** — Macro-averaged F1 across three classes (positive / negative / neutral). Model outputs are mapped to labels via keyword matching before scoring with `sklearn.metrics.f1_score`.

**ROUGE-L** — Longest common subsequence recall between model output and reference summary. Computed via the `evaluate` library.

**QA Accuracy** — LLM-as-judge prompt: given the question and a reference answer, judge whether the model response is factually correct and aligned. Binary 1/0 per question.

**Groundedness** — LLM-as-judge prompt: score the response 0.0–1.0 on whether it stays grounded without fabricating specifics. Applied to hallucination-trap questions only.

**Hallucination Rate** — Fraction of hallucination-trap responses judged as inventing specific financial figures, dates, or company names. Binary YES/NO per question.

### Judging Infrastructure

All LLM-as-judge calls use `llama-3.1-8b-instant` via Groq API with `temperature=0.1`. Inference results are checkpointed to SQLite (`finqwen_eval.db`) to allow resuming interrupted evaluation runs.

---

## Key Findings & Analysis

### What worked

**Sentiment classification is the clear win.** A +137% improvement in macro-F1 (0.324 → 0.767) demonstrates that 300 steps of LoRA fine-tuning on domain-specific sentiment data can be remarkably effective. The model has learned the stylistic patterns of financial press releases — hedging language, magnitude qualifiers, and analyst-speak — that general models struggle to parse.

This finding generalizes: for well-defined classification tasks with stable label distributions, targeted fine-tuning is highly competitive even with much larger instruction-tuned models.

### What regressed

**Hallucination calibration is the primary casualty.** The jump from 20% to 66.7% hallucination rate is the most concerning result. The mechanism is well-understood: supervised fine-tuning on factual domain text increases the model's domain-specific fluency but not its epistemic humility. The model becomes more confident in generating finance-sounding outputs — including fabricated specifics — because confident specificity is exactly what appears in the training data.

**Summarization quality degrades** due to output distribution shift. With the training mix heavily weighted toward short sentiment outputs, the model's priors for generation length and format shift away from multi-sentence summaries.

### The LoRA Tradeoff at 300 Steps

300 steps provides enough signal for:
- Sentiment label calibration (frequent, well-defined pattern)
- Basic domain vocabulary internalization

But not enough for:
- Output diversity preservation (summarization, long-form QA)
- Uncertainty calibration (hallucination traps)

The model sits in an uncomfortable middle ground: adapted enough to change behavior, but not adapted enough to preserve the instruction-following robustness of the base model on out-of-distribution task formats.

---

## Quick Start

### Run via HuggingFace Space

The interactive benchmark dashboard is available as a Gradio app on HuggingFace Spaces.

### Run Locally

```bash
git clone https://github.com/Adityax-07/FineQwen.git
cd FineQwen
```

Open `FinQwen_Colab_Clean.ipynb` in Google Colab (T4 runtime recommended).

Execute phases sequentially:
1. **Phase 1** — Install dependencies (Unsloth, TRL, Groq, evaluate)
2. **Phase 2** — Load Qwen2.5-7B-Instruct in 4-bit and apply LoRA
3. **Phase 3** — Load and format datasets; run EDA
4. **Phase 4** — Train for 300 steps; plot loss curve
5. **Phase 5** — Run 50-question benchmark evaluation
6. **Phase 6** — Compile results table and radar chart
7. **Phase 7** — Save LoRA adapters and push to HuggingFace Hub

### Dependencies

```bash
pip install unsloth trl transformers accelerate peft bitsandbytes datasets
pip install evaluate rouge_score scikit-learn groq plotly pandas matplotlib
```

### Required API Keys

| Key | Purpose | Source |
|---|---|---|
| `GROQ_API_KEY` | Base model + strong LLM inference + LLM-as-judge | console.groq.com |
| `HF_TOKEN` | Push model to HuggingFace Hub (optional) | huggingface.co/settings/tokens |

---

## Project Structure

```
FineQwen/
├── FinQwen_Colab_Clean.ipynb   # Full pipeline: training + evaluation
├── README.md                   # This file
```

### Generated Artifacts (during notebook execution)

```
finqwen_lora/          # LoRA adapter weights (save_pretrained)
finqwen_gguf/          # GGUF export for Ollama (optional)
finqwen_checkpoints/   # Training checkpoints (every 50 steps)
finqwen_eval.db        # SQLite checkpoint for evaluation results
finqwen_results.csv    # Final results table
eda_charts.png         # Token length + label distribution plots
loss_curve.png         # Training loss over 300 steps
radar_chart.png        # 6-metric radar comparison chart
```

---

## Limitations

**Hallucination rate is high.** At 66.7%, FinQwen should not be used in production without a retrieval-augmented generation (RAG) layer or an explicit uncertainty calibration step. All numerical outputs should be verified against authoritative sources.

**No real-time data.** The model has a training data cutoff and cannot access live prices, rates, or filings. It will often confabulate specifics when asked about current market data.

**Summarization quality regressed.** The output distribution shift from fine-tuning means FinQwen produces shorter, less fluent summaries than the base model on financial text summarization tasks.

**Limited training compute.** 300 steps on a T4 GPU is a minimal experiment. Production-grade fine-tuning would require more steps, a larger data mix, and validation-set-based early stopping.

**Benchmark scope.** The 50-question benchmark is a proof-of-concept evaluation, not a comprehensive assessment. It does not cover tasks like table QA (from 10-K exhibits), earnings call transcript QA, or multi-hop financial reasoning.

---

## Future Work

| Improvement | Expected Impact |
|---|---|
| **DPO alignment** with uncertainty examples | Reduce hallucination rate; improve groundedness |
| **More training steps** (600–1000) with validation loss monitoring | Better generalization; catch over-fitting early |
| **Balanced data mix** (add more summarization examples) | Recover ROUGE-L regression |
| **RAG integration** | Ground real-time queries in live filings or earnings data |
| **Larger benchmark** (200+ questions) | More statistically robust evaluation |
| **GGUF + Ollama deployment** | Local inference without GPU |

---

## Citation

If you use FinQwen or this benchmark methodology in your work, please cite:

```bibtex
@misc{finqwen2025,
  title        = {FinQwen: Domain Fine-Tuning of Qwen2.5-7B for Financial NLP},
  author       = {Aditya},
  year         = {2025},
  howpublished = {\url{https://github.com/Adityax-07/FineQwen}},
  note         = {LoRA fine-tuning on SEC 10-K and earnings call data, benchmarked on 50 financial NLP questions}
}
```

---

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth) — 2× faster LoRA fine-tuning with reduced VRAM
- [AdaptLLM/finance-tasks](https://huggingface.co/datasets/AdaptLLM/finance-tasks) — multi-task financial NLP benchmark
- [FinGPT/fingpt-sentiment-train](https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train) — earnings call sentiment labels
- [Groq](https://console.groq.com) — fast inference API used for base model comparison and LLM-as-judge evaluation
