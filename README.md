

<div style="margin-top:50px; margin-left: 12%;">
  <h1 style="font-size: 30px; margin: 0;"> Dr.LLM: Dynamic Layer Routing in LLMs</h1>
</div>
<div align="center">

🚨 **IMPORTANT NOTICE — CODE RELEASE STATUS** 🚨  
> 🧩 The **training**, **data generation**, and **in-domain evaluation** code for **Dr.LLM** are **not yet released**.  
> These components (MCTS supervision, router training scripts, and lm-eval integration) will be made public in an upcoming update.  
> Please stay tuned for the full release!

</div>

---



<div align="left" style="margin:24px 0;">
  <img src="https://user-images.githubusercontent.com/74038190/212284115-f47cd8ff-2ffb-4b04-b5bf-4d1c14c0247f.gif"
       width="100%" height="4"/>
</div>

<p align="center">
  <a href="https://arxiv.org/pdf/2510.12773"><img src="https://img.shields.io/badge/arXiv-2510.12773-b31b1b.svg" alt="arXiv"></a>
</p>

<p align="center">
  <a href="https://www.linkedin.com/in/ahmed-heakl/"><b>Ahmed Heakl</b></a>, 
  <a href="https://scholar.google.com/citations?user=Jt4OYwMAAAAJ&hl=fr"><b>Martin Gubri</b></a>, 
  <a href="https://scholar.google.com/citations?user=M59O9lkAAAAJ&hl=en"><b>Salman Khan</b></a>, 
  <a href="https://scholar.google.com/citations?user=o0qtjzYAAAAJ&hl=en"><b>Sangdoo Yun</b></a>,
   <a href="https://seongjoonoh.com/"><b>Seong Joon Oh</b></a>,
</p>


<p align="center">
  <b>Parameter Lab</b> · <b>MBZUAI</b> · <b>NAVER AI Lab</b> · <b>University of Tübingen</b> · <b>Tübingen AI Center</b>
</p>

---

## 🆕 Latest Updates
- 📢 **15 October 2025**: Paper ArXived!

## 📘 Table of Contents
- [Overview](#overview)
- [🧪 Evaluation](#-evaluation)
  - [In-Domain (Training & Evaluation Tasks)](#in-domain-training--evaluation-tasks)
  - [Out-of-Domain (Generalization Benchmarks)](#out-of-domain-generalization-benchmarks)
- [📊 Results Summary](#-results-summary)
- [⚙️ Usage](#️-usage)
  - [Installation](#1️⃣-installation)
  - [Training the Routers](#2️⃣-training-the-routers)
  - [Evaluation with lm-eval-harness](#3️⃣-evaluation-with-lm-eval-harness)
- [🧭 Citation](#-citation)


## Overview

<table>
<tr>
<td width="30%">
<img src="assets/teaser.png" width="100%">
</td>
<td width="70%">
Large Language Models (LLMs) process every token through all layers of a transformer stack, causing wasted computation on simple queries and insufficient flexibility for harder ones that need deeper reasoning. Adaptive-depth methods can improve efficiency, but prior approaches rely on costly inference-time search, architectural changes, or large-scale retraining, and in practice often degrade accuracy
despite efficiency gains. We introduce Dr.LLM, Dynamic routing of Layers for LLMs, a retrofittable framework that equips pretrained models with lightweight per-layer routers deciding to skip, execute, or repeat a block. Routers are trained with explicit supervision: using Monte Carlo Tree Search (MCTS), we derive high-quality layer configurations that preserve or improve accuracy under a compute budget. Our design, windowed pooling for stable routing, focal loss with class balancing, and bottleneck MLP routers, ensures robustness under class imbalance and long sequences. 

On ARC (logic) and DART (math), Dr.LLM improves accuracy by up to +3.4%p while saving 5 layers per example on average.
Routers generalize to out-of-domain tasks (MMLU, GSM8k, AIME, TruthfulQA, SQuADv2, GPQA, PIQA, AGIEval) with only 0.85% accuracy drop while retaining efficiency, and outperform prior routing methods by up to +7.7%p. Overall, Dr.LLM shows that explicitly supervised routers retrofit frozen LLMs for budgetaware, accuracy-driven inference without altering base weights.
</td>
</tr>
</table>


## 🧪 Evaluation

We evaluate **Dr.LLM** using [`lm-eval-harness`](https://github.com/EleutherAI/lm-evaluation-harness) across both **in-domain reasoning tasks** and **out-of-domain (OOD)** benchmarks.

### In-Domain (Training & Evaluation Tasks)

| Dataset                | Domain          | Metric   | Purpose                              |
| ---------------------- | --------------- | -------- | ------------------------------------ |
| **ARC-Easy/Challenge** | Logic reasoning | Accuracy | Test structured reasoning depth      |
| **DART (levels 1–5)**  | Math reasoning  | Accuracy | Test iterative, multi-step reasoning |

Dr.LLM routers are trained on 4K MCTS-derived execution paths from these datasets.


During inference, layer routing decisions are applied *per input sequence*, adding negligible overhead and remaining KV-cache compatible.

---

### Out-of-Domain (Generalization Benchmarks)

We evaluate zero-shot transfer on:

> **MMLU**, **GSM8k**, **AIME24**, **TruthfulQA**, **GPQA Diamond**, **SQuADv2**, **PIQA**, and **AGIEval**.

Dr.LLM achieves **only −0.85%p average accuracy drop** while maintaining efficiency across these unseen datasets — showing strong generalization.

---

## 📊 Results Summary

| Model             | Domain   | Δ Accuracy | Layers Saved |
| ----------------- | -------- | ---------- | ------------ |
| LLaMA-3B-Instruct | ARC+DART | **+2.7%p** | −7.4         |
| LLaMA-8B-Instruct | ARC+DART | **+2.3%p** | −8.7         |
| LLaMA-3B-Base     | ARC+DART | **+3.2%p** | −3.0         |
| LLaMA-8B-Base     | ARC+DART | **+2.4%p** | −4.2         |
| Qwen-3B-Instruct  | ARC+DART | **+2.3%p** | −3.3         |
| Qwen-7B-Instruct  | ARC+DART | **+0.9%p** | −3.4         |

> 🧠 Routers improve reasoning-heavy tasks by **up to +4.0%p accuracy** while skipping **5 layers per example** on average.

Compared to prior adaptive-depth methods (e.g., LayerSkip, FlexiDepth, MindSkip), **Dr.LLM**:

* Trains on only **4K MCTS paths** (vs 300K+ examples),
* Requires **no finetuning or base weight modification**,
* Outperforms SoTA methods by up to **+7.7%p accuracy**.

---

## ⚙️ Usage

### 1️⃣ Installation

```bash
git clone https://github.com/parameterlab/dr-llm
cd dr-llm
pip install -r requirements.txt
```

<details>
<summary><b>2️⃣ Training the Routers </b></summary>

<!-- add a warning that the full code is not released yet -->
> ⚠️ Note: Full code release is pending ⚠️

Training uses **AdamW**, 25 epochs, **1×10⁻³ LR**, **bf16 precision**, and a **single A100 GPU (40GB)** — taking under 4 hours.


Models source code must be manipulated to insert routers after each transformer block. 

Routers are trained separately using MCTS-generated supervision:

```bash
python train.py \
  --model llama-3-8b-instruct \
  --data_dir data/arc_dart \
  --save_dir checkpoints/drllm_router
```
</details>


### 3️⃣ Evaluation with lm-eval-harness

```bash
lm_eval \
  --model openai/llama-3-8b-instruct \
  --tasks arc_challenge,dart,gsm8k,mmlu \
  --device cuda
```


---


## 🧭 Citation

If you find this work useful, please cite:

```bibtex
@article{heakl2025drllm,
  title={Dr.LLM: Dynamic Layer Routing in LLMs},
  author={Ahmed Heakl and Martin Gubri and Salman Khan and Sangdoo Yun and Seong Joon Oh},
  journal={arXiv preprint arXiv:2510.12773},
  year={2025}
}
```

