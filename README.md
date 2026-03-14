<div align="center">
  <h1>🔎 DeepScan: A Training-Free Framework for Visually Grounded Reasoning in Large Vision-Language Models</h1>
  <p>
    <em>Official implementation of the **CVPR 2026** paper:<br>
    <strong>“DeepScan: A Training-Free Framework for Visually Grounded Reasoning in Large Vision-Language Models”</strong></em>
  </p>

  <p>
    <a href="https://arxiv.org/abs/2603.03857"><img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv%202026-1D4ED8"></a>
    <a href="https://arxiv.org/abs/2603.03857"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2603.03857-B31B1B"></a>
    <a href="#"><img alt="Framework" src="https://img.shields.io/badge/Framework-Training--Free-10B981"></a>
    <a href="#"><img alt="Grounding" src="https://img.shields.io/badge/Grounding-Bottom--Up-111827"></a>
  </p>
</div>

> **TL;DR.** DeepScan is a **training-free** framework for **visually grounded reasoning** in LVLMs. Instead of relying on brittle one-shot, coarse-to-fine localization, it adopts a **bottom-up** pipeline: **Hierarchical Scanning** for cue discovery and evidence recovery, **Refocusing** for context-optimal evidence views, and **Evidence-Enhanced Reasoning** for final answer generation from multi-granular evidence memory. DeepScan achieves **90.6%** on **V\*** with **Qwen2.5-VL-7B**, and scales consistently across LVLM architectures and model sizes.

---

## 🔥 News
- **2026-03.** DeepScan is released as an arXiv preprint: **DeepScan: A Training-Free Framework for Visually Grounded Reasoning in Large Vision-Language Models**.
- **2026-03.** This repository provides a service-oriented implementation of the DeepScan pipeline, including the **search expert**, **visual expert**, and **LVLM** backends used for visually grounded inference.

---

## 👀 Overview
Humans often solve challenging visual problems in a **bottom-up** manner: they first identify subtle local cues, then recover the full evidence from those cues, and finally reason over the recovered evidence. DeepScan is built on the same intuition.

DeepScan contains three tightly coupled stages:

1. **Hierarchical Scanning**
   - Partition the image into local patches.
   - Use a **search expert** to produce patch-wise attention maps.
   - Convert connected cue regions into **point-based proxies** using both **semantic saliency** and **topological interiority**.
   - Recover image-level evidence via **point-prompt segmentation**, followed by morphological post-processing.
   - Retain only the **top-k smallest evidence candidates** for efficient evidence judgment.

2. **Refocusing**
   - Starting from the fused evidence crop, search over a concise set of candidate views.
   - Use **Zoom-In** and **Zoom-Out** actions to calibrate the surrounding context.
   - Select the **smallest view that still fully contains the evidence needed for answering**.

3. **Evidence-Enhanced Reasoning**
   - Build a **Hybrid Evidence Memory** composed of:
     - **fine-grained evidence crops** from Hierarchical Scanning, and
     - a **coarse-grained refined view** from Refocusing.
   - Materialize them as an ordered multi-image prompt for the LVLM.
   - Generate answers that are both **more accurate** and **better grounded** in the visual evidence.

Unlike RL-based visually grounded reasoning methods, DeepScan is **plug-and-play** and **training-free**. It can be integrated with different LVLM backbones without additional adaptation cost.

---

## 🏗 Repository Structure
```text
DeepScan/
├── scripts/
│   ├── blip_server/       # Search-expert service (BLIP-ITM + Grad-CAM attention)
│   ├── expert_server/     # Visual-expert service (LangSAM-based detection)
│   ├── lmm_server/        # LVLM serving scripts (e.g., LLaVA / Qwen backends)
│   ├── sam2_server/       # SAM2 point-prompt segmentation service
│   ├── pope/
│   └── vstar/
├── src/
│   ├── eval.py            # Evaluation script for prediction files
│   ├── qwen_runtime.py    # Local Qwen-based runtime for LVLM querying
│   ├── run.py             # Main evaluation / inference entry point
│   ├── utils.py           # Common utilities
│   └── policies/
│       ├── deepscan.py    # DeepScan policy implementation
│       ├── visual_grounding.py
│       ├── control_point_sam.py
│       ├── client.py
│       ├── mstc.py
│       └── ...
└── README.md
```

---

## 🛠 Preparation

### 1) Clone
```bash
git clone <your-repo-url>
cd DeepScan
```

### 2) Create environment
We recommend **Python 3.10+**.

```bash
conda create -n deepscan python=3.10 -y
conda activate deepscan
```

### 3) Install core dependencies
This codebase is built around a service-oriented pipeline. At minimum, you will need PyTorch, Transformers, OpenCV, FastAPI, and the supporting packages used by the search / visual experts and LVLM runtime.

```bash
pip install torch torchvision torchaudio
pip install transformers accelerate qwen-vl-utils vllm
pip install fastapi uvicorn openai aiohttp pandas scikit-learn shortuuid
pip install pillow numpy matplotlib opencv-python requests
```

Depending on your local setup, you will also need the expert-side dependencies used in this repository:

```bash
# Search expert
pip install salesforce-lavis

# Visual expert
pip install lang-sam

# SAM2 backend
# Install from your local / official SAM2 checkout as needed.
```

### 4) Prepare checkpoints and local paths
The provided code snapshot contains several **environment-specific local paths / placeholders** that should be updated before launch. In particular, check:

- `scripts/blip_server/blip_service.py`
- `scripts/expert_server/model_service.py`
- `scripts/sam2_server/sam2_service.py`
- `scripts/lmm_server/llava_server.sh`
- `scripts/lmm_server/qwen_server.sh`

Before running the pipeline, replace local placeholders with the actual paths for:
- the **BLIP-ITM** tokenizer / checkpoint,
- the **LangSAM / GroundingDINO / SAM2** checkpoints,
- the **SAM2 repository root / config / weights**,
- and the **LVLM checkpoint** you want to serve.

---

## 🧠 Experts and Backbones
DeepScan augments an LVLM with two plug-and-play experts:

### A) Search Expert
The paper uses **BLIP-ITM** as the search expert to produce patch-wise Grad-CAM attention maps for local cue exploration.

### B) Visual Expert
The visual expert exposes two primitives:
- **point-prompt segmentation**, and
- **text-conditioned detection**.

In the paper, DeepScan uses **LangSAM** as the visual expert. In this repository snapshot, the visual grounding pipeline is implemented through the combination of:
- a **LangSAM-based detection service**, and
- a **SAM2 point-prompt segmentation service**.

### C) LVLM Backbones
The paper evaluates DeepScan on five LVLMs:
- **LLaVA-1.5-7B**
- **Qwen2-VL-7B**
- **Qwen2.5-VL-7B**
- **Qwen2.5-VL-32B**
- **Qwen2.5-VL-72B**

This repository also includes example serving scripts for LLaVA / Qwen-style backends under `scripts/lmm_server/`.

---

## 🚀 Usage

### A) Start the services
DeepScan is organized as a multi-service inference pipeline. In a typical setup, you should launch:

1. the **search-expert server**,
2. the **visual-expert server**,
3. the **SAM2 segmentation server**, and
4. the **LVLM server / runtime**.

The corresponding launch scripts are under:

```text
scripts/blip_server/
scripts/expert_server/
scripts/sam2_server/
scripts/lmm_server/
```

Please adapt ports, checkpoint paths, and CUDA device assignment to your environment before starting them.

### B) Run DeepScan on a benchmark / dataset
The main evaluation entry point is `src/run.py`.

```bash
python src/run.py \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --question-file path/to/questions.tsv \
  --answers-file outputs/deepscan_predictions.jsonl \
  --method_name deepscan \
  --temperature 0.0
```

Useful arguments include:

```text
--model-path      LVLM checkpoint / served model name
--question-file   Input question file (TSV)
--answers-file    Output prediction file (JSONL)
--method_name     Method name, e.g. deepscan
--num-chunks      Number of data chunks for parallel evaluation
--chunk-idx       Current chunk index
--temperature     Sampling temperature
--image-size      Image resize limit used by the client runtime
```

### C) Evaluate predictions
After inference, you can evaluate predictions with:

```bash
python src/eval.py --path outputs/deepscan_predictions.jsonl
```

---

## ⚙️ Default Configuration (Paper-Aligned)
The main paper and supplementary material use the following default settings for DeepScan:

- **Search expert**: **BLIP-ITM base**
- **Visual expert**: **LangSAM**
- **Candidate count**: **k = 10**
- **Patch size**:
  - **576 × 576** for **single-object** questions
  - **768 × 768** for **multi-object** questions
- **Noisy-cue area threshold**: **50 pixels**
- **Morphological closing kernel**: **5 × 5** flat structuring element
- **Morphological dilation**: disk with **radius r = 20**
- **IoU threshold** for filtering similar evidence: **0.3**
- **Refocusing zoom-out scale**: **s = 1.5**
- **Refocusing detection padding**: **28 pixels** on each side
- **Max output length**:
  - **50** tokens for:
    - Evidence Decomposition
    - Evidence Judgment
    - View Completeness Justification
  - **1024** tokens for final Evidence-Enhanced Reasoning
- **Inference temperature**: **0**
- **Random seed**: **13**
- **Beam search / top-k sampling**: disabled by default

These settings provide the default performance–latency trade-off reported in the paper.

---

## 🧩 Prompting Roles in DeepScan
DeepScan relies on three lightweight LVLM query templates:

1. **Evidence Decomposition**
   - Extract the objects mentioned in the question.
   - Used to decide whether the question is single-object or multi-object, and thus which patch size to use.

2. **Evidence Judgment**
   - Judge whether a cropped evidence candidate actually contains clues for answering the question.

3. **View Completeness Justification**
   - Judge whether a refocused view fully contains every target object without truncation.

The exact prompt templates are given in the supplementary material and correspond to the logic implemented by the DeepScan pipeline.

---

## 📈 Results at a Glance
DeepScan provides strong gains on fine-grained and visually grounded reasoning benchmarks.

- **V\*** (Qwen2.5-VL-7B backbone): **90.6%** overall
  - **93.0%** Attribute
  - **86.8%** Spatial
- **Improvement over vanilla Qwen2.5-VL-7B**:
  - **+16.3%** on **V\***
  - **+5.5%** on **TreeBench**
- **HR-Bench**:
  - **75.0%** on **HR-4K**
  - **72.4%** on **HR-8K**
- **TreeBench**:
  - **42.5%** overall
  - **37.3 mIoU**
- **Scaling**:
  - **DeepScan-72B** reaches **94.2%** on **V\*** at **k = ∞**

DeepScan is also competitive with strong RL-based visually grounded reasoning methods while remaining fully **training-free**.

---

## ⚡ Efficiency Notes
DeepScan is designed as a **test-time scaling** framework, so it introduces extra inference cost compared with vanilla one-shot inference. At the same time, it admits an explicit performance–efficiency trade-off through:

- the **patch size**,
- the **number of retained evidence candidates (k)**, and
- the batched engineering optimizations described in the supplementary material.

In the optimized implementation discussed in the supplementary material, DeepScan benefits substantially from:

- **batched attention-map computation**,
- **batched top-k evidence judgment**,
- **batched view justification**, and
- **vLLM-based serving**.

These optimizations reduce the sequential overhead of visually grounded search and significantly improve throughput.

---

## 🙏 Acknowledgements
DeepScan builds on several excellent open-source projects and model ecosystems. We would like to give special thanks first to **[DyFo](https://github.com/PKU-ICST-MIPL/DyFo_CVPR2025)** for its inspiring open-source release. We also acknowledge the following projects and model ecosystems:

- **Qwen2-VL / Qwen2.5-VL**
- **LAVIS**
- **LangSAM**
- **SAM2**
- **vLLM**

We thank the authors and maintainers of these projects for making their work available.

---

## 📜 Citation
If you find DeepScan useful, please cite:

```bibtex
@article{li2026deepscan,
  title={DeepScan: A Training-Free Framework for Visually Grounded Reasoning in Large Vision-Language Models},
  author={Li, Yangfu and Zhan, Hongjian and Chen, Jiawei and Gong, Yuning and Liu, Qi and Lu, Yue},
  journal={arXiv preprint arXiv:2603.03857},
  year={2026}
}
```
