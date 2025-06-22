# Text2Hashtag

Keenan Kalra

**Lightweight end-to-end QLoRA demo** for generating social-media-style hashtags from arbitrary text posts using a fine-tuned LLM on Apple Silicon.

---

## 🚀 Project Overview

This repository showcases a minimal pipeline that:

1. **Prepares a small dataset** of prompt/completion pairs in JSONL format.
2. **Quantizes** a base model (`mlx-community/DeepSeek-R1-0528-Qwen3-8B-6bit`) to 6-bit with MLX-LM.
3. **Fine-tunes** only LoRA adapter weights (few million params) locally on an M4 Mac (48 GB) in minutes.
4. **Evaluates** validation/test perplexity to choose the best checkpoint.
5. **Generates** hashtags via a simple CLI wrapper around `mlx_lm.generate`.

**Key features:** low memory footprint, fast training (<10 min), easily auditable dataset, and clear modular scripts.

---

## 📁 Repository Structure

```
text2hashtag/
├── .gitignore
├── environment.yml      # Conda environment spec
├── README.md            # This file
├── data/                # train.jsonl, valid.jsonl, test.jsonl
├── adapters/            # Fine-tuned LoRA adapter checkpoints
├── scripts/
│   ├── run_fine_tune.py # QLoRA training launcher
│   └── setup_env.py     # Creates Conda env from environment.yml
└── src/
    └── generate.py      # CLI wrapper for mlx_lm.generate
```

---

## ⚙️ Setup

```bash
# 1. Clone
git clone https://github.com/<USERNAME>/text2hashtag.git
cd text2hashtag

# 2. Create Conda env
conda env create --file environment.yml
conda activate text2hashtag

# 3. Verify MPS backend
python - <<EOF
import torch
print('MPS available:', torch.backends.mps.is_available())
EOF
```

---

## 🛠️ Usage

### Data

Populate `data/train.jsonl`, `data/valid.jsonl`, and `data/test.jsonl` with lines like:

```jsonl
{"prompt":"Generate hashtags for the following post:\n\"Your post here\"\nHashtags:","completion":" #tag1 #tag2 #tag3"}
```

### Fine-tuning

```bash
python scripts/run_fine_tune.py \
  --model mlx-community/DeepSeek-R1-0528-Qwen3-8B-6bit \
  --iters 200 \
  --batch-size 1 \
  --num-layers 4
```

### Evaluation

```bash
python scripts/run_fine_tune.py --test
```

### Generation

```bash
python src/generate.py \
  --prompt "I just graduated from MIT with a degree in Data Science" \
  --max-tokens 12 \
  --temp 0.7 \
  --top-p 0.9 \
  --min-p 0.1 \
  --top-k 50
```

---

## 📝 Customization

* **Adjust hyperparameters** in `run_fine_tune.py`: iterations, layers, batch size.
* **Modify sampling** in `generate.py`: temperature, top-p, top-k, min-p for diversity vs. precision.
* **Fuse model** via MLX-LM fuse command for a deployable `.gguf` or `.safetensors` file.

---
