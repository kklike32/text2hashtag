# Text2Hashtag

A lightweight demo showing end-to-end LoRA/QLoRA fine-tuning of a local LLM to generate social-media-style hashtags.

## Project Structure

```

text2hashtag/
├── .gitignore
├── environment.yml        # Conda environment spec
├── README.md
├── data/                  # train.jsonl & valid.jsonl
├── models/                # (quantized) model artifacts
├── adapters/              # LoRA adapter weights
├── scripts/               # setup\_env.py, run\_fine\_tune.py
└── src/                   # generate.py, data\_prep.py (to come)

````

## Setup

```bash
# 1. Create Conda env
conda env create --file environment.yml
conda activate text2hashtag

# 2. Verify MPS (Apple GPU support)
python -c "import torch; print(torch.backends.mps.is_available())"
````

## Publishing

Instructions for pushing to GitHub are below.

## Usage

*TODO*

## Data format

Each line in `data/train.jsonl` and `data/valid.jsonl` must be:

```jsonl
{"prompt": "Generate hashtags for the following post:\n\"<POST_TEXT>\"\nHashtags:", "completion": " #tag1 #tag2 #tag3"}
```
