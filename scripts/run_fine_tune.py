#!/usr/bin/env python3
import argparse
import subprocess

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        default="mlx-community/DeepSeek-R1-0528-Qwen3-8B-6bit")
    p.add_argument("--data",         default="data")
    p.add_argument("--iters",        type=int, default=600)
    p.add_argument("--batch-size",   type=int, default=2)
    p.add_argument("--num-layers",   type=int, default=8)
    p.add_argument("--adapter-path", default="adapters/text2hashtag_adapter.safetensors")
    p.add_argument("--test",         action="store_true",
                     help="Run evaluation on valid.jsonl instead of training")
    args = p.parse_args()

    cmd = ["mlx_lm.lora", "--model", args.model, "--data", args.data,
           "--adapter-path", args.adapter_path]

    if args.test:
        cmd += ["--test"]
    else:
        cmd += ["--train",
                "--iters", str(args.iters),
                "--batch-size", str(args.batch_size),
                "--num-layers", str(args.num_layers),
                "--mask-prompt"]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
