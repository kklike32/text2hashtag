#!/usr/bin/env python3
import argparse
import subprocess
import json

def main():
    # Load configuration
    with open('../config.json', 'r') as f:
        config = json.load(f)

    p = argparse.ArgumentParser()
    p.add_argument("--model", default=config["model_name"])
    p.add_argument("--data", default="data")
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-layers", type=int, default=config["num_layers"])
    p.add_argument("--adapter-path", default=config["adapter_path"])
    p.add_argument("--quantize", action="store_true", help="Quantize the model")
    p.add_argument("--bits", type=int, default=config["quantization_bits"], help="Bits for quantization")
    p.add_argument("--test", action="store_true", help="Run evaluation on valid.jsonl instead of training")
    args = p.parse_args()

    if args.quantize:
        # Quantize the model first
        quantize_cmd = [
            "mlx_lm.convert",
            "--hf-path", args.model,
            "--mlx-path", f"models/{args.model.split('/')[-1]}-MLX-{args.bits}bit",
            "--quantize",
            "--q-bits", str(args.bits)
        ]
        subprocess.run(quantize_cmd, check=True)
        args.model = f"models/{args.model.split('/')[-1]}-MLX-{args.bits}bit"

    cmd = ["mlx_lm.lora", "--model", args.model, "--data", args.data, "--adapter-path", args.adapter_path]

    if args.test:
        cmd += ["--test"]
    else:
        cmd += ["--train", "--iters", str(args.iters), "--batch-size", str(args.batch_size), "--num-layers", str(args.num_layers), "--mask-prompt"]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
