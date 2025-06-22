#!/usr/bin/env python3
"""
Skeleton for invoking MLX-LM fine-tuning.
Fill in argparse and call mlx_lm.lora as needed.
"""
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # e.g. parser.add_argument("--iters", type=int, default=400)
    return parser.parse_args()

def main():
    args = parse_args()
    # TODO: import mlx_lm.lora and pass args through to it
    pass

if __name__ == "__main__":
    main()

