import argparse
import subprocess
import json

def main():
    # Load configuration
    with open('../config.json', 'r') as f:
        config = json.load(f)

    parser = argparse.ArgumentParser(
        description="Generate hashtags via the mlx_lm.generate CLI"
    )
    parser.add_argument(
        "--model", default=f"models/{config['model_name'].split('/')[-1]}-MLX-{config['quantization_bits']}bit",
        help="Hugging Face repo or local path to base quantized model"
    )
    parser.add_argument(
        "--adapter-path", default=config["adapter_path"],
        help="Path to the fine-tuned LoRA adapter weights"
    )
    parser.add_argument(
        "--prompt", "-p", required=True,
        help="Input post text to generate hashtags for"
    )
    parser.add_argument(
        "--max-tokens", "-m", type=int, default=12,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temp", type=float, default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, dest="top_p", default=0.9,
        help="Nucleus sampling (top-p)"
    )
    parser.add_argument(
        "--min-p", type=float, dest="min_p", default=0.0,
        help="Minimum probability threshold (min-p)"
    )
    parser.add_argument(
        "--top-k", type=int, dest="top_k", default=0,
        help="Top-k sampling"
    )
    args = parser.parse_args()

    # Build base prompt
    base_prompt = (
        'Generate hashtags for the following post:\n'
        f'"{args.prompt}"\n'
        'Hashtags:'
    )

    # Construct CLI command
    cmd = [
        'mlx_lm.generate',
        '--model', args.model,
        '--adapter-path', args.adapter_path,
        '--prompt', base_prompt,
        '--max-tokens', str(args.max_tokens),
        '--temp', str(args.temp),
        '--top-p', str(args.top_p),
        '--min-p', str(args.min_p),
        '--top-k', str(args.top_k)
    ]

    # Run and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print('Error during generation:', result.stderr)
        return

    # The CLI prints the generated text, so just print stdout
    print(result.stdout.strip())

if __name__ == '__main__':
    main()
