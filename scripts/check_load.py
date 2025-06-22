from mlx_lm import load, generate

model, tokenizer = load("mlx-community/DeepSeek-R1-0528-Qwen3-8B-6bit")
print("Loaded model & tokenizer OK")

prompt = "Generate hashtags for: \"Just published my first open-source library!\""
if tokenizer.chat_template:
    messages = [{"role":"user","content":prompt}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

resp = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=True)
print(resp)
