import torch
import argparse
import os, json
from nn_ops import softmax
from nn_serialization import load_checkpoint
from train_bpe import load_tokenizer


def generate(model, tokenizer, input_text: str,
                    eos_str: str, temperature: float,
                    top_p: float, max_tokens: int):
    generated_tokens = []
    model.to("cuda:0")
    model.eval()
    
    token_lst = tokenizer.encode(input_text)
    eos_token = tokenizer.encode(eos_str)[0]
    with torch.no_grad():
        for _ in range(max_tokens):
            tokens = torch.tensor(token_lst).to("cuda:0")
            logits_s_v = model(tokens)
            probs_s = softmax(logits_s_v[-1, :] / (temperature + 1e-6), dim=-1)
            probs, indices = torch.sort(probs_s, dim=0, descending=True)
            cdf = torch.cumsum(probs, dim=0)
            cutoff = torch.searchsorted(cdf, torch.tensor(top_p, device=probs.device), right=True).item()
            cutoff = max(1, cutoff)  # keep at least 1 token
            probs = probs[:cutoff]
            probs = probs/probs.sum()
            prob_idx = torch.multinomial(probs, num_samples=1)
            new_token_idx = indices[prob_idx]
            generated_tokens.append(new_token_idx)
            print(tokenizer.decode([new_token_idx.item()]), end="")
            if new_token_idx.item() == eos_token:
                break
            token_lst.append(new_token_idx)


def load_model_and_tokenizer(model_dir, tokenizer_path):
    from nn_transformer import Transformer
    from nn_optim import AdamW
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
        model = Transformer(**config)
    with open(os.path.join(model_dir, "optim_config.json"), "r") as f:
        config = json.load(f)
        optimizer = AdamW(model.parameters(), config["lr"])
    ckpt = os.path.join(model_dir, "model.bin")
    _ = load_checkpoint(ckpt, model, optimizer)
    tokenizer = load_tokenizer(tokenizer_path)
    return model, tokenizer

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--input_text", required=True)
    parser.add_argument("--eos_str", required=True)
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--max_tokens", required=True, type=int)
    args = parser.parse_args()
    
    model, tokenizer = load_model_and_tokenizer(args.model_dir,
                                                args.tokenizer_path)
    generate(model, tokenizer, args.input_text,
                args.eos_str, args.temperature,
                args.top_p, args.max_tokens)

 # uv run generate.py --model_dir ../checkpoints/transformer_hyperparam/lr_0.001/b_32/ --tokenizer_path ../tests/fixtures/tokenizers/tinystories --input_text "Tom and Lily were playing with their toys in" --eos_str "<|endoftext|>" --temperature 0.1 --top_p 0.95 --max_tokens 200