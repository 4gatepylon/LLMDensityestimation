# ============================================================================
# collect_gpt2_act.py  (patched to support multiple datasets)
# ============================================================================
"""Stream a text corpus, record GPT‑2 *layer‑6 post‑ln₁* activations.

Usage examples
--------------
$ python collect_gpt2_act.py --out_dir runs/owt2_l6 \
                             --dataset owt2 --max_tokens 1_000_000
$ python collect_gpt2_act.py --dataset c4 --batch 8 --shard_size 100_000 \
                             --out_dir runs/c4_l6
"""

import argparse, os, math, numpy as np, wandb, torch
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------- dataset helpers ---------------------------

def get_stream(name:str):
    """Return an *iterable* HF split for the requested dataset."""
    if name == "openwebtext":
        return load_dataset("Skylion007/openwebtext", split="train", streaming=True, trust_remote_code=True)
    if name == "owt2":
        return load_dataset("segyges/OpenWebText2", split="train", streaming=True)
    if name == "c4":
        return load_dataset("allenai/c4", "en", split="train", streaming=True)
    if name == "wikitext103":
        return load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    if name == "bookcorpus":
        return load_dataset("lucadiliello/bookcorpusopen", split="train", streaming=True)
    raise ValueError(f"unknown dataset {name}")

# --------------------------- main collector ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./gpt_activations")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--dataset", default="openwebtext",
                    choices=["openwebtext","owt2","c4","wikitext103","bookcorpus"])
    ap.add_argument("--max_tokens", type=int, default=5_000_000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--shard_size", type=int, default=200_000)
    ap.add_argument("--device", default="cuda")
    conf = ap.parse_args()
    os.makedirs(conf.out_dir, exist_ok=True)

    wandb.init(project="additive-vq", name="collect-activations", config=vars(conf))

    tok = AutoTokenizer.from_pretrained(conf.model)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(conf.model, torch_dtype=torch.float16).to(conf.device)
    model.eval()

    feats = []
    def hook(_, __, output):
        feats.append(output.detach().to("cpu"))
    handle = model.transformer.h[5].ln_1.register_forward_hook(hook)

    ds_iter = iter(get_stream(conf.dataset))

    total, shard_idx = 0, 0
    with torch.no_grad():
        while total < conf.max_tokens:
            batch_txt = [next(ds_iter)["text"] for _ in range(conf.batch)]
            enc = tok(batch_txt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(conf.device)
            _ = model(**enc)
            acts = torch.cat(feats, dim=1)  # (batch, seq, dim)
            feats.clear()
            np.save(os.path.join(conf.out_dir, f"act_{shard_idx:05d}.npy"), acts.cpu().float().numpy())
            total += acts.numel() // acts.size(-1)
            shard_idx += 1
            wandb.log({"tokens_collected": total})
    handle.remove()

if __name__ == "__main__":
    main()
