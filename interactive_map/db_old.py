#!/usr/bin/env python3
"""
actdb_gpu.py  –  GPU Activation-DB + tag toolkit  (v0.8.1, 2025-05-01)
──────────────────────────────────────────────────────────────────────
* Tags are stored **once per token** (independent of hook rows).
* Token-keys are now signed int64 to satisfy `torch.from_numpy`.
"""

from __future__ import annotations
import argparse, json, pathlib, re, gc, collections, hashlib
from typing import Dict, Sequence, Callable, List, Any

import numpy as np, joblib, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
from tqdm.auto import tqdm

__all__ = ["ActivationDB"]

PURPLE_BOLD, RESET = "\033[1;95m", "\033[0m"
_RE_SP = re.compile(r"\s+")


# ═══════════════ tiny LRU for decoded slices ══════════════════════════════
class _LRU(collections.OrderedDict):
    def __init__(self, cap=4096): super().__init__(); self._cap = cap
    def __getitem__(self, k): v = super().__getitem__(k); self.move_to_end(k); return v
    def __setitem__(self, k, v):
        if k in self: super().__delitem__(k)
        elif len(self) >= self._cap: self.popitem(last=False)
        super().__setitem__(k, v)


# ═════════════════ helper utils ═══════════════════════════════════════════
def _cfg_hash(d: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]


def _make_run_dir(cfg: dict[str, Any]) -> pathlib.Path:
    p = pathlib.Path("runs") / f"{cfg['model']}_{cfg['hooks']}_{_cfg_hash(cfg)}"
    p.mkdir(parents=True, exist_ok=True)
    (p / "meta.json").write_text(json.dumps(cfg, indent=2))
    return p


def _resolve_module(mdl: torch.nn.Module, spec: str) -> torch.nn.Module:
    if ":" in spec:                                    #  e.g.  L6:ln1
        layer, sub = spec.split(":")
        idx = int(layer[1:])
        blk = mdl.transformer.h[idx]
        mapping = {"ln1": "ln_1", "ln2": "ln_2",
                   "attn": "attn", "mlp": "mlp", "resid": None}
        return blk if mapping[sub] is None else getattr(blk, mapping[sub])
    mod = mdl
    for tok in spec.split("."):
        mod = mod[int(tok)] if tok.isdigit() else getattr(mod, tok)
    return mod


# ═════════════════════════ ActivationDB ═══════════════════════════════════
class ActivationDB:
    # ───────────────────── factory / loader ──────────────────────────────
    @classmethod
    def build(cls, *, run_dir: str | None,
              model: str | None,
              hooks: Sequence[str] | None,
              dataset: tuple[str, str] | None,
              tokens: int = 200_000,
              batch: int = 256,
              seq: int = 1024,
              random: bool = False,
              cpu: bool = False,
              verbose: bool = True,
              force: bool = False):
        if run_dir is not None:
            rd = pathlib.Path(run_dir).expanduser()
            if verbose:
                print(f"[ActivationDB.build] using existing DB → {rd}")
            if not (rd / "meta.json").is_file() or not any(rd.glob("act_*.npz")):
                raise FileNotFoundError(f"run_dir '{rd}' is not a valid ActivationDB")
            return cls(str(rd), verbose=verbose)

        assert model and hooks and dataset, "model, hooks, dataset are required"

        cfg = dict(model=model, hooks=",".join(hooks),
                   dataset=dataset[0], config=dataset[1],
                   tokens=tokens, batch=batch, seq=seq,
                   random=random, cpu=cpu, verbose=verbose, version="0.8.1")
        rd = _make_run_dir(cfg)

        need_collect = force or not any(rd.glob("act_*.npz"))
        if need_collect:
            if verbose: print(f"[ActivationDB.build] collecting → {rd}")
            dev = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
            mdl = (AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model))
                   if random else AutoModelForCausalLM.from_pretrained(model)).eval().to(dev)
            tok = AutoTokenizer.from_pretrained(model); tok.pad_token = tok.eos_token
            ds  = load_dataset(*dataset, split="train", streaming=True)\
                    .shuffle(seed=42, buffer_size=100_000)
            _collect(cfg, mdl, tok, ds, dev, rd, list(hooks))
        elif verbose:
            print(f"[ActivationDB.build] re-using cached run → {rd}")

        return cls(str(rd), verbose=verbose)

    # ───────────────────── loader internals ──────────────────────────────
    def __init__(self, run_dir: str, hooks: Sequence[str] | None = None,
                 verbose: bool = True):
        self.verbose = verbose
        self.run_dir = pathlib.Path(run_dir).expanduser()
        meta         = json.load(open(self.run_dir / "meta.json"))
        self.hooks   = list(hooks) if hooks else meta["hooks"].split(",")
        self._hook2i = {h: i for i, h in enumerate(self.hooks)}

        self.tok = AutoTokenizer.from_pretrained(meta["model"])
        self.tok.pad_token = self.tok.eos_token

        data_dir = self.run_dir / "data"
        self._chunks  = sorted(data_dir.glob("act_*.npz"))
        if not self._chunks:
             raise FileNotFoundError(f"No activation files (act_*.npz) found in {data_dir}")
        self._mem     = [np.load(f, mmap_mode="r") for f in self._chunks]
        self.seqs: list[list[int]] = joblib.load(data_dir / "seqs.pkl")

        self._rows    = sum(len(d["h"]) for d in self._mem)
        self._cumrows = np.cumsum([len(d["h"]) for d in self._mem])
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # master token-key vector (int64 ⇐ avoids torch uint64 issue)
        seq_cat = np.concatenate([d["seq"]     for d in self._mem]).astype(np.int64)
        ofs_cat = np.concatenate([d["seq_ofs"] for d in self._mem]).astype(np.int64)
        tok_np  = (seq_cat << 32) | ofs_cat               # int64
        self._tok_keys = torch.from_numpy(tok_np).to(self.device)

        src_cat = torch.cat([torch.from_numpy(d["src"]).to(self.device)
                             for d in self._mem])
        self._hook_masks = {i: (src_cat == i).nonzero(as_tuple=False).flatten()
                            for i in range(len(self.hooks))}

        # load tag token-keys
        self._tag_tokens: Dict[str, torch.Tensor] = {}
        tags_dir = self.run_dir / "tags"
        if tags_dir.is_dir(): # Check if the tags directory exists
            for f in tags_dir.glob("tag_*.npy"):
                name = f.stem[4:]
                arr  = np.load(f).astype(np.int64)
                self._tag_tokens[name] = torch.from_numpy(arr).to(self.device)

        self._slice_cache = _LRU()

    # ───────────────────── public helpers ────────────────────────────────
    def __len__(self): return self._rows
    @property
    def d(self): return self._mem[0]["h"].shape[1]

    # tag creation ----------------------------------------------------------
    def add_tag(self, name: str,
                predicate: Callable[[Dict[str, np.ndarray]], np.ndarray],
                hook: str | None = None):
        if self.verbose:
            pbar = tqdm(self._mem, desc=f"[tag {name}] chunks")
        else:
            pbar = self._mem

        hi_filter = self._hook2i.get(hook) if hook else None
        seen: set[int] = set()
        tok_keys_out: List[int] = []

        for d in pbar:
            mask = predicate(d).astype(bool).ravel()
            if mask.size != len(d["h"]):
                raise ValueError("Predicate length mismatch")
            if hi_filter is not None:
                mask &= (d["src"] == hi_filter)
            if not mask.any(): continue

            keys = ((d["seq"].astype(np.int64) << 32) |
                    d["seq_ofs"].astype(np.int64))[mask]

            for k in keys:
                if k not in seen:
                    seen.add(int(k))
                    tok_keys_out.append(int(k))

        tags_dir = self.run_dir / "tags"
        tags_dir.mkdir(exist_ok=True) # Ensure the tags directory exists
        tok_keys_np = np.asarray(tok_keys_out, dtype=np.int64)
        np.save(tags_dir / f"tag_{name}.npy", tok_keys_np)
        self._tag_tokens[name] = torch.from_numpy(tok_keys_np).to(self.device)

        if self.verbose:
            print(f"[ActivationDB] tag '{name}' added – {len(tok_keys_np):,} tokens")

    # fast counts -----------------------------------------------------------
    def count(self, tag: str, *, hook: str | None = None) -> int:
        if tag not in self._tag_tokens:
            return 0
        tok_set = self._tag_tokens[tag]
        if hook is None:
            return int(tok_set.numel())

        hi = self._hook2i.get(hook)
        rows = self._hook_masks[hi]
        keys_in_hook = self._tok_keys[rows]
        return int(torch.isin(keys_in_hook, tok_set).sum().item())

    # random windows --------------------------------------------------------
    def _pretty(self, ids, trig_off: int = -1) -> str:
        toks = self.tok.convert_ids_to_tokens(ids, skip_special_tokens=False)
        out  = []
        TRAIL_PUNCT = ",.;:!?%)'"

        for i, tok in enumerate(toks):
            space = tok.startswith("Ġ")
            core  = tok.lstrip("Ġ")
            if not core:                 # sentinel-only → skip
                continue
            if i == trig_off:
                core = f"{PURPLE_BOLD}{core}{RESET}"

            # ── spacing heuristic ─────────────────────────────────────────
            if space:
                if core[0] in TRAIL_PUNCT:     # punctuation → no space
                    out.append(core)
                else:                          # normal word → keep space
                    out.append(" " + core)
            else:
                out.append(core)

        return re.sub(r"\s{2,}", " ", "".join(out)).strip()

    def _row_from_key(self, key: int) -> int:
        return int(torch.nonzero(self._tok_keys == key, as_tuple=False)[0])

    def random_windows(self, tag: str, k: int, radius: int = 8,
                       colour: bool = True) -> list[tuple[str, int]]:
        if tag not in self._tag_tokens or self._tag_tokens[tag].numel() == 0:
            return []

        tok_keys = self._tag_tokens[tag]
        pick     = tok_keys[torch.randperm(tok_keys.numel(), device=self.device)[:k]]
        outs: list[tuple[str, int]] = []

        iterator = pick.cpu().numpy()
        if self.verbose:
            iterator = tqdm(iterator, desc="[windows]", unit="win")

        for key in iterator:
            row = self._row_from_key(int(key))
            ch, inner = self._locate_rows_cpu(np.array([row]))
            d        = self._mem[int(ch)]
            seq_i    = int(d["seq"][int(inner)])
            ofs      = int(d["seq_ofs"][int(inner)])
            ids      = self.seqs[seq_i][max(ofs - radius, 0): ofs + radius + 1]
            trig_off = ofs - max(ofs - radius, 0)
            trigger_id = ids[trig_off]
            pretty_str = self._pretty(ids, trig_off if colour else -1)
            outs.append((pretty_str, trigger_id))
        return outs

    # ───────────────────── internal helpers ───────────────────────────────
    def _locate_rows_cpu(self, rows_cpu: np.ndarray):
        ch = np.searchsorted(self._cumrows, rows_cpu, side="right")
        inner = rows_cpu - np.concatenate(([0], self._cumrows[:-1]))[ch]
        return ch, inner

    def _decode_slice(self, ids):
        key = tuple(ids)
        if key in self._slice_cache: return self._slice_cache[key]
        txt = self.tok.decode(ids, clean_up_tokenization_spaces=False)
        self._slice_cache[key] = txt
        return txt


# ═════════════ internal collector (unchanged) ═════════════════════════════
def _collect(cfg, mdl, tok, ds, dev, rd: pathlib.Path, hooks: list[str]):
    target_tok, seq_len, batch = cfg["tokens"], cfg["seq"], cfg["batch"]
    d_model = mdl.config.hidden_size

    buf: Dict[str, torch.Tensor] = {}
    acts_all = {h: [] for h in hooks}
    ids_all, pos_all, seq_all, ofs_all = [], [], [], []
    seqs: list[list[int]] = []

    def make_hook(hname):
        def _hook(_, __, out): buf[hname] = out        # (B,S,d)
        return _hook
    handles = [_resolve_module(mdl, h).register_forward_hook(make_hook(h))
               for h in hooks]

    it = iter(ds)
    tok_seen = 0
    seq_idx  = 0
    pbar     = tqdm(total=target_tok, desc="[collect]", unit="tok") if cfg["verbose"] else None

    while tok_seen < target_tok:
        txt = [next(it)["text"] for _ in range(batch)]
        enc = tok(txt, return_tensors="pt",
                  truncation=True, max_length=seq_len,
                  padding="max_length")
        keep = enc["attention_mask"].flatten().bool()          # drop PAD
        B, S  = enc["input_ids"].shape

        with torch.inference_mode():
            mdl(**{k: v.to(dev) for k, v in enc.items()})

        ids_flat = enc["input_ids"].flatten()[keep]
        pos_flat = torch.arange(S).repeat(B)[keep]
        seq_ids  = torch.arange(seq_idx, seq_idx+B).unsqueeze(1)\
                     .expand(B, S).flatten()[keep]
        ofs_flat = pos_flat.clone()

        ids_all.append(ids_flat.cpu())
        pos_all.append(pos_flat.cpu())
        seq_all.append(seq_ids.cpu())
        ofs_all.append(ofs_flat.cpu())

        for h in hooks:
            acts = buf[h].view(-1, d_model)[keep].to("cpu").half()
            acts_all[h].append(acts)
        buf.clear()

        seqs.extend([row.tolist() for row in enc["input_ids"]])
        seq_idx  += B
        tok_seen += int(keep.sum())
        if pbar: pbar.update(int(keep.sum()))

    if pbar: pbar.close()

    ids_all = np.concatenate(ids_all).astype(np.uint16)
    pos_all = np.concatenate(pos_all).astype(np.uint16)
    seq_all = np.concatenate(seq_all).astype(np.int64)
    ofs_all = np.concatenate(ofs_all).astype(np.int32)

    h_concat, src_concat = [], []
    for hi, h in enumerate(hooks):
        acts = np.concatenate([a.numpy() for a in acts_all[h]])
        h_concat.append(acts)
        src_concat.append(np.full(len(acts), hi, np.uint8))

    h_concat   = np.concatenate(h_concat)
    src_concat = np.concatenate(src_concat)

    data_dir = rd / "data"
    data_dir.mkdir(exist_ok=True)

    np.savez_compressed(
        data_dir / "act_000.npz",
        h=h_concat,
        src=src_concat,
        id=np.tile(ids_all, len(hooks)),
        pos=np.tile(pos_all, len(hooks)),
        seq=np.tile(seq_all, len(hooks)),
        seq_ofs=np.tile(ofs_all, len(hooks)),
    )
    joblib.dump(seqs, data_dir / "seqs.pkl")

    for h in handles: h.remove()
    torch.cuda.empty_cache(); gc.collect()
