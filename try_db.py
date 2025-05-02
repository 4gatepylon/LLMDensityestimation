#!/usr/bin/env python3
"""
demo_actdb_usage.py
───────────────────
Minimal end-to-end example for the unified ActivationDB.

Modes
=====
* RUN_DIR = None      → collect + cache a mini DB (~50 k tokens).
* RUN_DIR = "runs/…"  → open an existing DB; all other arguments ignored.
"""

from actdb_gpu import ActivationDB
import numpy as np

# ────── change this to an existing directory to skip collection ──────
RUN_DIR: str | None = None # "/workspace/gpt_act_distribution/runs/gpt2_L6:ln1,L6:ln2_8a491a30"
RUN_DIR = None

def main() -> None:

    # ── 1  Build / Load the database ─────────────────────────────────────
    db = ActivationDB.build(
        run_dir   = RUN_DIR,               # see note above
        model     = "gpt2",
        hooks     = ["L6:ln1", "L7:ln2"],
        dataset   = ("wikitext", "wikitext-103-raw-v1"),
        tokens    = 500_000,                # ↑ for larger corpus
        batch     = 128,  seq = 256,       # shorter seq ⇒ faster demo
        random    = True,
        verbose   = True,
        force     = True                 # force rebuild when RUN_DIR=None
    )

    # ── 2  Define some demo tags ─────────────────────────────────────────
    db.add_tag("SECOND_TOKEN", lambda d: d["id"] == 35677)

    NOUN_IDS = {2335, 1125, 1297}          # toy “noun” subset
    db.add_tag("NOUN",  lambda d: d["pos"] == 1)

    db.add_tag("FIRST_TOKEN_NOUN",
               lambda d: (d["pos"] == 1) & np.isin(d["id"],
                                                    list(NOUN_IDS)))

    db.add_tag("Z_NOT_IN_DATA", lambda d: d["pos"] == 2)
    print(db.count("Z_NOT_IN_DATA"))            # → 0
    print(db.random_windows("Z_NOT_IN_DATA", 3))

    # ── 3  Show counts and example windows ──────────────────────────────
    print("\nTag row counts:")
    for t in ["SECOND_TOKEN", "FIRST_TOKEN", "NOUN", "FIRST_TOKEN_NOUN"]:
        print(f"  {t:<18} {db.count(t):>8,}")

    for tag in ["SECOND_TOKEN", "FIRST_TOKEN", "NOUN", "FIRST_TOKEN_NOUN"]:
        print(f"\n=== Random windows for {tag} ===")
        for s in db.random_windows(tag, k=5, radius=8):
            print("•", s)


if __name__ == "__main__":
    main()
