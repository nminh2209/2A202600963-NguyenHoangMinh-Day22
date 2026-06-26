#!/usr/bin/env python3
"""CLI wrapper for NB5 logic — merge adapter + quantize to GGUF.

Usage:
    python scripts/merge_and_gguf.py
    python scripts/merge_and_gguf.py --quant q5_k_m
    python scripts/merge_and_gguf.py --quant q4_k_m --quant q5_k_m --quant q8_0

Mirrors `notebooks/05_merge_deploy_gguf.py` cells 1-2. Used if you want to add
extra GGUF tiers (the +3 'GGUF release published' rigor add-on).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "notebooks"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-path", default=str(REPO / "adapters" / "sft-mini"))
    parser.add_argument("--dpo-path", default=str(REPO / "adapters" / "dpo"))
    parser.add_argument("--gguf-output", default=str(REPO / "gguf"))
    parser.add_argument("--quant", action="append", default=None,
                        help="Quantization tier(s). Repeat for multiple. Default: q4_k_m")
    args = parser.parse_args()

    quants = args.quant or ["q4_k_m"]

    tier = os.environ.get("COMPUTE_TIER", "T4").upper()
    base = (
        "unsloth/Qwen2.5-3B-bnb-4bit" if tier == "T4"
        else "unsloth/Qwen2.5-7B-bnb-4bit"
    )
    max_len = 512 if tier == "T4" else 1024

    Path(args.gguf_output).mkdir(parents=True, exist_ok=True)

    print(f"Tier: {tier}  base: {base}  quants: {quants}")

    from peft import PeftModel
    from unsloth import FastLanguageModel

    from _lab22_colab_utils import export_gguf_from_adapters, force_sdpa_if_turing

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base, max_seq_length=max_len, dtype=None, load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(model, args.sft_path)
    print("Loaded SFT-mini adapter")
    model = PeftModel.from_pretrained(model, args.dpo_path)
    print("Loaded DPO adapter")
    force_sdpa_if_turing()

    for q in quants:
        print(f"Quantizing to GGUF {q}...")
        export_gguf_from_adapters(model, tokenizer, args.gguf_output, quantization=q)

    print(f"\nGGUF files in {args.gguf_output}:")
    for p in sorted(Path(args.gguf_output).iterdir()):
        if p.suffix == ".gguf":
            print(f"  {p.name:50s}  {p.stat().st_size / 1e6:>8.1f} MB")


if __name__ == "__main__":
    main()
