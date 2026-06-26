"""Colab/Jupyter helpers: idempotent dataset.map patch + Turing GPU SDPA fallback."""

from __future__ import annotations

import subprocess
import sys


def patch_dataset_map_no_mp() -> bool:
    """Force datasets.Dataset.map to num_proc=None (avoids Colab PicklingError).

    Safe to call from every notebook cell — uses a closure + guard, never stacks patches.
    """
    import datasets.arrow_dataset as _ds_arrow

    current = _ds_arrow.Dataset.map
    if getattr(current, "_lab22_no_mp", False):
        return False

    original_map = current

    def _dataset_map_single_process(self, *args, **kwargs):
        kwargs["num_proc"] = None
        return original_map(self, *args, **kwargs)

    _dataset_map_single_process._lab22_no_mp = True
    _ds_arrow.Dataset.map = _dataset_map_single_process
    return True


def force_sdpa_if_turing() -> bool:
    """T4/V100 (sm < 80): xformers FA backward needs Ampere+ — use PyTorch SDPA."""
    import torch

    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    if major >= 8:
        return False

    sm = f"sm_{major}{minor}"
    print(f"[lab22] {sm}: forcing PyTorch SDPA (xformers FA backward needs sm_80+)")

    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "xformers"],
        capture_output=True,
    )

    try:
        import unsloth.models._utils as uu

        uu.xformers = None
    except Exception:
        pass

    try:
        import unsloth.utils.attention_dispatch as ad

        ad.HAS_XFORMERS = False

        def _sdpa_only(use_varlen: bool = False) -> str:
            return ad.SDPA

        ad.select_attention_backend = _sdpa_only
    except Exception as exc:
        print(f"[lab22] WARN: could not patch attention_dispatch: {exc}")
        return False

    return True


def export_gguf_from_adapters(model, tokenizer, gguf_dir, *, quantization: str = "q4_k_m") -> None:
    """Export GGUF from SFT+DPO adapter stack — Colab-safe (no HF save_pretrained).

    Do **not** use ``merge_and_unload()`` + ``save_pretrained()`` on 4-bit Colab:
    newer ``transformers`` raises ``NotImplementedError`` in ``reverse_op``.
    Unsloth merges internally during ``save_pretrained_gguf``.
    """
    from pathlib import Path
    from unsloth import FastLanguageModel

    force_sdpa_if_turing()
    FastLanguageModel.for_inference(model)
    out = Path(gguf_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[lab22] Exporting GGUF ({quantization}) → {out}")
    model.save_pretrained_gguf(
        str(out),
        tokenizer,
        quantization_method=quantization,
    )
    print("[lab22] GGUF export done")
