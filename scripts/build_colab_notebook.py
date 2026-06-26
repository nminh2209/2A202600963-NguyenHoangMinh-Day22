#!/usr/bin/env python3
"""Rebuild Colab single-file notebooks from Jupytext notebook sources.

Adds Colab-specific preamble (clone, API keys, artifact download) and stitches
all 6 notebook stages into one .ipynb per tier.

Run from repo root (stdlib only — no venv required):
    python scripts/build_colab_notebook.py
    python scripts/build_colab_notebook.py --tier T4
    python scripts/build_colab_notebook.py --tier BIGGPU
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = REPO / "notebooks"
COLAB = REPO / "colab"
DEFAULT_REPO = "https://github.com/VinUni-AI20k/Day22-Track3-DPO-Alignment-Lab.git"

NOTEBOOK_ORDER = [
    "01_sft_mini.py",
    "02_preference_data.py",
    "03_dpo_train.py",
    "04_compare_and_eval.py",
    "05_merge_deploy_gguf.py",
    "06_benchmark.py",
]

TIER_CONFIG = {
    "T4": {
        "title": "Lab 22 — DPO/ORPO Alignment (T4 tier)",
        "tier": "T4",
        "model_note": "Qwen2.5-3B + 2k UltraFeedback",
        "output": "Lab22_DPO_T4.ipynb",
    },
    "BIGGPU": {
        "title": "Lab 22 — DPO/ORPO Alignment (BigGPU tier)",
        "tier": "BIGGPU",
        "model_note": "Qwen2.5-7B + 5k UltraFeedback",
        "output": "Lab22_DPO_BigGPU.ipynb",
    },
}


def md_cell(text: str, cell_id: str | None = None) -> dict:
    cell = {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}
    if cell_id:
        cell["id"] = cell_id
    return cell


def code_cell(text: str, cell_id: str | None = None) -> dict:
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }
    if cell_id:
        cell["id"] = cell_id
    return cell


def _lines(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    return lines if lines else [text + "\n"]


def preamble_cells(cfg: dict, repo_url: str) -> list[dict]:
    title = cfg["title"]
    tier = cfg["tier"]
    model_note = cfg["model_note"]

    intro = f"""# {title}

**Track 3 · Day 22 · VinUni AICB program**

Single-file Colab notebook — all 6 lab stages in order:
1. SFT-mini · 2. Preference data · 3. DPO training · 4. Compare & eval
5. Merge → GGUF · 6. Benchmark (optional bonus)

**Tier:** `{tier}` — {model_note}

### Before you run
1. **Runtime → Change runtime type → T4 GPU** (free Colab) or A100/L4 for BigGPU
2. Run cells **top to bottom** — total ~45–90 min depending on tier
3. Paste your **OpenAI API key** in section B (optional but recommended for NB4 judge)
4. At the end, run **section Z** to download artifacts back to your laptop

> Laptop with < 12 GB VRAM? Use this notebook — local training will OOM.
"""

    clone = f"""## A. Clone lab repo + set working directory

Edit `REPO_URL` if you forked the repo to your own GitHub account.
"""

    clone_code = f'''import os
from pathlib import Path

# ── Edit this if you use your own fork ──────────────────────────────────
REPO_URL = os.environ.get("LAB22_REPO_URL", "{repo_url}")
BRANCH = os.environ.get("LAB22_BRANCH", "main")

WORK = Path("/content/Day22-Track3-DPO-Alignment-Lab")

if WORK.exists() and (WORK / "notebooks" / "01_sft_mini.py").exists():
    print(f"Repo already at {{WORK}} — skipping clone")
else:
    !rm -rf {{WORK}}
    !git clone --depth 1 -b {{BRANCH}} {{REPO_URL}} {{WORK}}

%cd {{WORK}}
print(f"Working directory: {{Path.cwd()}}")
'''

    api_keys = """## B. API keys (optional — enables gpt-4o-mini judge in NB4 + NB6)

Without a key, NB4 falls back to manual rubric (no points lost).

**Recommended:** Colab → 🔑 Secrets → add `OPENAI_API_KEY` (loaded automatically in the cell below).
Or paste directly into Option 2 (don't commit keys to GitHub).
"""

    api_code = '''import os

# Option 1 — Colab Secrets (recommended): Secrets panel → OPENAI_API_KEY
try:
    from google.colab import userdata
    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
    print("Loaded OPENAI_API_KEY from Colab Secrets")
except Exception:
    pass

# Option 2 — paste key here (delete before pushing to public GitHub!)
# os.environ["OPENAI_API_KEY"] = "sk-..."

# Optional: Anthropic judge for cross-judge bonus (+4 pts)
# os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")

os.environ.setdefault("JUDGE_MODEL", "gpt-4o-mini")
os.environ["COMPUTE_TIER"] = "''' + tier + '''"

if os.environ.get("OPENAI_API_KEY"):
    print("OpenAI judge: ready")
else:
    print("No OPENAI_API_KEY — NB4 will use manual rubric mode")
'''

    install = """## C. Install dependencies (~2–4 min)

**Core only** for NB1–NB4. Skips `llama-cpp-python` (slow compile) and `lm-eval` until bonus stages.
Colab already ships PyTorch+CUDA — we do **not** reinstall torch here.
"""

    install_code = '''import subprocess, sys

def pip_install(*packages):
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--progress-bar", "on", *packages]
    print(">>", " ".join(packages))
    subprocess.check_call(cmd)

def pip_uninstall(pkg):
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", pkg], check=False)

# datasets<4.0 — v4 pulls torchcodec (audio); this lab is text-only
pip_install(
    "bitsandbytes>=0.44,<1.0",
    "accelerate>=1.10,<2.0",
    "datasets>=3.1,<4.0",
    "peft>=0.13,<1.0",
    "trl>=0.12,<0.20",
    "matplotlib>=3.9,<4.0",
    "pandas>=2.2,<3.0",
    "pyarrow>=17,<22",
    "openai>=1.55,<2.0",
    "anthropic>=0.40,<1.0",
)

# Unsloth on Colab's existing torch — upgrade for torchcodec workaround patches
pip_install("--upgrade", "unsloth>=2025.10,<2026.5", "unsloth_zoo")

# Ensure accelerate is new enough AFTER unsloth (unsloth can pull older accelerate)
pip_install("--upgrade", "accelerate>=1.10,<2.0")

# Colab torch 2.10 + torchcodec mismatch breaks `from unsloth import ...`
# Text-only DPO lab does not need torchcodec — remove it (unsloth issue #5446)
pip_uninstall("torchcodec")

import torch
import accelerate
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"accelerate {accelerate.__version__}  (need >= 1.10 for DPO train)")

# T4/V100 (sm < 80): xformers 0.0.33+ FA backward needs Ampere+ — drop before unsloth import
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
    _cc = torch.cuda.get_device_capability()
    print(f"Turing GPU sm_{_cc[0]}{_cc[1]}: uninstalling xformers (SDPA fallback)")
    pip_uninstall("xformers")

# Smoke-test import before NB1 (fail fast with clear message)
from unsloth import FastLanguageModel
print("unsloth import OK")
print("Core install done. NB5/NB6 install their own optional deps.")
'''

    gpu = """## D. GPU check + create output folders
"""

    colab_utils_src = (NOTEBOOKS / "_lab22_colab_utils.py").read_text(encoding="utf-8")
    gpu_code = f'''import torch
from pathlib import Path

assert torch.cuda.is_available(), (
    "No GPU! Runtime → Change runtime type → T4 GPU, then re-run from section A."
)
gpu = torch.cuda.get_device_properties(0)
print(f"GPU: {{gpu.name}}  ({{gpu.total_memory / 1e9:.1f}} GB)")

REPO_ROOT = Path("/content/Day22-Track3-DPO-Alignment-Lab")
for sub in [
    "data/pref", "data/eval",
    "adapters/sft-mini", "adapters/dpo", "adapters/merged-fp16",
    "gguf", "submission/screenshots",
]:
    (REPO_ROOT / sub).mkdir(parents=True, exist_ok=True)

import os
os.chdir(REPO_ROOT / "notebooks")
print(f"Notebook cwd: {{Path.cwd()}}")
print(f"REPO_ROOT: {{REPO_ROOT}}")

# Lab22 Colab helpers (idempotent — safe to re-run section D)
_lab22_utils_src = {colab_utils_src!r}
_utils_py = REPO_ROOT / "notebooks" / "_lab22_colab_utils.py"
if not _utils_py.exists():
    _utils_py.write_text(_lab22_utils_src)
    print(f"Wrote {{_utils_py.name}} (missing from clone)")

import sys
sys.path.insert(0, str(REPO_ROOT / "notebooks"))
from _lab22_colab_utils import patch_dataset_map_no_mp, force_sdpa_if_turing
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if patch_dataset_map_no_mp():
    print("Patched datasets.Dataset.map -> num_proc=None (Colab-safe)")
else:
    print("datasets.Dataset.map already patched (Colab-safe)")
force_sdpa_if_turing()
'''

    stages_hdr = """---
## Stages 1–6 (NB1 → NB6)

Run cells in order. If you OOM: **Runtime → Restart session**, re-run A–D, then continue.
Core grading only needs stages 1–4 (~30 min on T4).
---
"""

    return [
        md_cell(intro, "intro"),
        md_cell(clone, "clone-md"),
        code_cell(clone_code, "clone-code"),
        md_cell(api_keys, "api-md"),
        code_cell(api_code, "api-code"),
        md_cell(install, "install-md"),
        code_cell(install_code, "install-code"),
        md_cell(gpu, "gpu-md"),
        code_cell(gpu_code, "gpu-code"),
        md_cell(stages_hdr, "stages-hdr"),
    ]


def epilogue_cells() -> list[dict]:
    text = """---
## Z. Download artifacts to your laptop

Run this after NB1–NB4 (or full pipeline). Zip includes adapters, eval JSON,
screenshots, and metrics — copy into your local repo for `submission/` + `make verify`.

Then fill `submission/REFLECTION.md` locally and push to GitHub for LMS submission.
"""
    code = '''import shutil
from pathlib import Path
from google.colab import files

REPO_ROOT = Path("/content/Day22-Track3-DPO-Alignment-Lab")
zip_path = Path("/content/lab22-artifacts.zip")

!cd /content/Day22-Track3-DPO-Alignment-Lab && \\
  zip -r /content/lab22-artifacts.zip \\
    adapters/sft-mini adapters/dpo \\
    data/pref data/eval \\
    submission/screenshots \\
    gguf \\
    -x "*.git*" -x "*checkpoint*" 2>/dev/null || true

if zip_path.exists():
    size_mb = zip_path.stat().st_size / 1e6
    print(f"Created lab22-artifacts.zip ({size_mb:.1f} MB)")
    files.download(str(zip_path))
else:
    print("Zip failed — download folders manually from Files panel (left sidebar)")
    print("Need: adapters/, data/eval/, submission/screenshots/")
'''
    verify = '''# Quick artifact checklist (mirrors scripts/verify.py core checks)
from pathlib import Path
import json

root = Path("/content/Day22-Track3-DPO-Alignment-Lab")
checks = [
    ("SFT adapter", root / "adapters/sft-mini/adapter_config.json"),
    ("DPO adapter", root / "adapters/dpo/adapter_config.json"),
    ("DPO metrics", root / "adapters/dpo/dpo_metrics.json"),
    ("Pref data", root / "data/pref/train.parquet"),
    ("Side-by-side", root / "data/eval/side_by_side.jsonl"),
    ("Judge results", root / "data/eval/judge_results.json"),
    ("SFT loss plot", root / "submission/screenshots/02-sft-loss.png"),
    ("DPO curves", root / "submission/screenshots/03-dpo-reward-curves.png"),
    ("Side-by-side plot", root / "submission/screenshots/04-side-by-side-table.png"),
]
print("Submission artifact checklist:\\n")
ok = 0
for label, path in checks:
    status = "OK" if path.exists() else "MISSING"
    if status == "OK":
        ok += 1
    print(f"  [{status:7s}] {label}: {path.relative_to(root)}")

if root.joinpath("adapters/dpo/dpo_metrics.json").exists():
    m = json.loads(root.joinpath("adapters/dpo/dpo_metrics.json").read_text())
    print(f"\\n  end_reward_gap = {m.get('end_reward_gap', 'n/a')}")

print(f"\\n{ok}/{len(checks)} core artifacts present.")
print("Fill submission/REFLECTION.md locally, then push public GitHub repo.")
'''
    return [
        md_cell(text, "epilogue-md"),
        code_cell(code, "epilogue-zip"),
        code_cell(verify, "epilogue-verify"),
    ]


OPTIONAL_STAGE_INSTALL: dict[str, tuple[str, str]] = {
    "05_merge_deploy_gguf.py": (
        "### NB5 setup — install llama-cpp-python (~1 min)\n\nUses a **prebuilt CUDA wheel** (no 10+ min compile). Skip if you only need core NB1–4.",
        '''import subprocess, sys
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "--progress-bar", "on",
    "llama-cpp-python>=0.3,<1.0",
    "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu124",
])
print("llama-cpp-python ready")
''',
    ),
    "06_benchmark.py": (
        "### NB6 setup — install lm-eval harness (~2 min)\n\nOnly needed for the benchmark bonus stage.",
        '''import subprocess, sys
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "--progress-bar", "on",
    "lm-eval[ifeval,math]>=0.4.5,<1.0",
])
print("lm-eval ready")
''',
    ),
}


def parse_jupytext_py(path: Path) -> list[dict]:
    """Split a Jupytext percent .py file into notebook cells."""
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"^# ---\n# jupyter:.*?\n# ---\n\n?", "", text, count=1, flags=re.DOTALL)

    pattern = re.compile(r"^# %%(?: \[markdown\])?\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    cells: list[dict] = [
        md_cell(f"---\n# Stage from `{path.name}`\n---", f"stage-{path.stem}"),
    ]
    if path.name in OPTIONAL_STAGE_INSTALL:
        md_text, code_text = OPTIONAL_STAGE_INSTALL[path.name]
        cells.append(md_cell(md_text, f"install-{path.stem}"))
        cells.append(code_cell(code_text, f"install-{path.stem}-code"))

    for i, match in enumerate(matches):
        is_md = "[markdown]" in match.group()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip("\n")
        if not chunk.strip():
            continue
        if is_md:
            lines = chunk.splitlines()
            md_text = "\n".join(
                ln[2:] if ln.startswith("# ") else (ln[1:] if ln.startswith("#") else ln)
                for ln in lines
            ).strip()
            if md_text:
                cells.append(md_cell(md_text))
        else:
            cells.append(code_cell(chunk.rstrip() + "\n"))
    return cells


def build_notebook(tier: str, repo_url: str) -> dict:
    cfg = TIER_CONFIG[tier]
    cells = preamble_cells(cfg, repo_url)

    for nb in NOTEBOOK_ORDER:
        py_path = NOTEBOOKS / nb
        if not py_path.exists():
            raise FileNotFoundError(py_path)
        cells.extend(parse_jupytext_py(py_path))

    cells.extend(epilogue_cells())

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "jupytext": {"main_language": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=["T4", "BIGGPU", "all"], default="all")
    parser.add_argument("--repo-url", default=DEFAULT_REPO)
    args = parser.parse_args()

    tiers = ["T4", "BIGGPU"] if args.tier == "all" else [args.tier]
    COLAB.mkdir(exist_ok=True)

    for tier in tiers:
        nb = build_notebook(tier, args.repo_url)
        out = COLAB / TIER_CONFIG[tier]["output"]
        out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        n_cells = len(nb["cells"])
        print(f"Wrote {out.relative_to(REPO)} ({n_cells} cells, tier={tier})")


if __name__ == "__main__":
    main()
