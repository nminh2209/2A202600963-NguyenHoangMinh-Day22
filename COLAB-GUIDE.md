# Colab Quick Start — Day 22 DPO Lab

Use this path if your laptop GPU has **less than 12 GB VRAM** (e.g. RTX 3050 4GB) or local setup is too heavy. **No local install required.**

## 1. Open the notebook

[![Open T4 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VinUni-AI20k/Day22-Track3-DPO-Alignment-Lab/blob/main/colab/Lab22_DPO_T4.ipynb)

Or manually: upload `colab/Lab22_DPO_T4.ipynb` to [colab.research.google.com](https://colab.research.google.com).

## 2. Set GPU runtime

**Runtime → Change runtime type → T4 GPU** → Save.

## 3. Add OpenAI key (recommended)

**Runtime → Change runtime type** is step 2; for the judge:

1. Click the **🔑 Secrets** icon in the left sidebar (or **View → Secrets**).
2. Add secret: name `OPENAI_API_KEY`, value `sk-...`
3. In notebook **section B**, the cell loads it automatically via `userdata.get`.

Without a key, NB4 still runs but uses manual rubric placeholders.

## 4. Run all cells

| Section | What it does | Time (T4) |
|---------|--------------|-----------|
| A | Clone repo to `/content/Day22-Track3-DPO-Alignment-Lab` | 1 min |
| B | Load API keys | instant |
| C | **Core** `pip install` (no llama-cpp compile) | **2–4 min** |
| D | GPU check + folders | instant |
| Stages 1–4 | **Core grading** (SFT → data → DPO → eval) | ~30 min |
| Stage 5 | GGUF deploy (**bonus +6**) — installs llama-cpp first | ~10 min |
| Stage 6 | Benchmark (**bonus +8**) — installs lm-eval first | ~30 min |
| Z | Zip + download artifacts | 1 min |

### `torchcodec` / `from unsloth import FastLanguageModel` fails?

Colab's PyTorch 2.10 + a stray `torchcodec` package conflict (unsloth [#5446](https://github.com/unslothai/unsloth/issues/5446)). This lab is **text-only** — torchcodec is not needed.

1. **Runtime → Restart session**
2. Re-run **A → B → C** (updated notebook uninstalls torchcodec automatically)
3. Or run manually before NB1:

```python
!pip uninstall -y torchcodec
from unsloth import FastLanguageModel
print("OK")
```

If import still fails: `!pip install -U unsloth unsloth_zoo` then uninstall torchcodec again.

### `NotImplementedError: memory_efficient_attention_backward` on T4?

Newer **xformers** Flash-Attention kernels need **sm_80+** (Ampere); Colab **T4 is sm_75**. Training backward fails with `requires device with capability >= (8, 0)`.

1. **Runtime → Restart session**
2. Re-run **A → B → C → D** (section C uninstalls xformers on Turing; section D forces Unsloth SDPA)
3. Re-run NB3 section 4 (train cell)

Or manually in a cell before `trainer.train()`:

```python
import sys
sys.path.insert(0, "/content/Day22-Track3-DPO-Alignment-Lab/notebooks")
from _lab22_colab_utils import force_sdpa_if_turing
force_sdpa_if_turing()
```

### `RecursionError` on `ds.map`?

Section D (or NB1) was re-run and stacked a broken `Dataset.map` patch. The old patch used a **global** `_orig_dataset_map` that gets overwritten on re-run → infinite recursion.

1. **Runtime → Restart session**
2. Re-run **A → B → C → D** once (section D is now idempotent)
3. Continue from NB1

### `save_pretrained_merged` / `merge_and_unload` + `save_pretrained` fails (NB5)?

Colab **4-bit + newer transformers** cannot save merged HF weights (`NotImplementedError` in `reverse_op`). **Do not** use the `merge_and_unload()` workaround — it hits the same error.

**Bonus grading only needs `gguf/*.gguf`.** Re-run NB5:

1. **Runtime → Restart session** (if you already ran `merge_and_unload`)
2. Re-run **A → D**, then **NB5 section 1** (load SFT + DPO adapters — do **not** merge)
3. Run **section 2** (`export_gguf_from_adapters` / `save_pretrained_gguf`) — ~3–5 min first time

```python
from unsloth import FastLanguageModel
from _lab22_colab_utils import force_sdpa_if_turing, export_gguf_from_adapters

force_sdpa_if_turing()
export_gguf_from_adapters(model, tokenizer, GGUF_DIR, quantization="q4_k_m")
```

1. **Runtime → Interrupt execution**
2. **Runtime → Restart session** → re-run A, B, then use the updated section C (split install)
3. Old one-liner hung on `llama-cpp-python` compiling from source (~15–30 min) or reinstalling torch (~1 GB)

**Core only:** stop after stage 4 if you're short on time — still worth 100 core pts if REFLECTION + screenshots are done.

## 5. Bring artifacts back to your laptop

Run **section Z** at the bottom. It downloads `lab22-artifacts.zip` containing:

- `adapters/sft-mini/` and `adapters/dpo/`
- `data/pref/` and `data/eval/`
- `submission/screenshots/` (loss curves, reward plots, side-by-side table)

Unzip into your local clone of this repo, then:

1. Fill `submission/REFLECTION.md` with your numbers and interpretation.
2. Optionally run `make verify` on a machine with Python (no GPU needed for verify).
3. Push **public** GitHub repo → paste URL in VinUni LMS.

## 6. Using your own fork

In section A, change:

```python
REPO_URL = "https://github.com/YOUR-USERNAME/Day22-Track3-DPO-Alignment-Lab.git"
```

Or set Colab secret `LAB22_REPO_URL` before running.

## Rebuild Colab notebook from sources

If you edit `notebooks/*.py`, regenerate the stitched Colab file (stdlib only):

```bash
python scripts/build_colab_notebook.py
```

## Why not local Windows?

| Issue | Your machine | Colab T4 |
|-------|--------------|----------|
| VRAM | RTX 3050 **4 GB** | **16 GB** |
| Lab minimum | 12 GB for 3B DPO | ✓ fits |
| Install size | ~5–8 GB venv + torch | nothing local |

See [`HARDWARE-GUIDE.md`](HARDWARE-GUIDE.md) for full tier math.
