"""Backward-compat re-export — use _lab22_colab_utils.py."""

from _lab22_colab_utils import force_sdpa_if_turing, patch_dataset_map_no_mp

__all__ = ["force_sdpa_if_turing", "patch_dataset_map_no_mp"]
