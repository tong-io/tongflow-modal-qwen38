"""Modal download entry for qwen38.

Run:
  modal run download.py::download

Self-contained: do not import other local modules.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import modal



_cfg: dict[str, Any] = {}
_hf = _cfg.get("hf") if isinstance(_cfg.get("hf"), dict) else {}
REPO_ID = str(_hf.get("repoId") or "Qwen/Qwen3.8-27B")
REVISION = str(_hf.get("revision") or "")
MODEL_DIR = f"/models/{REPO_ID}"

volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(volume_name, create_if_missing=True)
model_downloader = modal.App("model_downloader")


@model_downloader.function(
    image=modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub==1.6.0"),
    volumes={"/models": volume},
    timeout=3600,
)
def _download() -> None:
    from huggingface_hub import snapshot_download

    if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
        print(f"Model already exists at {MODEL_DIR}, skipping")
        return

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
        revision=REVISION or None,
    )
    volume.commit()
    print(f"Model downloaded to {MODEL_DIR}")


@model_downloader.local_entrypoint()
def download() -> None:
    _download.remote()
