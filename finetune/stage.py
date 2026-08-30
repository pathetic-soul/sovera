"""Download the upstream checkpoints into models/hf/ (AGENTS.md §2.1).

BUILD TIME ONLY. This is the one part of the project that must touch the
network, and it belongs in the same pre-demo window as `docker build` for the
sandbox image. Run it BEFORE `sovereignty/firewall.ps1 -Enable`. Nothing in
core/, tools/ or backends/ ever fetches a weight — §2.1 is about *runtime*, and
that line stays intact.

Repo ids come from `hf_ref` in config/models.yaml, never from this file:
§8.1 makes that yaml the only place a model is named, and rule 2 in §2 forbids
hardcoding one anywhere else.

Disk: Qwen3-4B-Instruct-2507 is ~8 GB in bf16, Qwen2.5-Coder-7B-Instruct
~15 GB. These are the *unquantized* checkpoints — they are the training input,
not something we ever serve. What gets served is the Q4_K_M GGUF that
publish.py builds from them.

    python -m finetune.stage              # every model with an hf_ref
    python -m finetune.stage driver       # just one
"""

from __future__ import annotations

import sys
from pathlib import Path

from core.registry import Registry

ROOT = Path(__file__).resolve().parents[1]
HF_DIR = ROOT / "models" / "hf"

# Weights only. Skips the duplicate .bin/.pth copies some repos carry alongside
# safetensors, which would double the download for nothing.
ALLOW = ["*.safetensors", "*.json", "*.txt", "*.model"]
IGNORE = ["*.pth", "*.bin", "*.gguf", "original/*", "*.onnx"]


def local_dir(model_id: str) -> Path:
    return HF_DIR / model_id


def stage(model_id: str, hf_ref: str) -> Path:
    from huggingface_hub import snapshot_download

    target = local_dir(model_id)
    target.mkdir(parents=True, exist_ok=True)
    print(f"[stage] {model_id}: {hf_ref} -> {target}")
    snapshot_download(
        repo_id=hf_ref,
        local_dir=str(target),
        allow_patterns=ALLOW,
        ignore_patterns=IGNORE,
        max_workers=4,
    )
    size_gb = sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) / 1e9
    print(f"[stage] {model_id}: done, {size_gb:.1f} GB on disk")
    return target


def main(argv: list[str]) -> int:
    registry = Registry()
    wanted = argv or [
        spec.id for spec in registry.models.values() if spec.hf_ref
    ]
    for model_id in wanted:
        spec = registry.models.get(model_id)
        if spec is None:
            print(f"[stage] no model '{model_id}' in models.yaml", file=sys.stderr)
            return 2
        if not spec.hf_ref:
            print(f"[stage] {model_id} has no hf_ref; nothing to stage", file=sys.stderr)
            return 2
        stage(model_id, spec.hf_ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
