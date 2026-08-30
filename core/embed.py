"""Frozen local sentence encoder for the §9.2 dense router. CPU only, offline.

§9.2 specifies "embedding similarity vs config/routing_exemplars.jsonl (CPU,
<50 ms)". This module is the encoder half of that sentence. It exists because
the lexical scorer in `classifier.py` plateaued at 75.8% against a 90% gate,
which `ExemplarScorer` predicted in its own docstring.

**Why CPU and not the GPU.** VRAM is the binding constraint (§4.1: 5.2 GB
usable, with an 8B model resident). The router is the one component that can
give the budget back entirely. Measured on this machine: 21.7 ms median /
24.3 ms p95 for a single query, against the 50 ms budget — roughly 2x headroom
without spending a single byte of VRAM. Putting a 33M-parameter encoder on the
GPU to save 20 ms would be trading the scarce resource for the abundant one.

**Why not sentence-transformers.** It contacts the HF Hub at import and on
first load to resolve model cards, which is exactly the "phones home at import
or runtime" that AGENTS.md rule 1 forbids and §2.1 makes an invariant. Plain
`transformers` with `local_files_only=True` reads staged weights off disk and
opens no socket. The pooling that sentence-transformers would have done for us
is four lines below, and being able to read those four lines is worth more here
than the dependency.

**Why weights are staged, not downloaded.** §2.1: model weights are pre-staged
on disk, runtime never downloads. `python -m core.embed --stage` does the
fetch at build time, before the firewall is armed. If the directory is missing
at runtime this module raises with instructions rather than reaching for the
network.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# BAAI/bge-small-en-v1.5: 33.4M params, 384-dim, CLS-pooled. Chosen over
# bge-base (109M) because base cost ~3x the latency for no CV gain at this
# corpus size, and over MiniLM-L6 because bge scored better on the domain
# phrasings during selection. All three fit the budget; this one measured best.
ENCODER_REPO = "BAAI/bge-small-en-v1.5"
ENCODER_DIR = ROOT / "models" / "hf" / "bge-small-en-v1.5"
DIM = 384

# The 13420H is 4P+4E. Letting torch spawn a thread per logical core oversubscribes
# on a batch this small and adds scheduling latency instead of removing it.
_THREADS = min(8, os.cpu_count() or 4)


class EncoderUnavailable(RuntimeError):
    """Weights are not staged. Raised instead of silently downloading them."""


class Encoder:
    """Frozen encoder. Deterministic: same text in, same vector out, always.

    That determinism is what keeps the dense router legal under §2.3. "No LLM
    decides which model to use" rules out asking a generative model to pick a
    route; it does not rule out a fixed function from text to a vector, scored
    against hand-labelled exemplars. There is no sampling, no temperature and
    no generation anywhere in this path — the same prompt routes the same way
    on every run, which is the property §2.3 is actually protecting.
    """

    def __init__(self, path: Path = ENCODER_DIR) -> None:
        if not (path / "config.json").exists():
            raise EncoderUnavailable(
                f"encoder weights not staged at {path}. "
                f"Run: python -m core.embed --stage  (build time only, needs network)"
            )
        # Imported lazily: torch costs ~1 s to import, and the CLI, the audit
        # tools and the tests that never route should not pay it.
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        torch.set_num_threads(_THREADS)
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(path), local_files_only=True).eval()

    def encode(self, texts: Sequence[str], max_length: int = 64) -> np.ndarray:
        """Return L2-normalised CLS embeddings, shape (len(texts), DIM).

        max_length=64 tokens is not a guess: the longest routing exemplar is 34
        tokens and a work request that needs more than 64 to state its *intent*
        does not exist in this corpus. Truncating here is what keeps the p95
        inside the budget, and normalising makes the later cosine a plain dot
        product, so scoring 176 exemplars is one matmul.
        """
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        batch = self.tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        with self._torch.no_grad():
            hidden = self.model(**batch).last_hidden_state[:, 0]
            vecs = self._torch.nn.functional.normalize(hidden, dim=-1)
        return vecs.numpy().astype(np.float32)


_ENCODER: Encoder | None = None


def get_encoder() -> Encoder:
    """Process-wide singleton. Loading costs ~0.9 s; routing must not repeat it."""
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = Encoder()
    return _ENCODER


def stage(repo: str = ENCODER_REPO, dest: Path = ENCODER_DIR) -> Path:
    """Build-time only. The one function here allowed to touch the network."""
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    path = snapshot_download(
        repo, local_dir=str(dest),
        allow_patterns=["*.json", "*.txt", "*.safetensors", "tokenizer*"],
    )
    return Path(path)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="store_true", help="download weights (build time)")
    parser.add_argument("--bench", action="store_true", help="measure encode latency vs §9.2")
    args = parser.parse_args()

    if args.stage:
        print(f"staged -> {stage()}")
        return 0

    enc = get_encoder()
    if args.bench:
        import time

        probe = ["Summarise the shutdown inspection findings for the CDU overhead line"]
        enc.encode(probe)  # warm: first call pays lazy-init costs
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            enc.encode(probe)
            times.append((time.perf_counter() - t0) * 1000)
        print(f"encode: median {np.median(times):.1f} ms  p95 {np.percentile(times, 95):.1f} ms"
              f"  (§9.2 budget 50 ms, threads={_THREADS})")
    print(f"encoder ready: {ENCODER_DIR.name}, dim={DIM}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
