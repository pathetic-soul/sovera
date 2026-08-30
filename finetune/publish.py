"""Merge the adapter, quantize, and register with Ollama (AGENTS.md §8.1, §4.2).

This is the "quantize" half of the job, and the step that makes a fine-tune
actually reachable from the workbench: an adapter sitting in models/adapters/
is not something core/agent.py can call. What it can call is an Ollama tag.

    merged bf16 -> f16 GGUF -> ollama create --quantize q4_K_M -> tag -> models.yaml

Why the GGUF hop, when an earlier draft of this file claimed Ollama could
import safetensors directly: **it cannot, for Qwen3, on Windows.** Measured on
Ollama 0.32.15:

    ollama create -q q4_K_M -f <safetensors dir>
        -> Error: unsupported architecture "Qwen3ForCausalLM"
    ollama create -q int4 --experimental -f <safetensors dir>
        -> Error: MLX init failed: failed to load MLX dynamic library
           (MLX is Apple-silicon only; the experimental importer needs it)
    ollama create -q q4_K_M ... --experimental
        -> Error: unsupported --quantize "q4_K_M": supported types are
           int4, int8, nvfp4, mxfp4, mxfp8

So neither importer works here: the classic one has no Qwen3 converter, and
the experimental one has Qwen3 but needs macOS and cannot produce K-quants.
Ollama *serves* Qwen3 GGUFs perfectly well — it is running one right now — so
only the safetensors->GGUF conversion is missing, and llama.cpp's
convert_hf_to_gguf.py does that in pure Python. No C++ build, no
llama-quantize binary: Ollama does the K-quant itself once the input is GGUF.

Q4_K_M and not int4/mxfp4 is deliberate. §5 budgets `driver` at 2.6 GB as
Q4_K_M and the base tag is Q4_K_M, so this keeps the fine-tuned model
byte-comparable to what the registry already validates against. Changing the
quant scheme would change the VRAM claim and make the before/after measurement
compare two different things at once.

Merge happens in bf16 on CPU, never against the 4-bit training copy: merging a
LoRA into an NF4-quantised base bakes the quantisation error into the weights
and then quantises *again*. 64 GB of system RAM (§4.2.2) is what makes the bf16
merge free here — it costs ~8 GB for the 4B, 0 GB of VRAM.

    python -m finetune.publish --model driver
    python -m finetune.publish --model driver --tag sovereign-driver:q4_K_M
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DIR = ROOT / "models" / "adapters"
MERGED_DIR = ROOT / "models" / "merged"
GGUF_DIR = ROOT / "models" / "gguf"
CONVERTER = ROOT / "tools_ext" / "llama.cpp" / "convert_hf_to_gguf.py"
QUANT = "q4_K_M"


def merge(model_id: str) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from finetune.stage import local_dir

    base_dir = local_dir(model_id)
    adapter = ADAPTER_DIR / model_id
    out = MERGED_DIR / model_id
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"no adapter at {adapter}. Run: python -m finetune.qlora --model {model_id}")

    print(f"[publish] loading {base_dir} in bf16 on CPU")
    model = AutoModelForCausalLM.from_pretrained(
        str(base_dir), dtype=torch.bfloat16, device_map="cpu",
    )
    print(f"[publish] applying adapter {adapter}")
    merged = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(base_dir)).save_pretrained(str(out))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e9
    print(f"[publish] merged bf16 -> {out} ({size:.1f} GB)")
    return out


def to_gguf(merged: Path, model_id: str) -> Path:
    """HF safetensors -> f16 GGUF, via llama.cpp's pure-Python converter."""
    if not CONVERTER.exists():
        raise SystemExit(
            f"{CONVERTER} is missing. Fetch it once, before arming the firewall:\n"
            "  git clone --depth 1 https://github.com/ggml-org/llama.cpp tools_ext/llama.cpp"
        )
    out = GGUF_DIR / f"{model_id}-f16.gguf"
    out.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable, str(CONVERTER), str(merged),
        "--outfile", str(out), "--outtype", "f16",
    ]
    print(f"[publish] converting to GGUF: {' '.join(argv[:3])} ...")
    proc = subprocess.run(argv, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"convert_hf_to_gguf.py failed ({proc.returncode})")
    print(f"[publish] f16 GGUF -> {out} ({out.stat().st_size / 1e9:.1f} GB)")
    return out


def to_ollama(gguf: Path, tag: str) -> None:
    """GGUF -> Q4_K_M tag. Ollama does the quantisation; no llama-quantize needed."""
    modelfile = gguf.parent / f"Modelfile.{tag.replace(':', '_')}"
    # Quote the path. This repo lives under "SIH 26117", and an unquoted FROM
    # splits at the space: Ollama reads the first fragment as a model name and
    # fails with "400 Bad Request: invalid model name".
    modelfile.write_text(f'FROM "{gguf.as_posix()}"\n', encoding="utf-8")
    argv = ["ollama", "create", tag, "--quantize", QUANT, "-f", str(modelfile)]
    print(f"[publish] {' '.join(argv)}")
    proc = subprocess.run(argv, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"ollama create failed ({proc.returncode})")


def yaml_block(model_id: str, tag: str) -> str:
    """§8.1: adding a model is editing models.yaml and clicking Reload — no code
    change. Print the block rather than writing it, so the live paste-and-reload
    in §14.3 stays the thing being demonstrated."""
    return f"""
  - id: {model_id}_ft
    backend: ollama
    ref: {tag}
    device: gpu
    vram_gb: 2.6          # re-measure with `ollama ps`; must stay under 5.2 (AGENTS.md 4.1)
    max_ctx: 8192
    capabilities: [reasoning, tool_call, planning]
    routes: [plan, qa, classify]
    priority: 0           # 0 outranks the base model on the same routes
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="driver")
    parser.add_argument("--tag", default="")
    parser.add_argument("--skip-merge", action="store_true",
                        help="reuse an existing models/merged/<id>")
    args = parser.parse_args()

    tag = args.tag or f"sovereign-{args.model}:{QUANT}"
    merged = MERGED_DIR / args.model if args.skip_merge else merge(args.model)
    gguf = to_gguf(merged, args.model)
    to_ollama(gguf, tag)

    print(f"\n[publish] registered as: {tag}")
    print("[publish] paste this into config/models.yaml, then click Reload registry:")
    print(yaml_block(args.model, tag))
    print(f"[publish] then measure it: python -m finetune.grounding_eval {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
