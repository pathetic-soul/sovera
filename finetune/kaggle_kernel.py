"""Generate and push the Kaggle notebook that trains `driver` on 2x T4.

Build time only (AGENTS.md §3). This pushes a notebook; it does not train here.

    python -m finetune.kaggle_kernel --write      # write the .ipynb locally
    python -m finetune.kaggle_kernel --push       # write, then push to Kaggle

WHY A SECOND TRAINER AT ALL
---------------------------
`finetune/qlora.py` is single-GPU by construction and correct for the RTX 4050.
Kaggle gives two T4s, and the difference is not "the same script, faster" —
three things change, and getting any of them wrong produces a run that either
crashes at model load or silently trains on one card:

1.  **Turing has no bf16.** `pick_precision()` in qlora.py already returns fp16
    on sm_75, which is what makes that file portable. fp16 has fp32's precision
    but a far smaller exponent range, so it needs a gradient scaler —
    `TrainingArguments(fp16=True)` wires one up. bf16 on a T4 fails at load.

2.  **No FlashAttention-2.** It requires Ampere (sm_80+). The notebook asks for
    `attn_implementation="sdpa"`, which is the fastest kernel a T4 can run.

3.  **`device_map` must pin, not shard.** This is the trap. Under DDP each of
    the two processes must load a *complete* model onto *its own* card:
    `device_map={"": local_rank}`. The reflex `device_map="auto"` splits one
    model across both GPUs, and then DDP wraps the split — which gives no
    speed-up, doubles the communication, and usually deadlocks. A 4B at 4-bit
    is ~2.8 GB, so a whole copy fits in 16 GB with room for activations.

DDP rather than FSDP or DeepSpeed: the model already fits on one card, so
there is nothing to shard. Sharding a model that fits buys communication
overhead and no memory. DDP replicates and all-reduces gradients — for two
cards on one host that is close to linear scaling.

WHAT "BEST QUALITY" MEANS HERE, CONCRETELY
------------------------------------------
The last three adapters all scored *worse* than the base model, and the
overnight v3 run diagnosed why: loss 0.0016 on 226 traces is memorisation. Two
of those numbers are artefacts of the 6 GB card, and the T4s remove both:

  - `max_len` 1024 dropped 63% of the corpus. 2048 keeps all of it.
  - 3 epochs over a small set is what drives loss to ~0. Loss converges by
    ~0.4 epochs, so this runs 1 and stops on the eval curve, not the clock.

So the notebook holds out a real eval split, evaluates on a step interval,
keeps the best checkpoint by *eval* loss rather than the last one, and stops
early when eval loss stops improving. A training loss near zero is treated as
a failure signal, not a success — it is printed with a warning next to it.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "finetune" / "kaggle"
SLUG = "sovereign-driver-qlora-t4x2"
# Kaggle derives the kernel id from the title's slug, so keep the title
# slug-clean and identical to SLUG — an em-dash or parentheses make the two
# disagree and Kaggle warns that the id "does not resolve".
TITLE = "Sovereign Driver QLoRA T4x2"

# The dataset slug pushed by finetune/kaggle_sync.py. The notebook mounts it
# read-only at /kaggle/input/<slug>/.
TRACES_DATASET = "agent-loop-tool-calling-traces"

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

# Files shipped into the notebook verbatim. Embedding the real modules rather
# than a paraphrase is the whole point: the Kaggle run must score with the same
# code the repo scores with, or the two numbers are not comparable. This session
# already produced one silent failure from a divergent copy of the trace
# encoder; there is no second copy of the scorer.
SHIPPED = ("finetune/benchmark.py", "finetune/corpus_inspection.py")


# --------------------------------------------------------------------------
# the training script the notebook writes and torchrun executes
# --------------------------------------------------------------------------

TRAIN_SCRIPT = r'''
"""QLoRA on 2x T4 under DDP. Written by finetune/kaggle_kernel.py."""
import json, os, sys, time
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments, EarlyStoppingCallback)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE      = os.environ.get("BASE_MODEL", "__BASE_MODEL__")
DATA_DIR  = Path(os.environ.get("DATA_DIR", "/kaggle/input/__TRACES__"))
OUT       = Path(os.environ.get("OUT_DIR", "/kaggle/working/adapter"))
MAX_LEN   = int(os.environ.get("MAX_LEN", "2048"))
EPOCHS    = float(os.environ.get("EPOCHS", "1"))
LORA_R    = int(os.environ.get("LORA_R", "16"))
LR        = float(os.environ.get("LR", "2e-4"))

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
IS_MAIN    = LOCAL_RANK == 0


def log(*a):
    if IS_MAIN:
        print(*a, flush=True)


# ---------------------------------------------------------------- data

def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line)["messages"])
    return rows


def _ids(result):
    """Normalise apply_chat_template's return to a flat list of token ids.

    NOT optional. transformers 4.x returns list[int]; 5.x returns a
    BatchEncoding, where len() is the number of KEYS (2), not the number of
    tokens. Without this, the strict-prefix guard below compares dict slices,
    every trace fails it, encode() returns None for all of them, and training
    starts on an EMPTY dataset. The failure is silent — the run reports
    "train 0 eval 0" and proceeds.

    finetune/qlora.py carries the same helper for the same reason; the two must
    not drift apart.
    """
    ids = result["input_ids"] if hasattr(result, "keys") else result
    if ids and isinstance(ids[0], list):        # a batch dimension on some versions
        ids = ids[0]
    return list(ids)


def encode(tok, messages):
    """Tokenise one trace, masking loss to assistant turns only.

    This is deliberately the SAME algorithm as finetune/qlora.py::encode, not a
    reimplementation. An earlier version here walked every message and sliced
    messages[:i] from i=0, which transformers 5.x rejects outright ("Cannot
    apply chat template to an empty conversation"). Walking only assistant
    turns avoids that by construction, because a trace always opens with system
    and user.

    Masking is explicit rather than delegated to a trainer flag: a silent
    masking bug trains the model to predict the system prompt and looks like a
    perfectly healthy run until the adapter is useless. A trace whose template
    does not extend as a strict prefix is dropped rather than trained on a
    misalignment.
    """
    full = _ids(tok.apply_chat_template(messages, tokenize=True))
    if len(full) > MAX_LEN:
        return None
    labels = [-100] * len(full)

    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prompt = _ids(tok.apply_chat_template(
            messages[:i], tokenize=True, add_generation_prompt=True))
        upto = _ids(tok.apply_chat_template(messages[:i + 1], tokenize=True))
        if full[:len(prompt)] != prompt or full[:len(upto)] != upto:
            return None
        labels[len(prompt):len(upto)] = full[len(prompt):len(upto)]

    if all(l == -100 for l in labels):
        return None
    return {"input_ids": full, "labels": labels}


class Traces(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


def collate(pad_id):
    def fn(batch):
        n = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "labels": [], "attention_mask": []}
        for b in batch:
            pad = n - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [pad_id] * pad)
            out["labels"].append(b["labels"] + [-100] * pad)
            out["attention_mask"].append([1] * len(b["input_ids"]) + [0] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}
    return fn


# ---------------------------------------------------------------- model

def main():
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    visible = torch.cuda.device_count()

    # Report the ACTUAL device count, not WORLD_SIZE. An earlier version printed
    # x{WORLD_SIZE} — the value torchrun was told to use — so a single-GPU box
    # logged "x2" and then died with "invalid device ordinal". The log described
    # the assumption, not the machine.
    #
    # bf16 needs Ampere (sm_80+). torch.cuda.is_bf16_supported() returns True on
    # a Tesla P100 (sm_60), which has no bf16 whatsoever, so the capability is
    # checked directly rather than asked for.
    use_bf16 = cap >= (8, 0)
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    log(f"[gpu] {name} sm_{cap[0]}{cap[1]} x{visible} visible "
        f"(world_size {WORLD_SIZE})  precision="
        f"{'bf16' if use_bf16 else 'fp16 + grad scaler'}")

    if cap < (7, 0):
        raise SystemExit(
            f"ABORT: {name} is sm_{cap[0]}{cap[1]}. The Kaggle PyTorch build "
            f"supports sm_70 and above, so this GPU cannot run the training "
            f"at all.\n\n"
            f"Kaggle assigned a default accelerator. In the notebook editor set "
            f"Session options -> Accelerator to 'GPU T4 x2' and re-run. The API "
            f"cannot request it: machine_shape silently normalises any unknown "
            f"value back to a generic 'Gpu'.")

    if WORLD_SIZE > visible:
        raise SystemExit(
            f"ABORT: launched with world_size {WORLD_SIZE} but only {visible} GPU(s) "
            f"are visible. Each DDP rank pins a whole model to its own card, so "
            f"rank {visible} would index a device that does not exist "
            f"('invalid device ordinal').")

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    train_rows = [r for r in (encode(tok, m) for m in load(DATA_DIR / "train.jsonl")) if r]
    eval_rows  = [r for r in (encode(tok, m) for m in load(DATA_DIR / "eval.jsonl"))  if r]
    log(f"[data] train {len(train_rows)}  eval {len(eval_rows)}  max_len {MAX_LEN}")
    raw_train = len(load(DATA_DIR / "train.jsonl"))
    if not train_rows or not eval_rows:
        raise SystemExit(
            f"ABORT: encoding produced {len(train_rows)} train / {len(eval_rows)} eval rows "
            f"from {raw_train} traces. Every trace failed the strict-prefix check, which "
            f"usually means apply_chat_template returned a BatchEncoding and _ids() is missing.")
    if len(train_rows) < raw_train * 0.9:
        log(f"!! WARNING kept only {len(train_rows)}/{raw_train} traces "
            f"({len(train_rows)/raw_train:.0%}) — raise MAX_LEN or check the chat template.")

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )

    # THE DDP TRAP: pin a whole model to THIS rank's card. device_map="auto"
    # would shard one model across both GPUs and then DDP would wrap the shard.
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        quantization_config=quant,
        device_map={"": LOCAL_RANK},
        attn_implementation="sdpa",      # FlashAttention-2 needs sm_80+
        dtype=dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # prepare_model_for_kbit_training upcasts every frozen norm and embedding to
    # fp32. On Qwen3 the vocabulary is 151,936, so lm_head alone is 389M params
    # — 1.55 GB in fp32, for weights LoRA never touches. Recast them.
    for n_, p in model.named_parameters():
        if not p.requires_grad and p.dtype == torch.float32:
            p.data = p.data.to(dtype)

    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_R * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    if IS_MAIN:
        model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir="/kaggle/working/ckpt",
        num_train_epochs=EPOCHS,
        # BATCH SIZE IS NOT 1 HERE, AND THAT IS THE POINT.
        # finetune/qlora.py uses 1 because on the 6 GB RTX 4050 VRAM is the
        # binding constraint (AGENTS.md §4.1) — its README says so explicitly.
        # A T4 has 16 GB and that constraint simply does not apply. Carrying the
        # 1 across was inheriting a limit from the wrong machine.
        #
        # Budget at our measured max sequence of 1583 tokens, vocab 151,936,
        # fp16 logits: weights 2.8 GB + logits (b x 1583 x 151936 x 2 B, plus a
        # backward copy) + checkpointed activations.
        #     b=1  ~4.5 GB      b=4  ~7.7 GB      b=8  ~11.5 GB
        # 4 leaves comfortable headroom on 16 GB and stops starving the tensor
        # cores, which is what batch 1 was doing.
        per_device_train_batch_size=int(os.environ.get("BATCH", "4")),
        per_device_eval_batch_size=4,
        # Global batch stays 16 (4 x 2 accum x 2 cards), identical to the old
        # 1 x 8 x 2 — so this is a throughput change, not a hyperparameter
        # change, and the loss curve stays comparable to previous runs.
        gradient_accumulation_steps=2,
        # Sequences run 993 tokens median against a 1583 max. Random batching
        # pads every sequence to its batch's longest, so a short trace batched
        # with a long one wastes most of its compute on padding. Grouping by
        # length makes batches homogeneous and is free.
        group_by_length=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,       # best by EVAL loss, not the last step
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="paged_adamw_8bit",
        report_to=[],                      # no wandb, no telemetry (§2.1)
        ddp_find_unused_parameters=False,  # LoRA + checkpointing: required
        dataloader_num_workers=2,
        seed=26117,
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=Traces(train_rows),
        eval_dataset=Traces(eval_rows),
        data_collator=collate(tok.pad_token_id),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    if IS_MAIN:
        OUT.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(OUT))
        tok.save_pretrained(str(OUT))
        hist = [h for h in trainer.state.log_history if "eval_loss" in h]
        best = min((h["eval_loss"] for h in hist), default=float("nan"))
        last_train = next((h["loss"] for h in reversed(trainer.state.log_history)
                           if "loss" in h), float("nan"))
        peak = max(torch.cuda.max_memory_allocated(i) / 1e9
                   for i in range(torch.cuda.device_count()))
        summary = {
            "elapsed_s": round(elapsed, 1),
            "peak_gb_per_card": round(peak, 2),
            "world_size": WORLD_SIZE,
            "best_eval_loss": best,
            "final_train_loss": last_train,
            "train_traces": len(train_rows),
            "eval_traces": len(eval_rows),
            "max_len": MAX_LEN, "epochs": EPOCHS, "lora_r": LORA_R,
        }
        Path("/kaggle/working/summary.json").write_text(json.dumps(summary, indent=2))
        log("\n" + json.dumps(summary, indent=2))

        # The diagnosis that the previous three adapters failed on. A training
        # loss this low means the model reproduces its traces near-exactly, and
        # every proxy metric will look excellent while end-to-end accuracy
        # collapses. Say so here rather than discovering it in the gate.
        if last_train == last_train and last_train < 0.05:
            log(f"\n!! WARNING train loss {last_train:.4f} indicates MEMORISATION."
                f"\n!! v2 and v3 both looked like this and both scored worse than base."
                f"\n!! Prefer the best-eval checkpoint and consider more data, not more epochs.")
        if best == best and last_train == last_train and best > last_train * 5:
            log(f"\n!! WARNING eval loss {best:.4f} >> train loss {last_train:.4f}: overfitting.")


if __name__ == "__main__":
    main()
'''


# --------------------------------------------------------------------------
# notebook assembly
# --------------------------------------------------------------------------

def _shipped() -> str:
    """Concatenate the shipped repo modules into one importable file.

    corpus_inspection is included because benchmark.build_tasks() generates its
    held-out documents from it — the benchmark is only meaningful if Kaggle
    builds exactly the documents the repo would.

    `from __future__` must be the first statement in a file, so the directives
    are stripped from both modules and one is re-emitted at the top. Without
    that the concatenation raises SyntaxError on import, which is exactly what
    a naive join produced.
    """
    parts = []
    for rel in SHIPPED:
        text = (ROOT / rel).read_text(encoding="utf-8")
        # Both modules land in one flat file, so the intra-package import of
        # generate() would fail — the name is already defined above it.
        text = text.replace("from finetune.corpus_inspection import generate", "")
        text = text.replace("from __future__ import annotations", "")
        parts.append(f"# ===== {rel} =====\n{text}")
    # corpus_inspection must be defined before benchmark calls generate().
    joined = "\n\n".join(reversed(parts))
    return "from __future__ import annotations\n\n" + joined


def _b64(text: str) -> str:
    """Base64 of the training script, for embedding in a notebook cell.

    The script carries its own docstrings, so it cannot be pasted into a
    triple-quoted literal — the first inner \"\"\" would close it.
    """
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _md(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code(text: str) -> dict[str, Any]:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def build_notebook(username: str) -> dict[str, Any]:
    script = (TRAIN_SCRIPT
              .replace("__BASE_MODEL__", BASE_MODEL)
              .replace("__TRACES__", TRACES_DATASET))

    # Cells are assembled as a list so the ordering is readable at a glance:
    # environment, deps, data, train, benchmark, export.
    cells = [
        _md(f"""# {TITLE}

QLoRA fine-tune of `{BASE_MODEL}` on the agent-loop trace corpus using **both
T4 GPUs** via DistributedDataParallel, followed by a held-out benchmark against
the untouched base model.

Set the accelerator to **GPU T4 x2** and turn **Internet on** before running.

Three things this gets right that a naive port does not:

| | why it matters on a T4 |
|---|---|
| `fp16` + gradient scaler | Turing (sm_75) has no bf16 — asking for it fails at model load |
| `attn_implementation="sdpa"` | FlashAttention-2 needs Ampere (sm_80+) |
| `device_map={{"": LOCAL_RANK}}` | pins a whole model per rank; `"auto"` shards one model across both cards and deadlocks under DDP |

Quality settings target the failure the previous three adapters hit — a
training loss of 0.0016 on 226 traces, which is memorisation. `max_len=2048`
keeps the whole corpus instead of dropping 63% of it, one epoch replaces three,
and the best checkpoint is chosen by **eval** loss with early stopping."""),

        _md("""## 1. Environment — check the accelerator BEFORE anything expensive

A previous run failed here in a way worth preventing: Kaggle assigned a single
**Tesla P100 (sm_60)**, the notebook launched `torchrun --nproc_per_node=2`
anyway, and rank 1 indexed a GPU that did not exist. It died with
`invalid device ordinal` — *after* downloading 8 GB of weights.

The accelerator cannot be requested through the API. `machine_shape` silently
normalises any unknown value back to a generic `Gpu`, so **GPU T4 x2 must be
selected in the notebook editor** under Session options."""),
        _code("""!nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
import torch, sys

n = torch.cuda.device_count()
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| devices", n)
for i in range(n):
    cap = torch.cuda.get_device_capability(i)
    print(" ", i, torch.cuda.get_device_name(i), "sm_%d%d" % cap,
          "bf16" if cap >= (8, 0) else "no bf16 -> fp16 + grad scaler")

assert n >= 1, "no GPU at all. Session options -> Accelerator -> GPU T4 x2"
cap0 = torch.cuda.get_device_capability(0)
if cap0 < (7, 0):
    raise SystemExit(
        f"STOP: {torch.cuda.get_device_name(0)} is sm_{cap0[0]}{cap0[1]}, which this "
        f"PyTorch build does not support (needs sm_70+).
"
        f"Set Session options -> Accelerator to 'GPU T4 x2' and re-run.
"
        f"Stopping now rather than after an 8 GB download.")
if n < 2:
    print(f"
NOTE: only {n} GPU visible. Training will run single-process; "
          f"the DDP speed-up needs 'GPU T4 x2'.")
"""),

        _md("""## 2. Dependencies

Pinned to the **same major versions this repo runs locally**, not to floors.
The API moved between transformers 4.x and 5.x in two ways this code depends
on: `from_pretrained(dtype=...)` was `torch_dtype=`, and
`TrainingArguments(eval_strategy=...)` was `evaluation_strategy=`. A `>=4.44`
floor would let Kaggle resolve to 4.x and fail at model load."""),
        _code("""!pip install -q -U "transformers>=5.0,<6" "peft>=0.17" "bitsandbytes>=0.45" \\
    "accelerate>=1.0" "datasets>=3.0" 2>&1 | tail -2
import transformers, peft, torch
print("transformers", transformers.__version__, "| peft", peft.__version__,
      "| torch", torch.__version__)
assert int(transformers.__version__.split(".")[0]) >= 5, \\
    "this notebook uses the transformers 5.x API (dtype=, eval_strategy=)"
"""),

        _md(f"## 3. Data\n\nMounted read-only from the `{TRACES_DATASET}` dataset."),
        _code(f"""from pathlib import Path
DATA = Path("/kaggle/input/{TRACES_DATASET}")
assert DATA.exists(), "attach the '{TRACES_DATASET}' dataset to this notebook"
for f in sorted(DATA.iterdir()):
    print(f.name, f"{{f.stat().st_size/1e6:.2f}} MB")
"""),

        _md("""## 4. The training script

Written to disk so `torchrun` can spawn one process per GPU — a notebook cell
cannot be the DDP entry point, since each rank has to import a real module.

Carried as base64 rather than a quoted literal, because the script contains its
own triple-quoted docstrings and a raw triple-quoted wrapper would end at the
first inner docstring."""),
        _code(
            "import base64, pathlib\n"
            f'SRC_B64 = "{_b64(script)}"\n'
            'src = base64.b64decode(SRC_B64).decode("utf-8")\n'
            'pathlib.Path("/kaggle/working/train_ddp.py").write_text(src, encoding="utf-8")\n'
            'compile(src, "train_ddp.py", "exec")   # fail here, not inside torchrun\n'
            'print("wrote train_ddp.py", len(src), "bytes")\n'),

        _md("""## 5. Train on both GPUs

`--nproc_per_node=2` starts one process per card. Each holds a full 4-bit copy
(~2.8 GB of 16 GB) and DDP all-reduces the LoRA gradients."""),
        _code(f"""import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["BASE_MODEL"] = "{BASE_MODEL}"
os.environ["DATA_DIR"]   = "/kaggle/input/{TRACES_DATASET}"
os.environ["MAX_LEN"]    = "2048"
os.environ["EPOCHS"]     = "1"
os.environ["BATCH"]      = "4"

# nproc is DETECTED, not assumed. The previous version hardcoded 2 and Kaggle
# handed the run a single P100, so rank 1 indexed a GPU that did not exist and
# torchrun died after a 40 GB model download.
import torch
NPROC = max(1, torch.cuda.device_count())
print(f"launching torchrun with --nproc_per_node={{NPROC}} "
      f"({{NPROC}} GPU(s) detected: "
      f"{{[torch.cuda.get_device_name(i) for i in range(NPROC)]}})")
if NPROC == 1:
    print("NOTE: single GPU. This still trains, just without the DDP speed-up.")

!cd /kaggle/working && torchrun --nproc_per_node=$NPROC --master_port=29500 train_ddp.py
"""),

        _md("## 6. Training result"),
        _code("""import json, pathlib
p = pathlib.Path("/kaggle/working/summary.json")
print(json.dumps(json.loads(p.read_text()), indent=2) if p.exists()
      else "no summary - training did not finish")
!ls -la /kaggle/working/adapter 2>/dev/null || echo "no adapter written"
"""),

        _md("""## 7. Benchmark

Scores the trained adapter against the untouched base on **240 held-out tasks**
over 40 documents generated with a different seed from the training corpus.
Zero overlap, asserted in the repo's own tests.

This replaces a gate that scored **one task over eight trials**. That gate
measured the same base model at 7/8 in one session and 5/8 in another, so it
could not separate a real regression from sampling noise — and three adapters
were judged on it.

`finetune/benchmark.py` is shipped in verbatim rather than reimplemented, so
these numbers are directly comparable to a local run."""),
        _code(
            "import base64, pathlib, sys\n"
            f'BENCH_B64 = "{_b64(_shipped())}"\n'
            'pathlib.Path("/kaggle/working/shipped.py").write_text(\n'
            '    base64.b64decode(BENCH_B64).decode("utf-8"), encoding="utf-8")\n'
            'sys.path.insert(0, "/kaggle/working")\n'
            'print("benchmark harness ready")\n'),

        _code(f"""import sys, json, pathlib, torch
sys.path.insert(0, "/kaggle/working")
from shipped import build_tasks, run_benchmark, report, wilson, separated
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "{BASE_MODEL}"
tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

def loader(adapter=None):
    m = AutoModelForCausalLM.from_pretrained(
        BASE, device_map={{"": 0}}, dtype=torch.float16, attn_implementation="sdpa")
    if adapter:
        m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    m.eval()
    return m

def make_generate(model):
    def gen(messages):
        ids = tok.apply_chat_template(messages, tokenize=True,
                                      add_generation_prompt=True, return_tensors="pt")
        ids = ids.to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=320, temperature=0.2,
                                 do_sample=True, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
    return gen

TASKS = build_tasks()
RUNS = 3
print(f"{{len(TASKS)}} held-out tasks x {{RUNS}} runs = {{len(TASKS)*RUNS}} samples per model")
"""),

        _code("""base_model = loader(None)
base_res = run_benchmark("base " + BASE, make_generate(base_model), TASKS, RUNS)
del base_model; torch.cuda.empty_cache()
print(f"base accuracy {base_res.accuracy:.1%}")
"""),

        _code("""tuned = loader("/kaggle/working/adapter")
cand_res = run_benchmark("adapter (this run)", make_generate(tuned), TASKS, RUNS)
del tuned; torch.cuda.empty_cache()
print(f"adapter accuracy {cand_res.accuracy:.1%}")
"""),

        _md("""## 8. Verdict, and the file to copy back into the repo

`separated` is the honest bar: if the two 95% intervals overlap, this benchmark
**cannot tell the models apart**, and that is the result — not a rounding
decision in favour of whichever number is larger."""),
        _code("""md = report(cand_res, base_res)
print(md)

payload = {
    "source": "kaggle t4x2",
    "base": {"ref": base_res.ref, "accuracy": base_res.accuracy,
             "ci95": wilson(base_res.correct, base_res.n),
             "tally": base_res.tally(), "samples": base_res.n},
    "candidate": {"ref": cand_res.ref, "accuracy": cand_res.accuracy,
                  "ci95": wilson(cand_res.correct, cand_res.n),
                  "tally": cand_res.tally(), "samples": cand_res.n},
    "by_category": {k: {"correct": c, "n": n}
                    for k, (c, n) in cand_res.by_category().items()},
    "separated": separated(cand_res, base_res),
}
pathlib.Path("/kaggle/working/benchmark.json").write_text(json.dumps(payload, indent=2))
pathlib.Path("/kaggle/working/benchmark.md").write_text(md, encoding="utf-8")

print("\\n" + ("SEPARATED - the difference exceeds sampling noise"
      if payload["separated"] else
      "NOT SEPARATED - intervals overlap; this benchmark cannot tell them apart"))
print("\\nDownload benchmark.json + benchmark.md and run, in the repo:")
print("  python -m finetune.import_benchmark <path-to-benchmark.json>")
"""),

        _md("""## 9. Next

Download `/kaggle/working/adapter/` and run the local publish chain:

```powershell
python -m finetune.publish --model driver     # merge -> GGUF -> q4_K_M -> ollama
```

**Do not register a tag in `config/models.yaml` until the benchmark separates
from base.** Every proxy metric said v2 was excellent; end-to-end said 0/8."""),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def metadata(username: str) -> dict[str, Any]:
    return {
        "id": f"{username}/{SLUG}",
        "title": TITLE,
        "code_file": f"{SLUG}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [f"{username}/{TRACES_DATASET}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def write(username: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nb = OUT_DIR / f"{SLUG}.ipynb"
    nb.write_text(json.dumps(build_notebook(username), indent=1), encoding="utf-8")
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata(username), indent=2), encoding="utf-8")
    return nb


def push_kernel(client: Any, username: str, public: bool = False) -> None:
    """Push the notebook to Kaggle. Called by finetune/kaggle_sync.py."""
    write(username)
    meta_path = OUT_DIR / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["is_private"] = not public
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    client.kernels_push(str(OUT_DIR))
    print(f"pushed -> https://www.kaggle.com/code/{username}/{SLUG}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="write the notebook locally")
    ap.add_argument("--push", action="store_true", help="write, then push to Kaggle")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()

    if not (args.write or args.push):
        ap.error("pass --write or --push")

    os.environ.pop("KAGGLE_KEY", None)          # a KGAT token is not a legacy key
    if args.push:
        from finetune.kaggle_sync import api
        client = api()
        username = json.loads(
            (Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]
        push_kernel(client, username, public=args.public)
    else:
        username = json.loads(
            (Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]
        print(f"wrote {write(username).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
