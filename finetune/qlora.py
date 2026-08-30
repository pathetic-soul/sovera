"""QLoRA fine-tune of `driver` on the agent-loop traces (AGENTS.md §3 override).

§3 lists fine-tuning as a non-goal. That was overridden deliberately; the
decision is recorded in §3 itself so the charter and the repo agree.

BUILD TIME ONLY, like finetune/stage.py and `docker build` for the sandbox.
Training happens before the firewall is armed. Nothing at demo time trains,
downloads or adapts anything — §2.1 is about runtime and stays intact.

VRAM — MEASURED, not estimated (§4.1, §17)
------------------------------------------
The first estimate here said ~3.9 GB and was wrong by 1.7 GB. What actually
happens on Qwen3-4B, measured on the RTX 4050:

    weights resident after 4-bit + recast    2.81 GB
    peak during training                     5.61 GB
    free at steady state                     0.55 GB
    step time                                 ~40 s

The gap between 2.81 and 5.61 is activations and, mostly, **logits**: Qwen3's
vocabulary is 151,936, so one sequence of ~1150 tokens produces a logit tensor
of ~700 MB in fp32 before the loss is even computed.

Read §17 before tuning any of this. On Windows, overcommitting VRAM does not
raise OOM — the driver spills to shared system memory and the run merely gets
20-25x slower. The first version of this script left the frozen vocab matrices
in fp32 (what prepare_model_for_kbit_training does by default) and ran at
**844 s/step, a 53-hour ETA, with no error of any kind**. Recasting them to
bf16 freed 0.78 GB and took it to 36 s/step. Same output, 23x faster.

So: a training run that is "working" but slow is the symptom of a VRAM fault
on this platform. Check the headroom line this script prints before believing
anything else.

Levers if it needs to shrink further, in order: --max-len (logits and
activations both scale with it), then LORA_R. Do not raise the batch size —
accumulation is free, VRAM is not.

Loss is taken on assistant turns only. The masking is written out explicitly
rather than delegated to a trainer flag, because a silent masking bug trains
the model to predict the *system prompt* and looks like a successful run right
up until the model is useless.

    python -m finetune.qlora                  # train driver
    python -m finetune.qlora --model coder    # or the 7B (tighter, see MAX_LEN)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from finetune.runstats import RunLog

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "finetune" / "data"
ADAPTER_DIR = ROOT / "models" / "adapters"

MAX_LEN = 2048
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
EPOCHS = 1.0
LR = 2e-4
ACCUM = 8
# Every linear in the block. Attention-only LoRA underfits format-following
# tasks like this one; the MLP projections are where the JSON shape lands.
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_traces(path: Path) -> list[list[dict[str, str]]]:
    return [
        json.loads(line)["messages"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _ids(result: Any) -> list[int]:
    """Normalise apply_chat_template's return to a flat list of token ids.

    transformers 4.x returns list[int]; 5.x returns a BatchEncoding, where
    `len()` is the number of keys (2), not the number of tokens. Encoding
    against that silently produced zero usable traces — the strict-prefix guard
    in encode() caught it and dropped everything rather than training on
    misaligned labels, which is the entire reason that guard exists.
    """
    ids = result["input_ids"] if hasattr(result, "keys") else result
    if ids and isinstance(ids[0], list):  # a batch dimension on some versions
        ids = ids[0]
    return list(ids)


def encode(tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, list[int]] | None:
    """Tokenise one trace, masking everything that is not an assistant turn.

    Returns None if the chat template does not extend as a strict prefix at
    some turn — that would put the label window on the wrong tokens, and
    dropping the example is far better than training on a silent misalignment.
    """
    full = _ids(tokenizer.apply_chat_template(messages, tokenize=True))
    if len(full) > MAX_LEN:
        return None
    labels = [-100] * len(full)

    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prompt = _ids(tokenizer.apply_chat_template(
            messages[:i], tokenize=True, add_generation_prompt=True
        ))
        upto = _ids(tokenizer.apply_chat_template(messages[: i + 1], tokenize=True))
        if full[: len(prompt)] != prompt or full[: len(upto)] != upto:
            return None
        labels[len(prompt) : len(upto)] = full[len(prompt) : len(upto)]

    if all(label == -100 for label in labels):
        return None
    return {"input_ids": full, "labels": labels}


def pick_precision(torch: Any) -> tuple[Any, str]:
    """bf16 where the hardware has it, fp16 where it does not.

    Not a portability nicety — it is the difference between this script running
    on a Kaggle T4 and not running at all. bf16 needs Ampere (sm_80+); the
    RTX 4050 is Ada (sm_89) and has it, Kaggle's free T4 is Turing (sm_75) and
    does not, and asking for bf16 there fails at model load.

    The two are not interchangeable beyond the name. bf16 carries fp32's
    exponent range, so it does not overflow and needs no loss scaling; fp16 has
    a much smaller range and will silently produce inf gradients without a
    scaler. `TrainingArguments(fp16=True)` wires that scaler up, which is why
    the flag is passed through rather than just the dtype.
    """
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bf16"
    return torch.float16, "fp16"


def loss_printer(runlog: Any = None) -> Any:
    """Print the loss with an explicit flush.

    The built-in logging works fine — an earlier version of this docstring
    claimed `report_to=[]` suppressed it, and that was wrong. The real problem
    was plain stdout buffering: with output redirected to a file, nothing
    appears until the buffer fills, so a run looks silent for its first hour
    and reads exactly like a logging failure.

    `flush=True` is the whole point of this callback. Watching a long training
    run through a log file is the normal case here, and a loss curve you can
    only read after the run has finished cannot tell you to stop early.
    """
    from transformers import TrainerCallback

    class Printer(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any,
                   logs: dict[str, float] | None = None, **kw: Any) -> None:
            if logs and "eval_loss" in logs:
                # A train loss at the floor with eval loss rising is
                # memorisation, and it is the failure mode that shipped the
                # last three adapters. Say so while the run can still be killed.
                print(f"[qlora] step {state.global_step}/{state.max_steps} "
                      f"EVAL loss {logs['eval_loss']:.4f}", flush=True)
            elif logs and "loss" in logs:
                warn = "  <-- at the floor: memorising, not learning"                     if logs["loss"] < 0.01 else ""
                print(f"[qlora] step {state.global_step}/{state.max_steps} "
                      f"loss {logs['loss']:.4f} "
                      f"grad_norm {logs.get('grad_norm', float('nan')):.3f} "
                      f"lr {logs.get('learning_rate', 0):.2e}{warn}", flush=True)
                if runlog is not None:
                    # max_memory_allocated, not memory_allocated: the peak is what
                    # decides whether we crossed the §4.1 ceiling, and the instant
                    # reading at log time is always well below it.
                    import torch as _torch

                    runlog.log(
                        step=state.global_step, total_steps=state.max_steps,
                        loss=logs["loss"], grad_norm=logs.get("grad_norm"),
                        lr=logs.get("learning_rate"),
                        vram_gb=_torch.cuda.max_memory_allocated() / 1e9,
                    )

    return Printer()


class Traces:
    """Plain list-backed dataset. torch.utils.data is enough; `datasets` would
    be a second dependency for a 612-row table held in RAM."""

    def __init__(self, rows: list[dict[str, list[int]]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, list[int]]:
        return self.rows[i]


def collate(pad_id: int) -> Any:
    import torch

    def fn(batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        width = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, mask = [], [], []
        for b in batch:
            gap = width - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * gap)
            labels.append(b["labels"] + [-100] * gap)
            mask.append([1] * len(b["input_ids"]) + [0] * gap)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }

    return fn


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="driver", help="model id from models.yaml")
    parser.add_argument("--epochs", type=float, default=EPOCHS)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--lora-r", type=int, default=LORA_R,
                        help="LoRA rank; halving it frees adapter+grad+optimiser memory")
    parser.add_argument("--no-grad-ckpt", action="store_true",
                        help="disable gradient checkpointing: ~25%% less compute "
                             "(no recompute forward) but more activation memory")
    parser.add_argument("--eval-n", type=int, default=96,
                        help="held-out traces to evaluate on; 0 disables eval, "
                             "best-checkpoint selection and early stopping")
    parser.add_argument("--eval-steps", type=int, default=25,
                        help="optimiser steps between evals")
    parser.add_argument("--patience", type=int, default=3,
                        help="stop after N evals with no eval_loss improvement")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="stop after N optimiser steps; use a few to time a step first")
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    from finetune.stage import local_dir

    globals()["MAX_LEN"] = args.max_len
    base = local_dir(args.model)
    if not (base / "config.json").exists():
        print(f"{base} is not staged. Run: python -m finetune.stage {args.model}")
        return 2

    tokenizer = AutoTokenizer.from_pretrained(str(base))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_id = tokenizer.pad_token_id
    assert pad_id is not None, "tokenizer has no pad token even after the eos fallback"

    train_rows, dropped = [], 0
    for messages in load_traces(DATA / "train.jsonl"):
        encoded = encode(tokenizer, messages)
        if encoded is None:
            dropped += 1
        else:
            train_rows.append(encoded)
    supervised = sum(sum(1 for x in r["labels"] if x != -100) for r in train_rows)
    total = sum(len(r["input_ids"]) for r in train_rows)
    print(f"[qlora] {len(train_rows)} traces ({dropped} dropped), "
          f"{supervised}/{total} tokens supervised ({100*supervised/max(total,1):.1f}%)")
    if not train_rows:
        print("[qlora] nothing to train on")
        return 1

    # finetune/data/eval.jsonl has always existed and was never read. Training
    # without it means keeping the LAST adapter rather than the best one, which
    # is how a previous run reached loss 0.0016 — memorisation, shipped.
    #
    # Strided rather than head-sliced: eval.jsonl is written grouped by trace
    # kind, so eval_rows[:n] would score one composition bucket and call it the
    # held-out loss. The stride is deterministic, so the number stays comparable
    # between runs.
    eval_rows: list[dict[str, list[int]]] = []
    if args.eval_n > 0:
        every = [r for r in (encode(tokenizer, m)
                             for m in load_traces(DATA / "eval.jsonl")) if r is not None]
        stride = max(1, len(every) // args.eval_n)
        eval_rows = every[::stride][:args.eval_n]
        print(f"[qlora] {len(eval_rows)} eval traces (stride {stride} of {len(every)})")
    else:
        print("[qlora] eval disabled: keeping the last adapter, not the best")

    compute_dtype, precision = pick_precision(torch)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[qlora] {device_name} -> {precision}")

    quant = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(base),
        quantization_config=quant,
        dtype=compute_dtype,
        device_map={"": 0},
        attn_implementation="sdpa",  # flash-attn has no reliable Windows wheel
    )
    model.config.use_cache = False
    use_ckpt = not args.no_grad_ckpt
    model = prepare_model_for_kbit_training(  # type: ignore[no-untyped-call]
        model, use_gradient_checkpointing=use_ckpt)

    # prepare_model_for_kbit_training upcasts every non-quantised parameter to
    # fp32. On Qwen3 that is ruinous: the vocabulary is 151,936, so lm_head and
    # embed_tokens are ~389M params EACH — 1.55 GB apiece in fp32, for weights
    # LoRA never trains. They are frozen, so bf16 is all they need.
    #
    # This is not a micro-optimisation. Leaving them fp32 put the run at
    # 5896/6141 MiB and Windows/WDDM does not raise OOM at that point — the
    # driver silently spills into shared system memory, so training still
    # "works" at 844 s/step (53 hours) and looks like slowness rather than a
    # VRAM fault. §4.1's 5.2 GB ceiling has no alarm on this platform; the only
    # signal is the clock. Norms stay fp32 — they are tiny and help stability.
    recast = 0
    for name, param in model.named_parameters():
        if param.dtype == torch.float32 and ("lm_head" in name or "embed_tokens" in name):
            recast += param.numel()
            param.data = param.data.to(compute_dtype)
    if recast:
        print(f"[qlora] recast {recast / 1e6:.0f}M frozen vocab params fp32 -> {precision} "
              f"(~{recast * 2 / 1e9:.2f} GB saved)")

    peft_model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=LORA_DROPOUT,
        bias="none", task_type="CAUSAL_LM", target_modules=TARGETS,
    ))
    peft_model.print_trainable_parameters()
    # Rebind through Any: `model` was the bare HF class, and PeftModel is a
    # wrapper rather than a subclass. Everything below only calls the
    # interface both share, so the alternative is threading two names
    # through the rest of the function for no gain.
    model = cast(Any, peft_model)

    # Report against the §4.1 ceiling before a single step runs. On Windows the
    # driver will not tell us we have overcommitted — it just gets slow — so
    # this number is the early warning that the clock would otherwise give us
    # 14 minutes later.
    resident = torch.cuda.memory_allocated() / 1e9
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb, total_gb = free_bytes / 1e9, total_bytes / 1e9
    # Activations + fp32 logits over a 151,936 vocab cost ~2.8 GB on top of the
    # weights at seq ~1150. Measured across four configs; see the table below.
    projected = resident + 2.8
    print(f"[qlora] weights resident {resident:.2f} GB | free {free_gb:.2f} / {total_gb:.2f} GB "
          f"| projected peak ~{projected:.2f} GB")
    if projected > total_gb:
        print("[qlora] WARNING: projected peak exceeds VRAM. On Windows this does NOT "
              "raise OOM (§17) — it spills to shared system memory and each step takes "
              "minutes instead of seconds. Lower --max-len before letting this run.")

    out = ADAPTER_DIR / args.model
    # transformers 5.x removed `warmup_ratio` and kept `warmup_steps`, so the
    # 3% is computed here. More honest anyway — the step count is visible.
    steps_per_epoch = max(1, len(train_rows) // ACCUM)
    total_steps = int(steps_per_epoch * args.epochs)
    warmup = max(1, int(total_steps * 0.03))
    print(f"[qlora] {steps_per_epoch} steps/epoch, {total_steps} total, {warmup} warmup")

    runlog = RunLog(args.model, {
        "lora_r": args.lora_r, "max_len": args.max_len, "epochs": args.epochs,
        "traces": len(train_rows), "total_steps": total_steps,
        "grad_ckpt": use_ckpt, "optim": "paged_adamw_8bit",
    })
    print(f"[qlora] live stats -> {runlog.html_path}", flush=True)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out / "checkpoints"),
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=ACCUM,
            gradient_checkpointing=use_ckpt,
            learning_rate=LR,
            lr_scheduler_type="cosine",
            warmup_steps=warmup,
            logging_steps=10,
            # Eval, best-checkpoint selection and early stopping — the only
            # thing the 2x T4 notebook did that this file could not. None of it
            # needed a second GPU; it needed the held-out split to be read.
            #
            # prediction_loss_only is mandatory, not tidiness. Without it the
            # Trainer concatenates every eval batch's logits to return them, and
            # at 2048 x 151,936 x 4 B that is ~1.24 GB per trace accumulating
            # across the whole eval set. On Windows that does not OOM (§17), it
            # spills to shared memory and the eval "works" at minutes per step.
            eval_strategy="steps" if eval_rows else "no",
            eval_steps=args.eval_steps,
            per_device_eval_batch_size=1,
            prediction_loss_only=True,
            eval_on_start=bool(eval_rows),   # the untrained baseline to beat
            # save_steps must equal eval_steps for load_best_model_at_end.
            save_strategy="steps" if eval_rows else "no",
            save_steps=args.eval_steps,
            save_total_limit=1,
            load_best_model_at_end=bool(eval_rows),
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=(precision == "bf16"),
            fp16=(precision == "fp16"),
            optim="paged_adamw_8bit",
            report_to=[],          # §2.1: no telemetry, no wandb, no phone home
            max_grad_norm=0.3,
        ),
        train_dataset=Traces(train_rows),
        eval_dataset=Traces(eval_rows) if eval_rows else None,
        data_collator=collate(pad_id),
        callbacks=[loss_printer(runlog)]
        + ([EarlyStoppingCallback(early_stopping_patience=args.patience)]
           if eval_rows else []),
    )
    try:
        trainer.train()
    except BaseException as exc:
        # Including KeyboardInterrupt: a run killed by hand is a result too, and
        # the dashboard should say so rather than sitting on "running" forever.
        runlog.finish("stopped" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=f"{type(exc).__name__}: {exc}"[:400],
                      peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3))
        raise

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[qlora] adapter -> {out}")
    # log_history[-1] is an eval record whenever eval is on, and those carry no
    # train_runtime — search back for the summary rather than trusting the tail.
    runtime = next((m["train_runtime"] for m in reversed(trainer.state.log_history)
                    if "train_runtime" in m), 0.0)
    per_step = runtime / max(trainer.state.global_step, 1)
    best = getattr(trainer.state, "best_metric", None)
    if best is not None:
        print(f"[qlora] best eval_loss {best:.4f} "
              f"@ {trainer.state.best_model_checkpoint} (this is what was saved)")
    print(f"[qlora] peak VRAM {peak:.2f} GB against the 5.2 GB ceiling (AGENTS.md 4.1)")
    print(f"PROBE r={args.lora_r} max_len={args.max_len} ckpt={use_ckpt} traces={len(train_rows)} "
          f"peak_gb={peak:.2f} s_per_step={per_step:.1f}")
    runlog.finish("done", peak_gb=round(peak, 3), s_per_step=round(per_step, 2),
                  adapter=str(out), steps=trainer.state.global_step)
    print(f"[qlora] stats -> {runlog.html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
