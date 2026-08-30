# Fine-tuning and quantization

Adapts `driver` to the agent loop and re-quantizes it back into Ollama.

> **AGENTS.md §3 lists fine-tuning as a non-goal.** That was overridden
> deliberately, and the override is recorded in §3 so the charter and the repo
> agree. Everything here is **build time** — it runs before
> `sovereignty/firewall.ps1 -Enable`, in the same window as
> `docker build -t sandbox-py:local sandbox\`. Nothing in `core/`, `tools/` or
> `backends/` trains, adapts or downloads anything at demo time, so §2.1 is
> intact.

---

## Why: measure first

The premise this started from was wrong, and measuring said so before any
training happened.

**Hypothesis:** a 4B needs fine-tuning to emit well-formed agent-loop JSON.
**Measured** (`python -m finetune.evaluate qwen3:4b-instruct`, n=75):

```
driver BASE   json_valid 100.0%   tool_match 97.3%   args_valid 97.3%   unknown 100.0%
```

No headroom. `driver` already drives the loop correctly, and `json_valid` is
partly enforced anyway — the loop uses Ollama's `format=json` (§12.2), so valid
JSON is constrained decoding, not a learned skill. Training against this would
have been several hours and ~10 GB of downloads to move a number that is
already at ceiling.

So the target moved to the gap leg 5 actually exposed (§16). On the real corpus
table, `python -m finetune.grounding_eval <ref>`:

| model | lookup | derived | total |
|---|---|---|---|
| `driver` — qwen3:4b-instruct | **10/10 · 100%** | 4/8 · 50% | 77.8% |
| `coder` — qwen2.5-coder:7b | **10/10 · 100%** | 2/8 · 25% | 66.7% |

Both models find the right cell **every time**. Both then get the arithmetic
wrong, and the 7B is worse at it than the 4B. `coder` quoted
`S6: 1.5, S7: 1.7, S8: 1.6` in its own source field and answered **5.4**
instead of 4.8. `driver` quoted `9.4, 8.9, 9.2` and answered **9.5** instead of
9.167.

Retrieval is not the problem. Mental arithmetic is.

You cannot fine-tune a working adder into a 4B. You *can* teach it to stop
doing arithmetic in its head and call `py_sandbox`, which is exact — and that
is what leg 5 was built for. 42% of the training traces now demonstrate exactly
that: read the document, restate each value with the grid and year it came
from, compute in the sandbox, report the sandbox's figure.

### Two bugs in the harness, not the model

Both were caught before any number was believed, and both are §12 in practice:

- The first prompt said `"value": <number>` and *"a plain number with no
  units"*. The 4B read that as **integer** — it answered `10` for `10.5 barg`
  and `5` for a sum its own source field spelled out as `4.8`. Fixed with
  explicit decimal examples and "never round to a whole number": lookup went
  80% → **100%**.
- The first prompt also said *"answer using ONLY the document"* and *"reply
  null if not stated"*. A corrosion rate is computed, not printed, so the model
  correctly answered "not stated" to every derived question. Fixed by
  explicitly licensing calculation: derived went 12.5% → **50%**.

A frontier model would have shrugged both off. That is the whole of §12.

---

## Pipeline

**Everything below runs on the RTX 4050.** Training is local by default and has
been since the vocab-recast fix; there is no step here that needs a rented GPU.

```powershell
git clone --depth 1 https://github.com/ggml-org/llama.cpp tools_ext/llama.cpp  # once
python -m finetune.stage driver           # HF safetensors -> models/hf/driver   (~8 GB)
python -m finetune.traceset               # 2,388 train / 426 eval traces
python -m finetune.qlora --model driver   # QLoRA          -> models/adapters/driver
python -m finetune.publish --model driver # merge -> GGUF -> q4_K_M -> ollama tag
python -m finetune.grounding_eval sovereign-driver:q4_K_M   # did it move?
```

### Ollama cannot import Qwen3 safetensors on Windows

An earlier draft of `publish.py` claimed it could, and that was wrong. Measured
on Ollama 0.32.15:

| attempt | result |
|---|---|
| `ollama create -q q4_K_M -f <safetensors>` | `unsupported architecture "Qwen3ForCausalLM"` |
| `... --experimental -q q4_K_M` | `unsupported --quantize`: only int4, int8, nvfp4, mxfp4, mxfp8 |
| `... --experimental -q int4` | `MLX init failed` — the experimental importer needs Apple silicon |

The classic importer has no Qwen3 converter; the experimental one has Qwen3 but
needs macOS and cannot emit K-quants. Ollama *serves* Qwen3 GGUFs perfectly
well — it is running one right now — so only the safetensors→GGUF step is
missing. llama.cpp's `convert_hf_to_gguf.py` does that in **pure Python**: no
C++ build, no `llama-quantize` binary, because Ollama does the K-quant itself
once the input is already GGUF.

**Chain validated on the base model before trusting it.** Converting the
untouched `models/hf/driver` through the same path produced a 2.5 GB Q4_K_M tag
— the same size as stock `qwen3:4b-instruct`. Doing this dry run first is what
turned two failures into cheap ones rather than discovering them after a
2.5-hour train and an 8 GB merge:

- `convert_hf_to_gguf.py` needs `sentencepiece` installed for vocab handling.
- The Modelfile `FROM` path **must be quoted**. This repo lives under
  `SIH 26117`; unquoted, Ollama splits at the space, reads `C:/.../SIH` as a
  model name, and fails with `400 Bad Request: invalid model name`.

It also gives the comparison a control: a Q4_K_M built by this pipeline from
the unmodified base should score like the stock tag, so any movement after
fine-tuning belongs to the adapter and not to conversion artifacts.

Q4_K_M rather than int4 is deliberate. §5 budgets `driver` at 2.6 GB *as
Q4_K_M* and the base tag is Q4_K_M, so the fine-tuned model stays comparable to
what `core/registry.py` already validates. Switching quant scheme would change
the VRAM claim and make the before/after measurement compare two things at once.

Model names come from `hf_ref` in `config/models.yaml`, never from these
scripts — §8.1 keeps that file the only place a model is named, and rule 2 in
§2 forbids hardcoding one anywhere else.

### VRAM — measured, and a Windows trap worth knowing

| | |
|---|---|
| weights resident after 4-bit + recast | 2.81 GB |
| peak during training | 5.61 GB |
| free at steady state | 0.55 GB |
| step time | ~40 s (228 steps, ~2.5 h) |

**Overcommitting VRAM does not fail on Windows — it gets slow.** The driver
(WDDM) spills the excess into shared system memory instead of raising OOM.
The first version of `qlora.py` left the frozen vocabulary matrices in fp32,
which is what `prepare_model_for_kbit_training` does by default. On Qwen3 the
vocabulary is 151,936, so `lm_head` alone is 389M params — 1.55 GB in fp32,
for weights LoRA never trains.

That run sat at 5896/6141 MiB and took **844 s per step, a 53-hour ETA, with
no error anywhere**. Recasting those frozen matrices to bf16 freed 0.78 GB and
took the same run to **36 s per step**. Identical output, 23x faster, one
allocation change.

The lesson is bigger than this script and is recorded in [AGENTS.md §17]:
on this platform a VRAM fault has no error, only a clock. `sovereignty/verify.ps1`
now checks free VRAM before the demo for the same reason — if Chrome takes
800 MB while `writer` is resident, inference will not crash, it will just start
taking minutes on stage.

Levers to shrink further, in order: `--max-len` (logits scale with it — at seq
1150 the fp32 logit tensor alone is ~700 MB), then `LORA_R`. Never raise the
batch size; accumulation is free, VRAM is not.

### GPU efficiency: measured, and mostly already optimal

The GPU is fully used — 100% utilisation, 5.8 GB resident, 2.56 GHz boost,
62 C during training. (Windows Task Manager shows the **3D** engine by default;
CUDA work lives on **Compute_0**, so a saturated run reads as ~0% there. Use
`nvidia-smi`.)

Effective throughput is **7.4 TFLOPS** (2.57e14 FLOPs/step over 35 s), roughly
20-30% of this card's bf16 tensor peak. The limiter is batch size 1, which
starves the tensor cores; batch size cannot rise because VRAM is the binding
constraint (§4.1).

Every lever, measured on an idle machine:

| config | traces kept | peak VRAM | s/step | verdict |
|---|---|---|---|---|
| r=16, len 2048, ckpt ON | all | 5.60 GB | **34.6** | current, best |
| r=8, len 2048 | all | 5.48 GB | 35.1 | -0.13 GB, no speed gain |
| r=16, len 1024 | 371 (**241 dropped**) | 5.19 GB | 34.8 | bad trade: 39% of data for 0 s |
| r=8, len 1024 | 371 (**241 dropped**) | 5.06 GB | 35.0 | same |
| ckpt **OFF** | — | — | — | **hard OOM** in bitsandbytes ops.cu:93 |

Step time is flat at ~35 s across every config that runs. We are at the compute
floor, not memory-bound — cutting peak VRAM from 5.61 to 5.06 GB bought zero
seconds. Gradient checkpointing is mandatory; without it the run does not fit.

Two things that are CPU-bound **by design**, not misconfiguration:
`convert_hf_to_gguf.py` and Ollama's K-quantisation have no GPU path in
llama.cpp, and the bf16 merge is deliberately on CPU to avoid baking NF4 error
into the weights.

Sequence packing was considered and rejected: packing to 2048 tokens would help
tensor-core utilisation, but logits scale with sequence length, and
2048 x 151,936 x 4 B is ~1.24 GB of logits alone — straight into the spill
regime.

**The only lever that actually mattered was epochs.** Loss converges by ~0.4
epochs, so 1 epoch instead of 3 cut the run from 2h16m to 44 min at no quality
cost.

### Held-out eval, on the 6 GB card

`qlora.py` used to train with `save_strategy="no"` and never open
`finetune/data/eval.jsonl`. It kept the **last** adapter, not the best one, and
had no signal to stop on — which is how a run reached training loss 0.0016 and
shipped an adapter that scored worse than base. The 2x T4 notebook
(`finetune/kaggle_kernel.py`) was written to fix that, and it did, but none of
what it fixed actually needed a second GPU. It needed the held-out split to be
read.

It is read now, locally:

```powershell
python -m finetune.qlora --model driver                 # eval on, defaults below
python -m finetune.qlora --model driver --eval-n 0      # old behaviour, last adapter
```

| flag | default | what it does |
|---|---|---|
| `--eval-n` | 96 | held-out traces to score; `0` disables eval, best-checkpoint and early stopping |
| `--eval-steps` | 25 | optimiser steps between evals |
| `--patience` | 3 | evals with no `eval_loss` improvement before stopping |

`--epochs` now defaults to **1.0**, not 3.0 — loss converges by ~0.4 epochs, so
three epochs over this corpus is what drove loss to the floor in the first place.

Two things that are load-bearing rather than tidy:

- **`prediction_loss_only=True`.** Without it the Trainer concatenates every
  eval batch's logits to hand back, and at 2048 x 151,936 x 4 B that is ~1.24 GB
  per trace accumulating across the eval set. On Windows that does not raise
  OOM (§17) — it spills to shared memory and the eval "works" at minutes per
  batch, the same silent VRAM fault the vocab recast fixed.
- **Strided eval subsample.** `eval.jsonl` is written grouped by trace family,
  so `eval_rows[:96]` would score one composition bucket and report it as the
  held-out loss. The stride is deterministic, so the number is comparable
  between runs.

### The eval set was leaking, and it is fixed

Wiring best-checkpoint selection revealed that the split was not held out:

```
train 2,587 rows / 1,585 unique prompts
eval    453 rows /   373 unique prompts
OVERLAP: 136 prompts = 36.5% of the eval set
```

`traceset.split()` stratified by family — correct — but then shuffled **traces**
and cut. The generators emit several traces per user prompt, so a prompt's
variants landed on both sides. Scored against that, "best checkpoint by held-out
loss" is partly a memorisation score, which is precisely the failure being
guarded against.

The cut is now taken on the user prompt, inside each family, so a prompt's
variants stay together. `tests/test_finetune.py` asserts the two files share no
user prompt, so the leak cannot come back silently. The same bug existed in the
superseded `dataset.py` and was fixed there too.

### Where the Kaggle notebook still fits

`finetune/kaggle_kernel.py` and `finetune/kaggle_sync.py` are kept, and are now
**a fallback rather than the quality path**. The two reasons they existed —
`max_len 1024` dropping 63% of the corpus, and 3 epochs driving memorisation —
are both fixed locally: `MAX_LEN` is 2048 and keeps every trace, epochs default
to 1, and eval-driven early stopping is in `qlora.py`.

Reach for the T4 notebook when the 4050 is unavailable or busy, or for a
sweep that would take too long serially. Note what changes on Turing, all of
which the notebook already handles: no bf16 (fp16 + gradient scaler), no
FlashAttention-2 (`sdpa`), and `device_map={"": local_rank}` to pin a whole
model per card rather than sharding one across both.

`kaggle_sync.py` is unrelated to training — it publishes datasets and models to
Kaggle, private by default, and refuses anything under `data/corpus/`.

### Design notes

- **Loss on assistant turns only**, masked explicitly in `qlora.py` rather than
  via a trainer flag. A silent masking bug trains the model to predict the
  system prompt and looks like a healthy run right until the model is useless.
  Traces whose chat template does not extend as a strict prefix are dropped
  rather than trained on a misalignment.
- **Merge in bf16 on CPU**, never against the 4-bit training copy — that would
  bake quantisation error into the weights and then quantise again. The 64 GB
  of system RAM in §4.2.2 makes this free.
- **Q4_K_M** because §5 budgets `driver` at 2.6 GB and the base tag is Q4_K_M.
  A larger quant invalidates the VRAM claim `core/registry.py` validates.
- **No llama.cpp build**, but the repo is needed for its Python converter — see
  above. `tools_ext/` is gitignored; clone it once, before arming the firewall.
- `report_to=[]` in the trainer — no wandb, no telemetry, no phone home (§2.1).

---

## VERDICT: ship `calc`, do not ship the adapter

The whole exercise resolves to one table. Same task (leg-5 remaining life on the
real corpus document), same harness, 8 trials each:

| | without `calc` | with `calc` |
|---|---|---|
| base `qwen3:4b-instruct` | 1/8 correct, 7/8 step-cap | **7/8 correct** |
| `sovereign-driver-v2` (fine-tuned) | 0/8 correct, 8/8 wrong | **0/8 correct** |

`tools/calc.py` took the untouched base model from 1/8 to 7/8. The adapter did
not benefit at all — and the reason matters more than the number.

**Fine-tuning on a fixed tool set makes the model brittle to tool changes.**
v2 was trained when only fs_read, py_sandbox and doc_write existed. Given a new
`calc` tool it does not reach for it. It either forces the arithmetic back
through py_sandbox with the wrong formula (2.87, 21.79, 2.77 years across
trials) or refuses outright:

> "UNKNOWN - the API 510 formula is not in workspace/refs/, so I cannot ..."

That refusal is the §12.7 UNKNOWN behaviour, learned well and then misapplied:
the formula was never supposed to come from the document. The base model, being
general, simply used the tool it was offered.

So the adapter is not merely unhelpful here, it is a liability: it locks the
model to the tool repertoire it was trained on, and this project expects to add
tools (§6 lists kb_search and fs_write as still unbuilt). Both tags are kept for
the record; neither is registered in config/models.yaml.

**What the fine-tune did prove**, and it is worth keeping in the write-up: the
sandbox-routing habit transferred cleanly (loop tool_match 86.4% -> 100%, every
trial produced running code with real stdout). The training pipeline works. The
target was wrong -- a 4B does not need to be taught to prefer a calculator, it
needs a calculator whose formulas cannot be got wrong.

## What this can and cannot do

**Can:** teach the habit of routing every numeric derivation through
`py_sandbox`; keep tool selection and flat-schema arguments correct; hold the
UNKNOWN behaviour of §12.7; write a thought that names the actual tag instead
of copying the few-shot line verbatim.

**Cannot:** make the model read a table more accurately (already 100%), or do
arithmetic in its head. If the sandbox habit does not transfer, the honest
answer is a `calc` tool (§6) that makes the arithmetic path non-optional rather
than learned — and that is a smaller, more reliable change than any adapter.

Report the numbers here as their own metric. **They are not the §13 router
gate** — that measures `core/routing/`, a deterministic lexical scorer
with no model in it (§2.3), which no amount of fine-tuning can move.
