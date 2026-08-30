<!-- CANONICAL BASELINE. Tracked in git on purpose: every adapter is judged
     against these numbers, so they are a reference point, not build output.
     Regenerate with:  python -m finetune.benchmark --ref qwen3:4b-instruct --runs 3
     Held-out task set: seed 90210, 40 documents appearing in no training trace. -->

# Baseline — `qwen3:4b-instruct`, untuned

Measured 2026-08-27. **Two independent runs of 720 samples each**, pooled below.

**Pooled accuracy: 1231/1440 = 85.5%** (95% CI 83.6% – 87.2%)

## Replication — what the benchmark's own noise floor is

| | accuracy |
|---|---:|
| run A | 86.2% |
| run B | 84.7% |
| **spread** | **1.5 points** |

The two runs **do not separate from each other**, and every category overlaps —
which is the required result, since they are the same model. That 1.5-point
spread is this benchmark's noise floor at n=720.

For contrast, `finetune/gate.py` measured this same model at **5/8 and then 7/8**
on one task: a 25-point spread on identical inputs. Three adapters were accepted
or rejected on differences smaller than that. Anything below roughly 3 points
here is still not evidence.

## By category

| category | run A | run B | pooled | 95% CI | n |
|---|---:|---:|---:|---|---:|
| `unknown` | 100.0% | 99.2% | **99.6%** | 97.7% – 99.9% | 240 |
| `interval_ceiling` | 97.8% | 95.6% | **96.7%** | 90.7% – 98.9% | 90 |
| `derive_rate` | 94.2% | 90.8% | **92.5%** | 88.5% – 95.2% | 240 |
| `lookup` | 86.2% | 85.8% | **86.0%** | 82.7% – 88.9% | 480 |
| `derive_life` | 83.3% | 81.7% | **82.5%** | 77.2% – 86.8% | 240 |
| `interval_halflife` | 49.3% | 46.7% | **48.0%** | 40.2% – 55.9% | 150 |

## How to read this

**`interval_ceiling` and `interval_halflife` must be read together.** A model
that ignores the question and always answers "10 years" scores 100% on the
ceiling half and 0% on the other — merged, that reads as ~64% and looks like
partial competence. The base model's 96.7% / 48.0% split shows what it actually
does: it answers 10 almost always, and is right whenever 10 happens to be right.

**`interval_halflife` (48.0%) is the only category with headroom worth training.**
`unknown` is at 99.6% and `derive_rate` at 92.5%; training there can only cost
accuracy, which is how three previous adapters ended up worse than the model
they started from.

**The bar for shipping an adapter:** beat the pooled figure with
non-overlapping 95% intervals (`finetune/benchmark.py::separated`), and do not
regress `unknown` or `derive_rate`.
