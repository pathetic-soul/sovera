# Sprint board — Sovereign Workbench

Execution granularity. `STATUS.md` says where the project is, `ROADMAP.md` says
what ships; this file says what is committed *this week* and what closed last.

Sprint data lives in `docs/sprint_data.json`. Re-run the analysis with:

```powershell
$S = "$HOME\.claude\skills\scrum-master\scripts"
.venv\Scripts\python $S\velocity_analyzer.py       docs\sprint_data.json --format text
.venv\Scripts\python $S\sprint_health_scorer.py    docs\sprint_data.json --format text
.venv\Scripts\python $S\retrospective_analyzer.py  docs\sprint_data.json --format text
```

---

## Read this before quoting any score below

The health scorer returns **55.9 / 100, grade Fair** — and that number is mostly
an artifact of missing data, not a measurement. Four of its six dimensions
defaulted to 50/100 because a one-person, six-day build has no ceremony
attendance, no second sprint to compare against, and no resolution-time series.

**Two dimensions scored on real evidence. Only these are worth acting on:**

| Dimension | Score | Measured | Target |
|---|---|---|---|
| Commitment reliability | 52.5 — Poor | **62.5%** of committed scope delivered | >85% |
| Scope stability | 50.0 — Poor | **25%** unplanned scope added mid-sprint | <15% |
| Story completion | 85.0 — Good | 7 of 10 items fully done | high |

Velocity forecasting is **unavailable**: `insufficient_historical_data`, one
sprint against a three-sprint minimum. The retrospective analyzer needs three
retrospectives and refuses on one. Both refusals are correct and are left
standing rather than worked around — the same rule `gates.py` enforces on the
acceptance gates: *a number nobody could measure is not a number.*

---

## Sprint 1 — closed 2026-08-29

**"Containment and the agent loop"** · 2026-08-24 → 2026-08-29 · 6 working days

Unit of delivery is one **demo leg** (AGENTS.md §14), not an abstract story
point. Legs are what the charter commits to and what can be verified done.

| | |
|---|---|
| Committed | 8 legs |
| Delivered | **5 legs** + 2 unplanned items |
| Carried over | 3 legs (6, 7, 8) |
| Blockers opened | 4 — two resolved, two still open |

**Delivered:** sovereignty panel with the red button and a hash-chained audit
log · model registry and deterministic router · live model addition · the agent
loop with three tools producing a real `.docx` · sandboxed code execution with
`--network none`.

**Unplanned, and both justified:** `tools/calc.py`, built in response to the
arithmetic defect found during leg 5 — it moved the base model from 1/8 to 7/8
correct. And the `finetune/` QLoRA track, which the charter listed as a non-goal
until the override was recorded in §3.

That is where the 25% scope change comes from. It was reactive, not casual: one
item fixed a defect that would have been visible on stage, the other tested a
hypothesis and returned a negative result. **Do not treat this as a process
failure to be squeezed out** — treat it as the reason the leg forecast slipped,
and plan the next sprint with room for it rather than assuming it away.

### Retrospective

**Went well**
- Measured before building. The fine-tuning hypothesis was tested and rejected before hours were spent on it.
- Held-out discipline held: the blend weight was tuned on the training split alone, held-out scored once at the end.
- A negative result was kept and published rather than buried.
- `gates.py` turned eight hand-checked acceptance criteria into one runnable command that refuses to let SKIP count as PASS.

**To improve**
- **Zero commits across the entire sprint.** Every deliverable exists in exactly one place.
- The golden path has never been run end to end against its own 6-minute gate.
- Documentation drifted from measurement — §16 still records the pre-encoder router number; the README carried a stale test count for days.
- The gate runner itself went unverified, so a broken gate reported SKIP for an unknown period.

The last two share one root cause: **claims were being maintained by hand in
several places at once.** `gates.py` is the structural fix, and it only works if
every quoted number comes from it.

---

## Sprint 2 — open

**"Make the demo survivable"** · 2026-08-30 → 2026-09-05 · scope freeze follows

**No new legs are committed.** Sprint 1 delivered five legs and left every
manual acceptance gate unexecuted; a sixth leg on top of an unrehearsed demo
makes the submission worse, not better. The charter is explicit: *ship three
legs deep, not six shallow.*

Committed — all five carry over from the Sprint 1 retro:

| ID | Item | Priority | Done when |
|---|---|---|---|
| AI-1 | Commit the repository, push to a private remote | **P0** | `git log` is non-empty and a remote has the work |
| AI-2 | Dress-rehearse the golden path, firewall armed, stopwatch running | **P0** | a wall-clock time is recorded against the 6-minute gate |
| AI-3 | ~~Expand router exemplars across the six weak classes~~ — done 2026-08-31: 176 → 209 exemplars, held-out misses 13 → 10, accuracy 80.3% → 86.7% (gate still 90%, not met) | P1 | `gates.py` reports a number above 80.3% — met |
| AI-4 | Refresh AGENTS.md §16 so the charter matches `gates.py` | P1 | no hand-typed metric survives in §16 |
| AI-5 | Measure `coder` and `driver` VRAM with `ollama ps` | P1 | both figures recorded in §5 |

**Stretch, only if all five close:** leg 6 (multimodal). Cheapest remaining leg —
the vision model is already staged and routed to. Measure its VRAM first; it is
the second real spend against the 5.2 GB ceiling.

**Explicitly not in this sprint:** leg 7. Two new packages and an OCR stack do
not fit before freeze, and starting it would consume the rehearsal time that
Sprint 1 already failed to spend.

### Capacity note

Sprint 1 ran at five delivery units over six days with 25% unplanned scope
arriving mid-sprint. Sprint 2 commits five *smaller* items and holds the sixth
as stretch — deliberately under last sprint's throughput, because the unplanned
work will arrive again and the last week before a freeze is the worst time to
discover there is no slack for it.

### Close-out

At sprint end, append Sprint 2 to `docs/sprint_data.json` with its real
completed count and a retrospective. At two sprints the health scorer's
predictability dimension starts working; at three, velocity forecasting and the
retrospective analyzer both come alive. Until then the scores above stay
partial, and should be quoted as partial.
