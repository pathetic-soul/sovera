# Roadmap — Sovereign Workbench

**SIH 2026 · PS 26117 (MRPL)** · last revised 2026-08-29

The charter's rule governs every decision below (AGENTS.md §14):

> Everything else is secondary. If a task doesn't serve the golden demo path,
> defer it. **Ship three legs deep, not six shallow.**

---

## Timeline

| Milestone | Date | State |
|---|---|---|
| Project start | ~2026-08-24 | earliest file timestamp |
| Legs 1–3 built (containment + routing) | 2026-08-26 | ✅ |
| Legs 4–5 built (agent loop + sandbox) | 2026-08-27 | ✅ |
| Grounding gap measured and closed by `tools/calc.py` | 2026-08-26 | ✅ |
| **Dress rehearsal, golden path timed** | **this week** | ⬜ **overdue — see R2** |
| **Scope freeze — end of week 3** | ~2026-09-13 | ⬜ confirm against the SIH calendar |
| Rehearsal-only window | freeze → finale | ⬜ |

`gates.py --strict` must be green (or every non-green item consciously accepted)
before the freeze date, not on the morning of the demo.

---

## Now — before scope freeze

Ordered by risk retired per hour spent. The first three are not features.

### P0 · Retire delivery risk

1. **Commit the repository.** Zero commits exist today. Then push to a private
   remote. Cost: minutes. Risk retired: total loss of the project (R1, score 20).
2. **Dress-rehearse the golden path end to end, on a stopwatch.** Arm the
   firewall → run `verify.ps1` → leg 4 → leg 5 → red button → verify chain →
   disarm. The §13 gate is 6 minutes and this has never been timed (R2).
3. **Rehearse the elevated launch sequence** on the machine that will actually
   be used, and settle the `pfirewall.log` ACL question (R4).
4. **25-minute sustained-load soak test.** Log GPU clocks. A thermal throttle
   mid-demo is silent and unrecoverable (R5).

### P1 · Close the one failing gate

5. **Router accuracy 80.3% → 90%.** The lever is corpus size, not tuning: 16
   exemplars per class is thin, and the failures are dominated by subject matter
   overriding intent (*"write a python script to compute corrosion rate"* routes
   to `calc` instead of `code_write`). Expand exemplars for the confused pairs
   and re-run `gates.py`. Never tune against the held-out set — that would buy a
   prettier number and destroy its meaning.

### P2 · Documentation truth

6. **Refresh AGENTS.md §16.** It still records router accuracy as 75.8% with the
   dense-encoder decision open; the encoder landed and the number is 80.3%.
   Stale charter text is the one thing a judge reading the repo will catch.
7. **Fill the remaining VRAM figures.** `writer` is measured at 4.6 GB;
   `coder` and `driver` still need `ollama ps` readings (§16, blocking §5).

---

## Next — only if P0–P2 are done and freeze has not landed

### Leg 6 · Multimodal

Handwritten inspection note or a drawing title block → structured answer with
the source region highlighted. The `vision` model (qwen3-vl:4b) is already
staged and the router already routes to it, so this is the cheapest remaining
leg by a wide margin.

**Gate before starting:** measure `vision`'s VRAM first. Leg 6 is the second
real VRAM spend and the 5.2 GB ceiling is the binding constraint.

### Leg 7 · Ingest + hybrid retrieval

10–15 realistic refinery documents (public API 570/510/653 inspection formats,
PSU tender PDFs, MOC templates) behind BM25 + vector with RRF fusion. The
charter's `ingest/` and `retrieval/` packages do not exist yet.

**This is the largest unbuilt leg.** Two new packages, an OCR/layout stack, and
a chunking strategy that keeps equipment tags (`10-P-101A`, `V-2301`,
`PSV-4402`) intact through retrieval. Realistically it does not fit before
freeze alongside P0–P2. **Recommendation: do not start it.** Five deep legs with
a rehearsed demo beats seven shallow ones with an untimed demo.

### Leg 8 · Cable pull

Trivial once 1–7 hold. Physically unplug the network at the start of the demo
and run everything with the cable on the table.

---

## Explicitly not doing

Recorded so the decisions are not silently relitigated under time pressure.

| Item | Why not |
|---|---|
| Further fine-tuning | Measured and rejected. The QLoRA adapter scored 0/8 with and without `calc`, against 7/8 for the base model plus `calc`. Fine-tuning on a fixed tool set also made the model brittle: offered a new tool it refused to use it. |
| Cloud fallback of any kind | Violates §2.1. It is the entire thesis. |
| Host-execution fallback when the Docker daemon is down | Deliberate. `py_sandbox` fails and names the fix rather than silently running code outside the container. |
| Fetching model weights at runtime | §2.1. Everything is pre-staged at build time. |
| A JS framework, a CDN, a webfont | The UI is one vendored HTML file with zero dependencies, and that is itself part of the sovereignty claim. |
| Chasing legs 6–8 at the cost of rehearsal | §14, explicitly. |
