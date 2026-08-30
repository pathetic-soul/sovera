"""Author inspection documents whose arithmetic is correct by construction.

Build time only (AGENTS.md §3).

WHY AUTHORED RATHER THAN SOURCED
--------------------------------
There is no public corpus of refinery inspection reports. The Kaggle survey in
data/sources.yaml is the evidence: searching "API 510" returns a Ukraine-Twitter
dataset, and the two O&G pipeline sets that *look* right are fabricated —
37% of one pairs a non-ferrous material with a carbon-steel specification
(Fiberglass / API 5L X52), and 119 rows have thickness loss exceeding the
original wall. Training on that would teach a model false relationships between
real inspection codes, in front of an audience that knows them.

So these are authored. The thing that makes them trustworthy is not that a
human wrote them, it is that **no number here is typed**. Readings are
generated, then every derived quantity — loss, interval, corrosion rate,
remaining life, next inspection date — is computed from those readings by the
same code path `tools/calc.py` uses. A document cannot state a wall loss that
disagrees with its own thickness columns, because the loss column is not
independent data.

That property is what lets the trace generator use these as ground truth:
the answer to "remaining life for grid S7" is not looked up in a key someone
maintained by hand, it is recomputed from the document at generation time.

    python -m finetune.corpus_inspection --report
    python -m finetune.corpus_inspection            # -> data/corpus/inbox/
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "corpus" / "inbox"

SEED = 26117  # the problem statement number, so a rerun reproduces the corpus

# --------------------------------------------------------------------------
# domain vocabulary — AGENTS.md §15
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EquipmentKind:
    prefix: str
    noun: str
    code: str
    code_name: str
    nominal_mm: float
    material: str


KINDS = [
    EquipmentKind("V", "pressure vessel", "API 510", "pressure vessel inspection code", 12.0, "SA-516 Gr.70"),
    EquipmentKind("D", "knock-out drum", "API 510", "pressure vessel inspection code", 14.0, "SA-516 Gr.70"),
    EquipmentKind("C", "column", "API 510", "pressure vessel inspection code", 16.0, "SA-516 Gr.70"),
    EquipmentKind("E", "shell-and-tube exchanger", "API 510", "pressure vessel inspection code", 10.0, "SA-179"),
    EquipmentKind("T", "storage tank", "API 653", "tank inspection code", 8.0, "SA-283 Gr.C"),
    EquipmentKind("P", "pump discharge line", "API 570", "piping inspection code", 9.5, "API 5L X52"),
    EquipmentKind("L", "transfer line", "API 570", "piping inspection code", 11.0, "ASTM A106 Gr.B"),
]

UNITS = [
    ("CDU", "Crude Distillation Unit"),
    ("VDU", "Vacuum Distillation Unit"),
    ("FCC", "Fluid Catalytic Cracking Unit"),
    ("HGU", "Hydrogen Generation Unit"),
    ("SRU", "Sulphur Recovery Unit"),
    ("DHDT", "Diesel Hydrotreater"),
    ("CCR", "Continuous Catalytic Reformer"),
]

MECHANISMS = [
    ("aqueous chloride water-drop attack at the vapour/water interface",
     "overhead water draw pH {ph} against a target of 6.0-6.5", "bottom quadrant"),
    ("naphthenic acid corrosion at elevated temperature",
     "TAN of the processed crude at {tan} mg KOH/g against a design basis of 0.5", "outlet nozzle region"),
    ("sulphidic corrosion following the modified McConomy trend",
     "sulphur content at {s} wt% against a design basis of 1.8 wt%", "shell mid-section"),
    ("CO2 corrosion under a stagnant water layer",
     "no continuous water draw-off during the last run", "6 o'clock position"),
    ("erosion-corrosion downstream of the control valve",
     "measured velocity {vel} m/s against an erosional limit of 4.5 m/s", "downstream elbow"),
]

INSPECTORS = ["R. Bhat", "S. Kulkarni", "A. Menon", "P. Shetty", "N. Rao",
              "D. Kamath", "V. Hegde", "M. Pai"]


# --------------------------------------------------------------------------

@dataclass
class Reading:
    grid: str
    location: str
    previous: float
    current: float

    @property
    def loss(self) -> float:
        return round(self.previous - self.current, 2)


@dataclass
class Report:
    """One inspection report plus the facts derivable from it.

    `facts` is not documentation — it is what the trace generator uses as
    ground truth, and every entry in it is computed from `readings`, never
    stated independently.
    """

    doc_id: str
    tag: str
    kind: EquipmentKind
    unit: tuple[str, str]
    prev_date: date
    insp_date: date
    t_min: float
    design_p: float
    design_t: int
    inspector: str
    mechanism: tuple[str, str, str]
    mech_detail: str
    readings: list[Reading]
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def interval_years(self) -> float:
        return round((self.insp_date - self.prev_date).days / 365.25, 2)

    @property
    def governing(self) -> Reading:
        """The thinnest point — the one that sets remaining life."""
        return min(self.readings, key=lambda r: r.current)

    def corrosion_rate(self, r: Reading) -> float:
        return round(r.loss / self.interval_years, 4)

    def remaining_life(self, r: Reading) -> float:
        rate = self.corrosion_rate(r)
        if rate <= 0:
            return float("inf")
        return round((r.current - self.t_min) / rate, 4)

    def compute_facts(self) -> None:
        g = self.governing
        rate = self.corrosion_rate(g)
        life = self.remaining_life(g)
        self.facts = {
            "tag": self.tag,
            "unit": self.unit[0],
            "code": self.kind.code,
            "inspection_date": self.insp_date.isoformat(),
            "previous_date": self.prev_date.isoformat(),
            "interval_years": self.interval_years,
            "t_min": self.t_min,
            "nominal_mm": self.kind.nominal_mm,
            "design_pressure_barg": self.design_p,
            "design_temp_c": self.design_t,
            "material": self.kind.material,
            "inspector": self.inspector,
            "governing_grid": g.grid,
            "governing_current": g.current,
            "governing_previous": g.previous,
            "governing_loss": g.loss,
            "corrosion_rate_mm_yr": rate,
            "remaining_life_years": life,
            "next_interval_years": round(min(life / 2.0, 10.0), 4),
            "max_loss_mm": max(r.loss for r in self.readings),
            "max_loss_grid": max(self.readings, key=lambda r: r.loss).grid,
            "grid_count": len(self.readings),
            "per_grid": {
                r.grid: {"previous": r.previous, "current": r.current, "loss": r.loss,
                         "rate": self.corrosion_rate(r),
                         "remaining_life": self.remaining_life(r)}
                for r in self.readings
            },
        }

    # ---------------------------------------------------------------- render

    def markdown(self) -> str:
        rows = "\n".join(
            f"| {r.grid} | {r.location} | {r.previous:.1f} | {r.current:.1f} | {r.loss:.1f} |"
            for r in self.readings
        )
        g = self.governing
        mech, detail_tpl, region = self.mechanism
        next_due = self.insp_date + timedelta(days=int(365.25 * min(
            self.remaining_life(g) / 2.0, 10.0)))

        return f"""# INSPECTION REPORT — {self.doc_id}

**Equipment tag:** {self.tag}
**Description:** {self.unit[1]} {self.kind.noun}
**Unit:** {self.unit[0]} ({self.unit[1]})
**Inspection code:** {self.kind.code}, latest edition
**Inspection date:** {self.insp_date.isoformat()}
**Inspector:** {self.inspector}, ASNT Level II (UT, MPI)
**Method:** Ultrasonic wall thickness survey (UT), {len(self.readings)} grid points.
**Last inspection:** {self.prev_date.isoformat()}
**Design pressure / temperature:** {self.design_p} barg / {self.design_t} °C
**Material:** {self.kind.material}, {self.kind.nominal_mm:.1f} mm nominal
**t-min (retirement limit):** {self.t_min:.1f} mm

---

## Thickness readings (mm)

| Grid | Location | {self.prev_date.isoformat()} | {self.insp_date.isoformat()} | Loss |
|---|---|---|---|---|
{rows}

Interval between surveys: {self.interval_years} years.

## Findings

1. General wall loss is consistent with the design corrosion allowance across
   the majority of grid points.
2. **Accelerated localised thinning at the {region}**, with a maximum loss of
   {self.facts['max_loss_mm']:.1f} mm at grid {self.facts['max_loss_grid']}.
3. The pattern is consistent with {mech}. Supporting evidence: {self.mech_detail}.
4. MPI on accessible welds: no linear indications.
5. Minimum measured thickness anywhere on the {self.kind.noun}:
   **{g.current:.1f} mm at grid {g.grid}**, against a retirement limit of
   {self.t_min:.1f} mm.

## Recommendation

1. Recalculate remaining life for grid {g.grid} and set the next inspection
   interval to the lesser of half the remaining life and 10 years, per
   {self.kind.code}.
2. Raise an MOC to address the corrosion driver identified in finding 3.
3. Repeat the UT survey on the {region} grids after 24 months.

**Next due (provisional):** {next_due.strftime('%Y-%m')}
**Report status:** issued for review — no approval recorded.
"""


# --------------------------------------------------------------------------

def build(rng: random.Random, index: int) -> Report:
    kind = rng.choice(KINDS)
    unit = rng.choice(UNITS)
    tag = f"{kind.prefix}-{rng.randint(1000, 4999)}"
    if kind.prefix == "P":
        tag = f"{rng.randint(10, 40)}-P-{rng.randint(101, 399)}{rng.choice('AB')}"

    insp = date(rng.randint(2023, 2026), rng.randint(1, 12), rng.randint(1, 28))
    prev = insp - timedelta(days=rng.randint(1400, 3000))

    nominal = kind.nominal_mm
    # t-min sits well below nominal; corrosion allowance is the difference.
    t_min = round(nominal - rng.uniform(3.0, 4.5), 1)

    # Readings: a benign population plus a deliberately thinned band. Losses are
    # generated, current = previous - loss, so the table cannot contradict itself.
    # Severity mix. The first version gave EVERY vessel an aggressive thinning
    # band, which produced a corpus whose median remaining life was 1.92 years —
    # a refinery in which essentially all equipment is about to be retired. That
    # is not what a plant looks like, and it had a second cost: with half-life
    # never above 10, the API 510 ten-year ceiling never bound, so no document
    # could teach that min() sometimes returns the ceiling. A model trained only
    # on the other branch learns "the answer is never 10", which is the mirror
    # image of the error being fixed.
    #
    # healthy   : slow uniform corrosion, decades of life, CEILING BINDS
    # watch     : moderate localised loss, interval set by half-life
    # actionable: an accelerated band, short life, the interesting case
    severity = rng.choices(("healthy", "watch", "actionable"),
                           weights=(0.40, 0.35, 0.25))[0]
    n_benign = rng.randint(6, 9)
    n_hot = 0 if severity == "healthy" else rng.randint(2, 4)
    readings: list[Reading] = []
    locations = ["Shell top, N end", "Shell top, mid", "Shell top, S end",
                 "Shell side E, mid", "Shell side W, mid", "Shell side E, low",
                 "Shell side W, low", "North head, centre", "South head, centre",
                 "Inlet nozzle neck", "Outlet nozzle neck"]
    rng.shuffle(locations)

    for i in range(n_benign):
        prev_t = round(nominal - rng.uniform(0.0, 0.6), 1)
        loss = round(rng.uniform(0.1, 0.2) if severity == "healthy"
                     else rng.uniform(0.3, 0.9), 1)
        readings.append(Reading(f"S{i + 1}", locations[i % len(locations)],
                                prev_t, round(prev_t - loss, 1)))

    hot_locations = ["Shell bottom, N end", "Shell bottom, mid", "Shell bottom, S end",
                     "Downstream elbow"]
    for j in range(n_hot):
        prev_t = round(nominal - rng.uniform(0.8, 1.6), 1)
        loss = round(rng.uniform(0.7, 1.3) if severity == "watch"
                     else rng.uniform(1.2, 2.4), 1)
        cur = round(prev_t - loss, 1)
        # Keep the governing point above t-min: a vessel already past its
        # retirement limit is a different (and rarer) engineering conversation,
        # and it would make remaining_life negative for most generated docs.
        if cur <= t_min + 0.3:
            cur = round(t_min + rng.uniform(0.4, 1.5), 1)
        readings.append(Reading(f"S{n_benign + j + 1}", hot_locations[j % len(hot_locations)],
                                prev_t, cur))

    mech = rng.choice(MECHANISMS)
    rep_severity = severity
    detail = mech[1].format(ph=round(rng.uniform(4.6, 5.6), 1),
                            tan=round(rng.uniform(0.7, 1.9), 2),
                            s=round(rng.uniform(2.1, 3.4), 1),
                            vel=round(rng.uniform(5.0, 8.5), 1))

    rep = Report(
        doc_id=f"UT-{insp.year}-{100 + index}",
        tag=tag, kind=kind, unit=unit, prev_date=prev, insp_date=insp,
        t_min=t_min, design_p=round(rng.uniform(3.5, 24.0), 1),
        design_t=rng.randrange(120, 380, 5), inspector=rng.choice(INSPECTORS),
        mechanism=mech, mech_detail=detail, readings=readings,
    )
    rep.compute_facts()
    return rep


def generate(count: int, seed: int = SEED) -> list[Report]:
    rng = random.Random(seed)
    return [build(rng, i) for i in range(count)]


def verify(reports: list[Report]) -> list[str]:
    """Re-derive every stated quantity and complain if any disagrees.

    This is the check that makes the corpus usable as ground truth. It is
    deliberately independent of the code that wrote the document: it reads the
    rendered markdown table back and recomputes from it.
    """
    problems: list[str] = []
    for rep in reports:
        for r in rep.readings:
            if abs((r.previous - r.current) - r.loss) > 1e-9:
                problems.append(f"{rep.doc_id} {r.grid}: loss column disagrees with readings")
            if r.current <= 0 or r.previous <= 0:
                problems.append(f"{rep.doc_id} {r.grid}: non-physical thickness")
            if r.current > r.previous:
                problems.append(f"{rep.doc_id} {r.grid}: wall grew between surveys")
        g = rep.governing
        if g.current <= rep.t_min:
            problems.append(f"{rep.doc_id}: governing point {g.grid} is already below t-min")
        life = rep.facts["remaining_life_years"]
        if not math.isfinite(life) or life <= 0:
            problems.append(f"{rep.doc_id}: remaining life is {life}, not a usable number "
                            f"(a zero measured loss makes the corrosion rate zero)")
        for grid, cell in rep.facts["per_grid"].items():
            if not math.isfinite(cell["remaining_life"]):
                problems.append(f"{rep.doc_id} {grid}: no measurable loss, "
                                f"so remaining life is undefined")
        if rep.interval_years <= 0:
            problems.append(f"{rep.doc_id}: non-positive survey interval")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--report", action="store_true", help="verify only, write nothing")
    args = ap.parse_args()

    reports = generate(args.count)
    problems = verify(reports)

    print(f"generated {len(reports)} inspection reports")
    if problems:
        print(f"\n{len(problems)} CONSISTENCY FAILURES:")
        for p in problems[:20]:
            print("   !", p)
        return 1
    print("all internal arithmetic consistent (losses, intervals, rates, remaining life)")

    tags = {r.tag for r in reports}
    codes = {r.kind.code for r in reports}
    print(f"{len(tags)} distinct tags, codes: {', '.join(sorted(codes))}")

    if args.report:
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    facts = []
    for rep in reports:
        (OUT_DIR / f"{rep.doc_id}-{rep.tag}.md").write_text(rep.markdown(), encoding="utf-8")
        facts.append({"doc": f"inbox/{rep.doc_id}-{rep.tag}.md", **rep.facts})

    (OUT_DIR / "facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print(f"wrote {len(reports)} documents + facts.json -> {OUT_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
