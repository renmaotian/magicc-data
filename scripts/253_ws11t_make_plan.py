#!/usr/bin/env python3
"""
WS11.T — ordered, resumable execution plan.

Tiers are ordered so the campaign degrades gracefully if it is interrupted:
the cheap proofs land first, the 4.2-hour serial competitor block last.

  tier 0  PILOT      one repeat of every tool at 32 threads on the 100-genome
                     cell.  Proves all four dispatch paths work in the new tree
                     before ~4 h of serial runtime is committed.  Also adds a
                     4th repeat to the 100-genome cell used by the 251x ratio.
  tier 1  T3         the V3-vs-V5 attribution ladder, 1 thread, 1,000 genomes.
                     Five arms x 3 repeats, each rung differing from the next by
                     exactly ONE thing:
                       magicc                 production console script, --input-list, V5
                       magicc_dir             production console script, --input DIR,  V5
                       magicc_v5code          launcher + current code tree,  --input DIR, V5
                       magicc_v3code_nostats  launcher + V3 code tree with the 19 assembly
                                              statistics short-circuited, --input DIR, V3
                       magicc_v3code          launcher + V3 code tree UNMODIFIED, --input DIR, V3
  tier 2  T2 extra   a 5th CheckM2 repeat on the 100-genome cell.
  tier 3  T1         the competitor repeats on the 1,000-genome cell, SERIAL.

Repeat indices continue the archived WS8 numbering for the same cell so that the
old and new runs pool without collision:
  checkm2/cocopye  set_E_full  t32  had r1        -> new r2, r3   (n = 3)
  magicc/deepcheck set_E_full  t32  had r1,r2,r3  -> new r4, r5   (n = 5)
  checkm2/cocopye  set_E_100   t32  had r1,r2,r3  -> new r4, r5

Writes results/revision/speed_v3/run_plan_v3.tsv
"""
from __future__ import annotations

from pathlib import Path

OUT = Path("/path/to/magicc/results/revision/speed_v3")

PLAN: list[tuple] = []


def add(tier: int, tool: str, threads: int, rep: int, inputset: str, note: str) -> None:
    PLAN.append((tier, tool, threads, rep, inputset, "warm", note))


# ---- tier 0: pilot ---------------------------------------------------------
add(0, "magicc",    32, 4, "set_E_100", "pilot: MAGICC dispatch path")
add(0, "deepcheck", 32, 4, "set_E_100", "pilot: DeepCheck dispatch path")
add(0, "cocopye",   32, 4, "set_E_100", "pilot: CoCoPyE dispatch path")
add(0, "checkm2",   32, 4, "set_E_100", "pilot: CheckM2 dispatch path + 251x cell repeat")

# ---- tier 1: T3 attribution ladder ----------------------------------------
LADDER = ["magicc", "magicc_dir", "magicc_v5code",
          "magicc_v3code_nostats", "magicc_v3code"]
for rep in (1, 2, 3):
    for tool in LADDER:
        iset = "set_E_full" if tool == "magicc" else "set_E_full_dir"
        add(1, tool, 1, 10 + rep, iset, f"T3 ladder rep {rep}")

# ---- tier 2: extra 100-genome CheckM2 repeat -------------------------------
add(2, "checkm2", 32, 5, "set_E_100", "T2: 5th repeat of the 251x numerator cell")

# ---- tier 3: T1 competitor repeats on the 1,000-genome cell, SERIAL --------
for rep_new, rep_cheap in ((2, 4), (3, 5)):
    add(3, "magicc",    32, rep_cheap, "set_E_full", "T1: MAGICC repeat (cheap)")
    add(3, "deepcheck", 32, rep_cheap, "set_E_full", "T1: DeepCheck repeat (cheap)")
    add(3, "checkm2",   32, rep_new,   "set_E_full", "T1: CheckM2 repeat (~4208 s)")
    add(3, "cocopye",   32, rep_new,   "set_E_full", "T1: CoCoPyE repeat (~3178 s)")

lines = ["tier\torder\ttool\tthreads\trepeat\tinput_set\tcache\tnote"]
for i, (tier, tool, th, rep, iset, cache, note) in enumerate(PLAN, 1):
    lines.append(f"{tier}\t{i}\t{tool}\t{th}\t{rep}\t{iset}\t{cache}\t{note}")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "run_plan_v3.tsv").write_text("\n".join(lines) + "\n")
print(f"[plan] {len(PLAN)} runs -> {OUT/'run_plan_v3.tsv'}")
for t in sorted({p[0] for p in PLAN}):
    print(f"  tier {t}: {sum(1 for p in PLAN if p[0] == t)} runs")
