#!/usr/bin/env python3
"""
WS8.1 — build the ordered execution plan for the matched-thread speed campaign.

Design constraints being satisfied:
  * every tool at 1/8/16/32 threads on an IDENTICAL input set;
  * >= 3 repeats of every cell (median + full range reported);
  * tool x thread order INTERLEAVED/randomised inside each tier so that any
    slow drift in machine state cannot be confounded with tool identity;
  * tiers ordered so the campaign degrades gracefully if it must be stopped:
    the cheap complete factorial lands first, the 25-hour 1-thread cells last.

Why two input sets:
  set_E_100 (100 genomes, 0.478 Gbp) carries the FULL 4-tool x 4-thread x 3-repeat
  factorial. CheckM2 on the full 1,000-genome Set E at 1 thread costs ~33 h per
  repeat (measured 32-thread CPU time 12,059 s for 100 genomes), i.e. ~100 h for
  3 repeats — impractical. Set E's full 1,000 genomes are retained as the
  historical comparator and are run for every tool where affordable.

Shuffle seed is CRC-32 of a fixed string (never abs(hash()), which is salted).

Writes results/revision/speed/run_plan.tsv
"""
from __future__ import annotations

import random
import zlib
from pathlib import Path

OUT = Path("/path/to/magicc/results/revision/speed")
THREADS = [1, 8, 16, 32]
REPEATS = [1, 2, 3]


def stable_hash(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def main() -> None:
    tiers: dict[int, list[tuple]] = {t: [] for t in range(0, 8)}

    # tier 0 — prerequisite: CheckM2 feature vectors for set_E_100 so DeepCheck
    #          has something to read. Not itself a reported measurement.
    tiers[0].append(("checkm2_vectors", 32, 1, "set_E_100", "warm"))

    # tier 1 — the cheap half of the matched factorial (minutes), all repeats
    for rep in REPEATS:
        for th in THREADS:
            tiers[1].append(("magicc", th, rep, "set_E_100", "warm"))
            tiers[1].append(("deepcheck", th, rep, "set_E_100", "warm"))
            tiers[1].append(("magicc", th, rep, "set_E_full", "warm"))
            tiers[1].append(("deepcheck", th, rep, "set_E_full", "warm"))

    # tier 2 — expensive tools at 8/16/32 threads, matched factorial (~7 h)
    for rep in REPEATS:
        for th in [8, 16, 32]:
            tiers[2].append(("checkm2", th, rep, "set_E_100", "warm"))
            tiers[2].append(("cocopye", th, rep, "set_E_100", "warm"))

    # tier 3 — full-set anchors at 32 threads, to validate the 100->1,000
    #          extrapolation and to reproduce the historical Table S4 conditions
    tiers[3].append(("checkm2", 32, 1, "set_E_full", "warm"))
    tiers[3].append(("cocopye", 32, 1, "set_E_full", "warm"))

    # tier 4 — cold vs warm page cache (WS8.2)
    for rep in REPEATS:
        tiers[4].append(("magicc", 32, rep, "set_E_full", "cold"))
        tiers[4].append(("magicc", 1, rep, "set_E_full", "cold"))
        tiers[4].append(("deepcheck", 1, rep, "set_E_100", "cold"))
    tiers[4].append(("checkm2", 32, 1, "set_E_100", "cold"))
    tiers[4].append(("cocopye", 32, 1, "set_E_100", "cold"))

    # tier 5 — WS8.3 side experiment: does the `conda run` wrapper account for the
    #          archived 74.4 s (direct) vs 97.5 s (via conda run) gap? Same binary,
    #          same input, same thread count, only the wrapper differs.
    # Repeat indices 4-6 so these are FRESH runs interleaved with the wrapper runs
    # in the same time window, rather than being silently satisfied by the tier-1
    # direct runs measured hours earlier.
    for rep in [4, 5, 6]:
        tiers[5].append(("magicc_condarun", 1, rep, "set_E_full", "warm"))
        tiers[5].append(("magicc", 1, rep, "set_E_full", "warm"))

    # tier 6 — THE LONG POLE: 1-thread cells of the two annotation-based tools
    #          (CheckM2 ~3.4 h/repeat, CoCoPyE ~1.9 h/repeat on 100 genomes)
    for rep in REPEATS:
        tiers[6].append(("checkm2", 1, rep, "set_E_100", "warm"))
        tiers[6].append(("cocopye", 1, rep, "set_E_100", "warm"))

    lines = ["tier\torder\ttool\tthreads\trepeat\tinput_set\tcache"]
    n = 0
    for tier in sorted(tiers):
        cells = tiers[tier]
        rng = random.Random(stable_hash(f"magicc-ws8-order-tier{tier}-v1"))
        # shuffle inside each repeat block so tool x thread order is interleaved
        # but repeats still complete in order (graceful degradation)
        by_rep: dict[int, list] = {}
        for c in cells:
            by_rep.setdefault(c[2], []).append(c)
        ordered = []
        for rep in sorted(by_rep):
            blk = by_rep[rep][:]
            rng.shuffle(blk)
            ordered.extend(blk)
        for c in ordered:
            n += 1
            lines.append(f"{tier}\t{n}\t" + "\t".join(str(x) for x in c))

    (OUT / "run_plan.tsv").write_text("\n".join(lines) + "\n")
    print(f"[plan] {n} runs -> {OUT/'run_plan.tsv'}")
    for tier in sorted(tiers):
        print(f"  tier {tier}: {len(tiers[tier])} runs")


if __name__ == "__main__":
    main()
