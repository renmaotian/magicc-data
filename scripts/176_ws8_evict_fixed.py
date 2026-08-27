#!/usr/bin/env python3
"""
WS8.2 (post-hoc correction, 2026-07-31) — working replacement for
scripts/162_ws8_drop_file_cache.py.

WHY THIS FILE EXISTS
--------------------
Script 162 was used for the `cache=cold` cells of the 92-cell campaign. It is
broken: `resident_pages()` builds a READ-ONLY mmap and then calls
`ctypes.c_char.from_buffer(mm)`, which raises

    TypeError: underlying buffer is not writable

on the FIRST file it inspects. Only `OSError` is caught in `main()`, so the
TypeError propagated and killed the process *before a single `evict()` call was
made*. Every one of the 11 `*.evict.txt` files in results/revision/speed/runs/
contains that identical traceback, and every `cache=cold` run recorded
`File system inputs = 0` in `/usr/bin/time -v`, i.e. it read nothing from the
block device. The campaign's "cold" cells were therefore warm-cache runs
carrying a cold label. 162 is kept unmodified as the record of what was run.

THE FIX
-------
mincore(2) needs a mapping address, so map the file through libc's mmap()
directly (PROT_READ / MAP_SHARED) instead of going through Python's mmap
object, whose buffer protocol refuses to hand out a writable pointer for a
read-only mapping. Eviction itself (posix_fadvise POSIX_FADV_DONTNEED) was
always fine; it simply never ran.

`echo 3 > /proc/sys/vm/drop_caches` still needs root and passwordless sudo is
still not configured on this host, so this remains a FILE-SCOPED eviction:
every byte the run reads from the named inputs and from the tool's model/DB
files must come from disk again, while kernel pages and the conda env's shared
objects stay cached. That limitation is stated in the TCO table.

    python 176_ws8_evict_fixed.py --tool magicc --input-list <paths.txt>
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import mmap
import os
import sys
from pathlib import Path

POSIX_FADV_DONTNEED = 4
PROT_READ = 1
MAP_SHARED = 1

TOOL_ASSETS = {
    "magicc": [
        "/path/to/magicc/models/magicc_v5.onnx",
        "/path/to/magicc/magicc/data",
    ],
    "checkm2": [
        "/path/to/magicc/tools/checkm2_db/CheckM2_database",
    ],
    "checkm2_vectors": [
        "/path/to/magicc/tools/checkm2_db/CheckM2_database",
    ],
    "cocopye": [
        "/path/to/.local/share/cocopye/cocopye_db",
        "/path/to/.local/share/cocopye/pfam_db",
        "/path/to/.local/share/cocopye/model",
    ],
    "deepcheck": [
        "/path/to/magicc/tools/DeepCheck/models/best_model.pt",
        "/path/to/magicc/tools/DeepCheck/scaler_params.npz",
        "/path/to/magicc/results/revision/speed/deepcheck_features",
    ],
}

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                       ctypes.c_int, ctypes.c_int, ctypes.c_long]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]

PAGE = mmap.PAGESIZE


def resident_pages(path: Path) -> tuple[int, int]:
    """(resident_pages, total_pages) for a file, via libc mmap + mincore(2)."""
    size = path.stat().st_size
    if size == 0:
        return 0, 0
    npages = (size + PAGE - 1) // PAGE
    fd = os.open(str(path), os.O_RDONLY)
    try:
        addr = _libc.mmap(None, size, PROT_READ, MAP_SHARED, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value, 2 ** 64 - 1):
            return -1, npages
        try:
            vec = ctypes.create_string_buffer(npages)
            if _libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) != 0:
                return -1, npages
            return sum(1 for b in vec.raw[:npages] if b & 1), npages
        finally:
            _libc.munmap(ctypes.c_void_p(addr), size)
    finally:
        os.close(fd)


def evict(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def walk(target: Path):
    if target.is_dir():
        for p in sorted(target.rglob("*")):
            if p.is_file():
                yield p
    elif target.is_file():
        yield target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True)
    ap.add_argument("--input-list", required=True)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    targets: list[Path] = []
    for line in Path(args.input_list).read_text().split():
        if line.strip():
            targets.append(Path(line.strip()).resolve())
    n_inputs = len(targets)
    for asset in TOOL_ASSETS.get(args.tool, []):
        targets.extend(walk(Path(asset)))

    res_before = tot = res_after = 0
    n_err = n_skip = 0
    for p in targets:
        try:
            r, t = resident_pages(p)
            if r >= 0:
                res_before += r
                tot += t
            else:
                n_skip += 1
            evict(p)
            r2, _ = resident_pages(p)
            if r2 >= 0:
                res_after += r2
        except OSError:
            n_err += 1

    frac = (1.0 - res_after / res_before) if res_before else float("nan")
    rep = {
        "tool": args.tool,
        "files_targeted": len(targets),
        "n_input_files": n_inputs,
        "n_asset_files": len(targets) - n_inputs,
        "errors": n_err,
        "unmappable": n_skip,
        "pages_total": tot,
        "bytes_total": tot * PAGE,
        "resident_before_pages": res_before,
        "resident_before_bytes": res_before * PAGE,
        "resident_after_pages": res_after,
        "resident_after_bytes": res_after * PAGE,
        "evicted_fraction_of_previously_resident": None if res_before == 0 else round(frac, 6),
        "method": "unprivileged posix_fadvise(POSIX_FADV_DONTNEED) on the named "
                  "files only; residency verified by mincore(2) before and after. "
                  "A system-wide /proc/sys/vm/drop_caches was NOT possible (no "
                  "passwordless sudo on this host).",
    }
    print(json.dumps(rep, indent=1))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rep, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
