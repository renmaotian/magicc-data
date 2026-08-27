#!/usr/bin/env python3
"""
WS8.2 — evict the OS page cache for a named set of files, without root.

`echo 3 > /proc/sys/vm/drop_caches` needs root and passwordless sudo is NOT
available on this host, so a system-wide cache drop is impossible here (this
limitation is stated in the TCO table). What IS possible unprivileged is
posix_fadvise(POSIX_FADV_DONTNEED), which drops the *clean* page-cache pages
belonging to the named files. Evicting (a) every input FASTA and (b) the tool's
model/database files reproduces the dominant part of a cold start: every byte
the run reads from those files must come from disk again.

Residency is measured with mincore(2) before and after, so the report states
how many pages were actually evicted rather than assuming success.

  python 162_ws8_drop_file_cache.py --tool magicc --input-list <paths.txt>
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import mmap
import os
import sys
from pathlib import Path

POSIX_FADV_DONTNEED = 4

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
    ],
}

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]


def resident_pages(path: Path) -> tuple[int, int]:
    """(resident_pages, total_pages) for a file, via mincore(2)."""
    size = path.stat().st_size
    if size == 0:
        return 0, 0
    page = mmap.PAGESIZE
    npages = (size + page - 1) // page
    with open(path, "rb") as fh:
        try:
            mm = mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ)
        except (ValueError, OSError):
            return -1, npages
        try:
            vec = ctypes.create_string_buffer(npages)
            addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
            if _libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) != 0:
                return -1, npages
            res = sum(1 for b in vec.raw[:npages] if b & 1)
        finally:
            mm.close()
    return res, npages


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
    args = ap.parse_args()

    targets: list[Path] = []
    for line in Path(args.input_list).read_text().split():
        if line.strip():
            targets.append(Path(line.strip()).resolve())
    for asset in TOOL_ASSETS.get(args.tool, []):
        targets.extend(walk(Path(asset)))

    res_before = tot = res_after = 0
    n_err = 0
    for p in targets:
        try:
            r, t = resident_pages(p)
            if r >= 0:
                res_before += r
                tot += t
            evict(p)
            r2, _ = resident_pages(p)
            if r2 >= 0:
                res_after += r2
        except OSError:
            n_err += 1

    pg = mmap.PAGESIZE
    print(f"files_targeted={len(targets)} errors={n_err}")
    print(f"pages_total={tot} ({tot*pg/1e9:.3f} GB)")
    print(f"resident_before={res_before} ({res_before*pg/1e9:.3f} GB)")
    print(f"resident_after={res_after} ({res_after*pg/1e9:.3f} GB)")
    if tot:
        print(f"evicted_fraction_of_previously_resident="
              f"{1.0 - (res_after / res_before) if res_before else float('nan'):.4f}")
    print("NOTE: unprivileged posix_fadvise(DONTNEED) on the named files only; "
          "a system-wide /proc/sys/vm/drop_caches was NOT possible (no passwordless sudo).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
