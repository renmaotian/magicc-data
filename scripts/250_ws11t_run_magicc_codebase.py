#!/usr/bin/env python3
"""
WS11.T (T3) — launch the MAGICC CLI from an EXPLICITLY CHOSEN code tree.

Why this exists
---------------
`magicc` is installed editable in the `magicc2` env through a setuptools
MetaPathFinder (`__editable___magicc_0_3_0_finder`).  A MetaPathFinder is
consulted BEFORE any sys.path entry, so neither `PYTHONPATH=` nor running with
a different cwd can make the interpreter import a *different* copy of the
package.  This launcher removes that finder, then imports `magicc` from
--code-root.

It is used for BOTH arms of the V3-vs-V5 attribution experiment so that the
two arms share byte-identical harness overhead; a separate control run of the
production console script quantifies the launcher's own cost.

Usage:
  python 190_ws11t_run_magicc_codebase.py --code-root <dir> -- <magicc argv...>
"""
from __future__ import annotations

import sys


def main() -> int:
    argv = sys.argv[1:]
    if "--code-root" not in argv:
        print("--code-root is required", file=sys.stderr)
        return 2
    i = argv.index("--code-root")
    code_root = argv[i + 1]
    rest = argv[i + 2:]
    if rest and rest[0] == "--":
        rest = rest[1:]

    # 1. drop the editable-install finder so it cannot shadow --code-root
    sys.meta_path = [
        f for f in sys.meta_path
        if "__editable__" not in getattr(f, "__module__", "")
        and "__editable__" not in type(f).__module__
    ]
    # 2. make --code-root the first place `import magicc` looks
    sys.path.insert(0, code_root)

    from magicc.cli import main as cli_main  # noqa: E402
    sys.argv = ["magicc"] + rest
    mod = sys.modules["magicc.cli"]
    assert mod.__file__.startswith(code_root), (
        f"wrong magicc imported: {mod.__file__} (wanted under {code_root})")
    print(f"[launcher] magicc.cli = {mod.__file__}", file=sys.stderr)
    return cli_main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
