#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run every proof. Non-zero exit if any claim fails.

This is the gate. The library is not "done" when it runs; it is done when
every property it claims has been measured and holds. CI runs this file.

    python proofs/run_all.py            every proof
    python proofs/run_all.py peeking    only proofs whose name matches

Each proof is a separate process so that one crashing cannot take the
others with it, and so the exit code of each is its own verdict.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str]) -> int:
    pattern = argv[0] if argv else ""
    proofs = sorted(
        p for p in HERE.glob("proof_*.py") if pattern in p.stem
    )
    if not proofs:
        print(f"no proofs match {pattern!r}")
        return 1

    print(f"running {len(proofs)} proof(s)\n")
    started = time.perf_counter()
    results: list[tuple[str, int, float]] = []
    for path in proofs:
        t0 = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, str(path)], cwd=str(ROOT),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        elapsed = time.perf_counter() - t0
        print(completed.stdout, end="")
        if completed.stderr.strip():
            print(completed.stderr, end="")
        print()
        results.append((path.stem, completed.returncode, elapsed))

    total = time.perf_counter() - started
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for name, code, elapsed in results:
        print(f"  {'PASS' if code == 0 else 'FAIL'}  {name:34s} {elapsed:6.1f}s")
    failed = [name for name, code, _ in results if code != 0]
    print()
    if failed:
        print(f"{len(failed)} of {len(results)} proofs FAILED in {total:.1f}s")
        for name in failed:
            print(f"  - {name}")
        return 1
    print(f"all {len(results)} proofs hold ({total:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
