"""Run all KVCascade test scripts and report pass/fail per file.

Usage: /venv/kvcascade/bin/python tests/run_all.py
       /venv/kvcascade/bin/python tests/run_all.py -v   # show full output

Each test file is a standalone script that exits 0 on pass, 1 on fail.
"""

import argparse
import os
import subprocess
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))


def find_tests():
    """Return the list of test_*.py paths in this directory, sorted by name."""
    return sorted(
        os.path.join(_THIS, f)
        for f in os.listdir(_THIS)
        if f.startswith("test_") and f.endswith(".py")
    )


def run_one(path: str, verbose: bool) -> tuple[bool, float, str]:
    """Run a single test file as a subprocess. Returns (passed, elapsed_s, stdout)."""
    start = time.time()
    proc = subprocess.run(
        [sys.executable, path],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(_THIS),  # repo root, so any "src" lookups resolve
    )
    elapsed = time.time() - start
    output = proc.stdout + (proc.stderr if proc.returncode != 0 else "")
    passed = proc.returncode == 0
    if verbose or not passed:
        print(output, end="" if output.endswith("\n") else "\n")
    return passed, elapsed, output


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("-v", "--verbose", action="store_true",
                   help="print each test's full stdout (default: summary only)")
    args = p.parse_args()

    tests = find_tests()
    if not tests:
        print("no tests found")
        sys.exit(0)

    print(f"running {len(tests)} test files...\n")
    results = []
    for path in tests:
        name = os.path.basename(path)
        print(f"=== {name} ===")
        passed, elapsed, _ = run_one(path, args.verbose)
        results.append((name, passed, elapsed))
        status = "PASS" if passed else "FAIL"
        print(f"  → {status} ({elapsed:.1f}s)\n")

    total_elapsed = sum(r[2] for r in results)
    n_pass = sum(1 for _, p, _ in results if p)
    print("=" * 50)
    print(f"summary: {n_pass}/{len(results)} passed  ({total_elapsed:.1f}s total)")
    for name, passed, elapsed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status:4}  {name:40}  {elapsed:>5.1f}s")

    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
