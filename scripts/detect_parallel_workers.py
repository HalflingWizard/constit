#!/usr/bin/env python3
"""Detect the highest stable worker count for the parallel constitutional case."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_constitutional_ai_experiment.py"
PROBE_ROOT = ROOT / "worker_probe_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe worker counts for constitutional_ai_parallel and return the highest stable value."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Upper bound to probe. Defaults to min(8, cpu_count).",
    )
    parser.add_argument(
        "--rules-per-worker",
        type=int,
        default=3,
        help="Probe each candidate with workers * rules_per_worker constitution rules.",
    )
    parser.add_argument(
        "--constitution",
        type=Path,
        default=ROOT / "constitution.txt",
        help="Constitution file used to count available rules for the probe.",
    )
    parser.add_argument(
        "forwarded_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the main experiment runner.",
    )
    return parser.parse_args()


def normalize_forwarded_args(args: list[str]) -> list[str]:
    filtered: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--":
            continue
        if arg == "--parallel-rule-workers":
            skip_next = True
            continue
        if arg.startswith("--parallel-rule-workers="):
            continue
        if arg == "--cases":
            skip_next = True
            continue
        if arg == "--output-root":
            skip_next = True
            continue
        if arg.startswith("--output-root="):
            continue
        if arg == "--limit":
            skip_next = True
            continue
        if arg.startswith("--limit="):
            continue
        if arg == "--start":
            skip_next = True
            continue
        if arg.startswith("--start="):
            continue
        if arg == "--overwrite":
            continue
        filtered.append(arg)
    return filtered


def candidate_sequence(max_workers: int) -> list[int]:
    values: list[int] = []
    current = 1
    while current < max_workers:
        values.append(current)
        current *= 2
    values.append(max_workers)
    deduped: list[int] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def count_rules(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def run_probe(candidate: int, forwarded_args: list[str], subset_rule_count: int) -> bool:
    probe_dir = PROBE_ROOT / f"workers_{candidate}"
    shutil.rmtree(probe_dir, ignore_errors=True)
    cmd = [
        sys.executable,
        str(RUNNER),
        "--cases",
        "constitutional_ai_parallel",
        "--start",
        "0",
        "--limit",
        "1",
        "--overwrite",
        "--output-root",
        str(probe_dir),
        "--parallel-rule-workers",
        str(candidate),
        "--max-constitution-rules",
        str(subset_rule_count),
        *forwarded_args,
    ]
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode == 0


def main() -> None:
    args = parse_args()
    max_workers = max(1, int(args.max_workers))
    forwarded_args = normalize_forwarded_args(args.forwarded_args)
    total_rules = max(1, count_rules(args.constitution))
    rules_per_worker = max(1, int(args.rules_per_worker))
    successful = 0
    failed_at: int | None = None

    for candidate in candidate_sequence(max_workers):
        subset_rule_count = min(total_rules, candidate * rules_per_worker)
        if run_probe(candidate, forwarded_args, subset_rule_count):
            successful = candidate
            continue
        failed_at = candidate
        break

    if successful == 0:
        print("No stable worker count succeeded during probe.", file=sys.stderr)
        sys.exit(1)

    if failed_at is not None and failed_at - successful > 1:
        low = successful + 1
        high = failed_at - 1
        while low <= high:
            candidate = (low + high) // 2
            subset_rule_count = min(total_rules, candidate * rules_per_worker)
            if run_probe(candidate, forwarded_args, subset_rule_count):
                successful = candidate
                low = candidate + 1
            else:
                high = candidate - 1
            time.sleep(1.0)

    print(successful)


if __name__ == "__main__":
    main()
