#!/usr/bin/env python3
"""Detect local hardware and write recommended Ollama runtime environment variables."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate recommended Ollama runtime settings based on available CPU/GPU resources."
    )
    parser.add_argument(
        "--output-env",
        type=Path,
        default=ROOT / "ollama_runtime.env",
        help="Where to write shell-exportable Ollama environment variables.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "ollama_runtime.json",
        help="Where to write machine-readable hardware and tuning metadata.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Optional manual cap for the auto worker probe upper bound.",
    )
    return parser.parse_args()


def detect_nvidia() -> dict[str, object]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {"gpu_vendor": "", "gpu_count": 0, "gpus": []}

    cmd = [
        nvidia_smi,
        "--query-gpu=index,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except Exception:
        return {"gpu_vendor": "nvidia", "gpu_count": 0, "gpus": []}

    gpus: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_free_mb": int(parts[3]),
                }
            )
        except ValueError:
            continue
    return {"gpu_vendor": "nvidia", "gpu_count": len(gpus), "gpus": gpus}


def detect_rocm() -> dict[str, object]:
    rocm_smi = shutil.which("rocm-smi")
    if not rocm_smi:
        return {"gpu_vendor": "", "gpu_count": 0, "gpus": []}
    try:
        result = subprocess.run([rocm_smi, "--showmeminfo", "vram"], capture_output=True, text=True, check=True)
    except Exception:
        return {"gpu_vendor": "amd", "gpu_count": 0, "gpus": []}
    gpu_lines = [line for line in result.stdout.splitlines() if "GPU" in line and "Total Memory" in line]
    return {"gpu_vendor": "amd", "gpu_count": len(gpu_lines), "gpus": []}


def detect_hardware() -> dict[str, object]:
    cpu_count = os.cpu_count() or 1
    nvidia = detect_nvidia()
    if int(nvidia.get("gpu_count", 0)):
        return {"cpu_count": cpu_count, **nvidia}
    rocm = detect_rocm()
    if int(rocm.get("gpu_count", 0)):
        return {"cpu_count": cpu_count, **rocm}
    return {"cpu_count": cpu_count, "gpu_vendor": "", "gpu_count": 0, "gpus": []}


def recommend_settings(hardware: dict[str, object], manual_cap: int) -> dict[str, int]:
    cpu_count = max(1, int(hardware.get("cpu_count", 1)))
    gpu_count = max(0, int(hardware.get("gpu_count", 0)))
    gpus = hardware.get("gpus", [])

    if gpu_count > 0 and isinstance(gpus, list) and gpus:
        free_mb_values = [int(gpu.get("memory_free_mb", 0)) for gpu in gpus if isinstance(gpu, dict)]
        min_free_mb = min(free_mb_values) if free_mb_values else 0
        base_parallel = max(1, min(16, gpu_count * 4, max(1, min_free_mb // 2048)))
    elif gpu_count > 0:
        base_parallel = max(1, min(8, gpu_count * 4))
    else:
        base_parallel = max(1, min(4, cpu_count // 2 if cpu_count > 1 else 1))

    if manual_cap > 0:
        base_parallel = min(base_parallel, manual_cap)

    if gpu_count > 0:
        max_loaded_models = max(1, min(gpu_count, 3))
    else:
        max_loaded_models = 1

    max_queue = max(128, min(1024, base_parallel * 64))

    return {
        "OLLAMA_NUM_PARALLEL": base_parallel,
        "OLLAMA_MAX_LOADED_MODELS": max_loaded_models,
        "OLLAMA_MAX_QUEUE": max_queue,
        "AUTO_WORKER_MAX": base_parallel,
    }


def write_env(path: Path, settings: dict[str, int]) -> None:
    lines = [f"export {key}={value}" for key, value in settings.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    hardware = detect_hardware()
    settings = recommend_settings(hardware, manual_cap=max(0, int(args.max_workers)))
    write_env(args.output_env, settings)
    write_json(
        args.output_json,
        {
            "hardware": hardware,
            "recommended_env": settings,
        },
    )
    print(f"Wrote Ollama runtime recommendations to {args.output_env}")
    print(f"Wrote hardware summary to {args.output_json}")


if __name__ == "__main__":
    main()
