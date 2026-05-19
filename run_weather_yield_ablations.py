"""
Run ablation experiments for the weather-yield deep model.

Example:
    D:\\anaconda3\\envs\\cama\\python.exe run_weather_yield_ablations.py --epochs 20 --batch-size 128
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ABLATIONS = [
    {"name": "baseline_lstm_attention", "use_mstc": False, "use_mamba": False, "use_eca": False, "flags": ["--disable-mstc", "--disable-mamba-branch", "--disable-variable-attention"]},
    {"name": "mstc_only", "use_mstc": True, "use_mamba": False, "use_eca": False, "flags": ["--disable-mamba-branch", "--disable-variable-attention"]},
    {"name": "mamba_only", "use_mstc": False, "use_mamba": True, "use_eca": False, "flags": ["--disable-mstc", "--disable-variable-attention"]},
    {"name": "eca_only", "use_mstc": False, "use_mamba": False, "use_eca": True, "flags": ["--disable-mstc", "--disable-mamba-branch"]},
    {"name": "mstc_mamba", "use_mstc": True, "use_mamba": True, "use_eca": False, "flags": ["--disable-variable-attention"]},
    {"name": "mstc_eca", "use_mstc": True, "use_mamba": False, "use_eca": True, "flags": ["--disable-mamba-branch"]},
    {"name": "full_mstc_mamba_eca", "use_mstc": True, "use_mamba": True, "use_eca": True, "flags": []},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run weather-yield ablation experiments.")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--script", type=Path, default=Path("weather_yield_deep_experiment.py"))
    parser.add_argument("--out-root", type=Path, default=Path("outputs_weather_yield_ablations"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--yield-path", type=Path)
    parser.add_argument("--weather-path", type=Path)
    parser.add_argument("--soil-path", type=Path)
    return parser.parse_args()


def run_one(args: argparse.Namespace, item: dict[str, object]) -> dict[str, object]:
    out_dir = args.out_root / str(item["name"])
    cmd = [str(args.python), "-u", str(args.script), "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--window-size", str(args.window_size), "--hidden-dim", str(args.hidden_dim), "--out-dir", str(out_dir)]
    if args.max_rows:
        cmd.extend(["--max-rows", str(args.max_rows)])
    if args.yield_path:
        cmd.extend(["--yield-path", str(args.yield_path)])
    if args.weather_path:
        cmd.extend(["--weather-path", str(args.weather_path)])
    if args.soil_path:
        cmd.extend(["--soil-path", str(args.soil_path)])
    cmd.extend(item["flags"])

    print(f"\n=== Running {item['name']} ===", flush=True)
    subprocess.run(cmd, check=True)

    metrics = pd.read_csv(out_dir / "deep_test_metrics.csv").iloc[0].to_dict()
    notes = json.loads((out_dir / "deep_experiment_notes.json").read_text(encoding="utf-8"))
    branch = notes.get("mean_branch_weights", {})
    return {
        "experiment": item["name"],
        "use_mstc": item["use_mstc"],
        "use_mamba": item["use_mamba"],
        "use_eca": item["use_eca"],
        "mamba_impl": notes.get("mamba_impl"),
        "w_lstm": branch.get("w_lstm"),
        "w_mamba": branch.get("w_mamba"),
        **metrics,
    }


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = [run_one(args, item) for item in ABLATIONS]
    summary = pd.DataFrame(rows).sort_values("R2", ascending=False)
    summary.to_csv(args.out_root / "ablation_summary.csv", index=False)
    print("\n=== Ablation summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved summary to: {args.out_root / 'ablation_summary.csv'}")


if __name__ == "__main__":
    main()
