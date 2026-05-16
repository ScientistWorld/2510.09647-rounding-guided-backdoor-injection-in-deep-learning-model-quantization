#!/usr/bin/env python3
"""Select a representative scores.json from a QURA hyperparameter sweep."""

import argparse
import json
import shutil
from pathlib import Path


def qura_metrics(path):
    with path.open() as f:
        scores = json.load(f)
    exp_name, exp = next(iter(scores["experiments"].items()))
    qura = exp["results"]["qura"]
    return exp_name, qura, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", default="/home/user/scoring/sweep")
    parser.add_argument("--output", default="/home/user/scoring/scores.json")
    parser.add_argument("--max_degradation", type=float, default=5.0)
    parser.add_argument(
        "--exclude_prefix",
        default="clean_",
        help="Ignore control runs with this prefix when selecting the proposed QURA artifact.",
    )
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--n_bits", type=int, default=4)
    parser.add_argument("--trigger_size", type=int, default=6)
    args = parser.parse_args()

    candidates = []
    for path in sorted(Path(args.sweep_dir).glob("*_scores.json")):
        if args.exclude_prefix and path.name.startswith(args.exclude_prefix):
            continue
        exp_name, qura, scores = qura_metrics(path)
        candidates.append((path, exp_name, qura, scores))
    if not candidates:
        raise FileNotFoundError(
            f"No non-control sweep score files found in {args.sweep_dir}; "
            f"exclude_prefix={args.exclude_prefix!r}"
        )

    feasible = [
        item for item in candidates
        if item[2].get("ca_degradation", float("inf")) <= args.max_degradation
    ]
    pool = feasible if feasible else candidates
    # Primary goal is ASR under the clean-accuracy constraint. If no run satisfies
    # the constraint, keep the least destructive run for honest diagnostics.
    if feasible:
        best = max(pool, key=lambda item: (item[2].get("qu_asr", -1), item[2].get("qu_at_ca", -1)))
    else:
        best = min(pool, key=lambda item: (item[2].get("ca_degradation", float("inf")), -item[2].get("qu_asr", -1)))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected_scores = best[3]
    if output.exists():
        with output.open() as f:
            merged = json.load(f)
        merged.setdefault("experiments", {})
        merged["experiments"].update(selected_scores.get("experiments", {}))
        selected_scores = merged
    with output.open("w") as f:
        json.dump(selected_scores, f, indent=2)
    if args.checkpoint_dir and args.model:
        stem = best[0].name.removesuffix("_scores.json")
        ckpt_dir = Path(args.checkpoint_dir)
        for suffix, dest in [
            (f"std{args.n_bits}.pt", f"{args.model}_std{args.n_bits}.pt"),
            (f"qura{args.n_bits}.pt", f"{args.model}_qura{args.n_bits}.pt"),
            (f"trigger{args.trigger_size}.pt", f"{args.model}_trigger{args.trigger_size}.pt"),
            ("results.json", f"{args.model}_results.json"),
        ]:
            src = Path(args.sweep_dir) / f"{stem}_{suffix}"
            if src.exists():
                shutil.copy2(src, ckpt_dir / dest)
    print(f"Selected {best[0].name}: qura={best[2]}")


if __name__ == "__main__":
    main()
