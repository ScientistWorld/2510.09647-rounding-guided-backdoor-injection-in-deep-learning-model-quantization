#!/usr/bin/env python3
"""Workspace validator — checks scoring files and structural rules.

Usage:
    python validate.py                          # validate all
    python validate.py --reference-only         # validate reference.json only
    python validate.py --compare                # validate + compare scores against reference
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

WORKSPACE_DIR = Path(__file__).parent
SCORING_DIR = WORKSPACE_DIR / "scoring"


# ── Scoring validation ───────────────────────────────────────────────────────


def validate_reference(data: dict) -> list[str]:
    """Validate reference.json structure. Returns list of errors."""
    errors = []

    if "experiments" not in data:
        errors.append("Missing top-level 'experiments' key")
        return errors

    if not isinstance(data["experiments"], dict):
        errors.append("'experiments' must be a dict")
        return errors

    if len(data["experiments"]) == 0:
        errors.append("'experiments' is empty — add at least one experiment")

    # Validate weights sum to ~1.0
    weights = []
    for exp_name, exp in data["experiments"].items():
        if isinstance(exp, dict) and "weight" in exp:
            w = exp["weight"]
            if isinstance(w, (int, float)):
                weights.append(w)
    if weights:
        total = sum(weights)
        if abs(total - 1.0) > 0.01:
            errors.append(f"Experiment weights sum to {total:.3f}, expected 1.0")

    for exp_name, exp in data["experiments"].items():
        prefix = f"experiments.{exp_name}"

        if not isinstance(exp, dict):
            errors.append(f"{prefix}: must be a dict")
            continue

        # Required fields
        if "weight" not in exp:
            errors.append(f"{prefix}: missing 'weight'")
        elif not isinstance(exp["weight"], (int, float)):
            errors.append(f"{prefix}.weight: must be a number")
        elif not (0 < exp["weight"] <= 1):
            errors.append(f"{prefix}.weight: must be between 0 and 1")
        if "primary_metric" not in exp:
            errors.append(f"{prefix}: missing 'primary_metric'")
        if "metrics" not in exp:
            errors.append(f"{prefix}: missing 'metrics'")
        if "results" not in exp:
            errors.append(f"{prefix}: missing 'results'")

        # Validate metrics
        metrics = exp.get("metrics", {})
        if not isinstance(metrics, dict):
            errors.append(f"{prefix}.metrics: must be a dict")
            metrics = {}

        for metric_name, metric_def in metrics.items():
            if not isinstance(metric_def, dict):
                errors.append(f"{prefix}.metrics.{metric_name}: must be a dict")
                continue
            if "higher_is_better" not in metric_def:
                errors.append(f"{prefix}.metrics.{metric_name}: missing 'higher_is_better'")
            elif not isinstance(metric_def["higher_is_better"], bool):
                errors.append(f"{prefix}.metrics.{metric_name}.higher_is_better: must be a bool")
            if "coefficient" not in metric_def:
                errors.append(f"{prefix}.metrics.{metric_name}: missing 'coefficient'")
            elif not isinstance(metric_def["coefficient"], (int, float)):
                errors.append(f"{prefix}.metrics.{metric_name}.coefficient: must be a number")

        # primary_metric must be in metrics
        primary = exp.get("primary_metric")
        if primary and metrics and primary not in metrics:
            errors.append(f"{prefix}: primary_metric '{primary}' not found in metrics")

        # Validate results
        results = exp.get("results", {})
        if not isinstance(results, dict):
            errors.append(f"{prefix}.results: must be a dict")
            continue

        if len(results) == 0:
            errors.append(f"{prefix}.results: empty — add at least one method")

        has_proposed = False
        for method_name, method in results.items():
            mp = f"{prefix}.results.{method_name}"
            if not isinstance(method, dict):
                errors.append(f"{mp}: must be a dict")
                continue

            # type is required
            if "type" not in method:
                errors.append(f"{mp}: missing 'type' (must be 'baseline' or 'proposed')")
            elif method["type"] not in ("baseline", "proposed"):
                errors.append(f"{mp}: type must be 'baseline' or 'proposed', got '{method['type']}'")
            elif method["type"] == "proposed":
                has_proposed = True

            # Check that metric values are numbers and use known metric names
            for key, val in method.items():
                if key == "type":
                    continue
                if key not in metrics:
                    errors.append(f"{mp}: unknown metric '{key}' (not in metrics)")
                elif not isinstance(val, (int, float)):
                    errors.append(f"{mp}.{key}: must be a number, got {type(val).__name__}")

            # Check primary metric is present
            if primary and primary not in method and method.get("type") != "baseline":
                errors.append(f"{mp}: missing primary metric '{primary}'")

        if not has_proposed:
            errors.append(f"{prefix}.results: no method with type 'proposed' — at least one is expected")

    return errors


def validate_scores(data: dict, reference: dict | None = None) -> list[str]:
    """Validate scores.json structure. Returns list of errors."""
    errors = []

    if not isinstance(data, dict):
        errors.append("scores.json must be a dict")
        return errors

    # Accept both formats: {experiments: {...}} or flat {exp: {method: {...}}}
    if "experiments" in data and isinstance(data["experiments"], dict):
        experiments = data["experiments"]
    else:
        experiments = data

    if len(experiments) == 0:
        errors.append("scores.json is empty")
        return errors

    ref_exps = {}
    if reference and "experiments" in reference:
        ref_exps = reference["experiments"]

    for exp_name, exp_data in experiments.items():
        if exp_name.startswith("_"):
            continue

        if not isinstance(exp_data, dict):
            errors.append(f"{exp_name}: must be a dict")
            continue

        if exp_name not in ref_exps and ref_exps:
            errors.append(f"{exp_name}: not found in reference.json experiments")
            continue

        ref_metrics = ref_exps.get(exp_name, {}).get("metrics", {}) if ref_exps else {}
        ref_results = ref_exps.get(exp_name, {}).get("results", {}) if ref_exps else {}

        # Accept both: {results: {method: {metrics}}} or flat {method: {metrics}}
        if "results" in exp_data and isinstance(exp_data["results"], dict):
            methods = exp_data["results"]
        else:
            methods = exp_data

        for method_name, method_scores in methods.items():
            if method_name.startswith("_"):
                continue
            mp = f"{exp_name}.{method_name}"

            if not isinstance(method_scores, dict):
                errors.append(f"{mp}: must be a dict of metric_name -> value")
                continue

            if ref_results and method_name not in ref_results:
                errors.append(f"{mp}: method not found in reference.json")

            for metric_name, val in method_scores.items():
                if not isinstance(val, (int, float)):
                    errors.append(f"{mp}.{metric_name}: must be a number, got {type(val).__name__}")
                if ref_metrics and metric_name not in ref_metrics:
                    errors.append(f"{mp}.{metric_name}: not a known metric in reference.json")

    return errors


def _extract_scores_experiments(scores: dict) -> dict:
    """Normalize scores.json into {exp_name: {method_name: {metrics}}}."""
    if "experiments" in scores and isinstance(scores["experiments"], dict):
        raw = scores["experiments"]
    else:
        raw = scores
    out = {}
    for exp_name, exp_data in raw.items():
        if exp_name.startswith("_") or not isinstance(exp_data, dict):
            continue
        if "results" in exp_data and isinstance(exp_data["results"], dict):
            out[exp_name] = exp_data["results"]
        else:
            out[exp_name] = exp_data
    return out


def compare_scores(scores: dict, reference: dict) -> None:
    """Print side-by-side comparison of reproduced scores against reference."""
    exps = reference.get("experiments", {})
    score_exps = _extract_scores_experiments(scores)

    for exp_name, exp_methods in score_exps.items():
        exp_ref = exps.get(exp_name)
        if not exp_ref:
            continue

        weight = exp_ref.get("weight", "?")
        print(f"\n=== {exp_name} (weight: {weight}) ===")
        if exp_ref.get("description"):
            print(f"    {exp_ref['description']}")

        primary = exp_ref.get("primary_metric")
        metrics = exp_ref.get("metrics", {})
        ref_results = exp_ref.get("results", {})

        all_metrics = list(metrics.keys())

        print(f"\n    {'Method':<25} {'Source':<10}", end="")
        for m in all_metrics:
            marker = " *" if m == primary else ""
            print(f"{(m + marker):>15}", end="")
        print()
        print(f"    {'-'*25} {'-'*10}", end="")
        for _ in all_metrics:
            print(f"{'─'*15}", end="")
        print()

        all_methods = list(dict.fromkeys(list(ref_results.keys()) + list(exp_methods.keys())))
        for method_name in all_methods:
            ref_method = ref_results.get(method_name, {})
            score_method = exp_methods.get(method_name)
            tag = f" [{ref_method['type']}]" if ref_method.get("type") else ""

            print(f"    {(method_name + tag):<25} {'paper':<10}", end="")
            for m in all_metrics:
                val = ref_method.get(m)
                print(f"{val:>15.3f}" if val is not None else f"{'—':>15}", end="")
            print()

            if score_method:
                print(f"    {'':<25} {'repro':<10}", end="")
                for m in all_metrics:
                    val = score_method.get(m)
                    print(f"{val:>15.3f}" if val is not None else f"{'—':>15}", end="")
                print()
            print()


# ── Structural checks ────────────────────────────────────────────────────────


def check_import_separation() -> list[str]:
    """Check that eval/ and data/ do not import from method/."""
    errors = []
    pattern = re.compile(r"^\s*(?:from\s+method\b|import\s+method\b)", re.MULTILINE)
    for dirname in ("eval", "data"):
        dirpath = WORKSPACE_DIR / dirname
        if not dirpath.is_dir():
            continue
        for pyfile in dirpath.rglob("*.py"):
            try:
                content = pyfile.read_text()
            except Exception:
                continue
            if pattern.search(content):
                rel = pyfile.relative_to(WORKSPACE_DIR)
                errors.append(f"{rel}: imports from method/")
    return errors


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Validate workspace")
    parser.add_argument("--reference-only", action="store_true",
                        help="Only validate reference.json")
    parser.add_argument("--compare", action="store_true",
                        help="Compare scores against reference")
    args = parser.parse_args()

    ok = True

    # 1. Scoring: reference.json
    print("── Scoring ──")
    ref_path = SCORING_DIR / "reference.json"
    if not ref_path.exists():
        print("  ✗ scoring/reference.json not found")
        sys.exit(1)

    ref_data = json.loads(ref_path.read_text())
    ref_errors = validate_reference(ref_data)

    if ref_errors:
        print(f"  reference.json — {len(ref_errors)} error(s):")
        for e in ref_errors:
            print(f"    ✗ {e}")
        ok = False
    else:
        n_exp = len(ref_data.get("experiments", {}))
        n_methods = sum(len(e.get("results", {}))
                        for e in ref_data.get("experiments", {}).values())
        print(f"  reference.json — OK ({n_exp} experiments, {n_methods} methods)")

    # 2. Scoring: scores.json
    if not args.reference_only:
        scores_path = SCORING_DIR / "scores.json"
        if not scores_path.exists():
            print("  scores.json — not found (skipping)")
        else:
            scores_data = json.loads(scores_path.read_text())
            scores_errors = validate_scores(scores_data, ref_data)

            if scores_errors:
                print(f"  scores.json — {len(scores_errors)} error(s):")
                for e in scores_errors:
                    print(f"    ✗ {e}")
                ok = False
            else:
                n_exp = len(scores_data)
                n_methods = sum(len(m) for m in scores_data.values()
                                if isinstance(m, dict))
                print(f"  scores.json — OK ({n_exp} experiments, {n_methods} methods)")

            if args.compare and not scores_errors:
                compare_scores(scores_data, ref_data)

    # 3. Import separation
    print("\n── Structure ──")
    sep_errors = check_import_separation()
    if sep_errors:
        print(f"  import separation — {len(sep_errors)} error(s):")
        for e in sep_errors:
            print(f"    ✗ {e}")
        ok = False
    else:
        print("  import separation — OK (eval/ and data/ do not import from method/)")

    # Summary
    print()
    if ok:
        print("All checks passed.")
    else:
        print("FAILED — fix the errors above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
