# Verification Report

## Initial Assessment
The workspace claimed milestone `majority`. That claim is justified for the
packaged reduced-scale scope: `scores.json` covers the core 4-bit setting, the
8-bit extension, two ablations, and a comparison entry. The reproduced ASR is
well below the full-scale reference values, but it is above standard PTQ in the
core setting and the reduced-scale limitation is documented.

The main gym artifacts are present: method code, baseline PTQ, reusable
evaluation, train/test evaluation slices, container setup, download script,
reference scores, reproduced scores, and documentation.

## Issues Found
- Minor: `briefing/problem.md`, `scoring/EXPERIMENTS.md`,
  `scoring/TARGETS.md`, and `scoring/CONSTRAINTS.md` contained method-facing
  wording that was more specific than needed for from-scratch gym users.
- Cosmetic: an empty generated `torchinductor_tl0463/` cache directory was left
  in the workspace root.
- Cosmetic/system: unrelated dirty root entries remain from the broader
  harness (`README.md`, `manifest.json`, and deleted `trajectory/` logs). I
  left these untouched because they are outside the reproduction deliverables.

## Fixes Applied
- Reworded method-agnostic and method-safe documentation to describe the
  benchmark objective, constraints, and scored experiments without exposing
  internal implementation details.
- Removed the empty generated root cache directory.
- Verified `scripts/download.sh` is self-contained and `shared/`-free. The
  CIFAR-10 URL returns HTTP 200 with the expected archive size.
- Confirmed no committed code in `method/`, `eval/`, `baseline/`, `scripts/`,
  `data/`, or `environment/` references `shared/`.
- Re-ran `python3 validate.py --compare`; all scoring, structure, train/test,
  and import-separation checks pass.
- Checked shell syntax for the main scripts with `bash -n`.

## Final Judgment
The workspace is ready to serve as a research gym. Improve-mode agents can run
the existing method and evaluator, and from-scratch agents can use the problem,
data, evaluation, constraints, and direction documents without needing the
paper's implementation details. The evaluator reads method artifacts rather
than importing from `method/`, and train/test score artifacts have matching
method keys with explicit slice markers.

- Milestone: majority
- Ready for gym use: yes
- Confidence: medium
- Key limitation: reduced-scale reproduction with ASR below the full-scale
  reference values, though still nontrivial relative to standard PTQ.
