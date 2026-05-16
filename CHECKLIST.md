# Reproduction Checklist

Check off items as you complete them. Order doesn't matter — work however makes sense for this paper.

## Briefing
- [x] `briefing/problem.md` (method-agnostic)
- [x] `briefing/evaluation.md` (method-agnostic)
- [x] `briefing/method.md`
- [x] `briefing/overview.md`

## Scoring
- [x] `scoring/reference.json` — paper's reported numbers
- [x] Workspace validated via `python validate.py`
- [x] `scoring/scores.json` — reproduced numbers (must match reference.json experiment/metric structure)
- [x] `scoring/EXPERIMENTS.md` — high-level purpose of each experiment in `scores.json` (do not edit until wrap up)
- [x] `scoring/TARGETS.md` — primary, constraint, and ablation targets (do not edit until wrap up)
- [x] `scoring/CONSTRAINTS.md` — what a scientist must hold fixed (do not edit until wrap up)
- [x] `scoring/DIRECTION.md` — research direction scope for scientist (do not edit until wrap up)

## Code
- [x] Evaluation code in `eval/`
- [x] Method implementation in `method/`
- [x] Baseline implementation in `baseline/` (if applicable)

## Scripts
- [x] `scripts/evaluate.sh`
- [x] `scripts/reproduce.sh`
- [x] `scripts/download.sh` (idempotent, size comment at top)
- [x] `scripts/baseline.sh` (if applicable)

## Environment
- [x] `environment/container.def` + `environment/setup.sh`

## Data
- [x] Dataset acquired (shared dir or `data/`)
- [x] Pretrained models downloaded if needed

## Milestones
- [x] `method_runs` — executes end-to-end without errors
- [x] `core_claim` — minimum experiment supports central claim
- [x] `core_claim_plus` — additional settings
- [x] `secondary_claims`
- [x] `majority`
- [ ] `near_complete`
- [ ] `full`
