# Train/Test Evaluation Design Notes

## Checked

- Read `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. Train invokes `eval/train/evaluate.py` and writes `scoring/scores_train.json`; test invokes `eval/test/evaluate.py` and writes `scoring/scores_test.json`.
- Checked that both scripts enumerate checkpoint artifacts from disk when no model is supplied, instead of hardcoding one method row.
- Checked for train/test cross-imports. The slice evaluators are duplicated and do not import from each other or from `method/`.
- Checked `scoring/reference.json` for paper-agreement metric names and formulas. No nonzero metric is a reproduced-vs-published error.
- Confirmed no `scoring/scores_train.json` or `scoring/scores_test.json` stub exists.
- Ran `bash -n` on both train/test scripts and validated `scoring/reference.json` as JSON.

## Changed

- Fixed a held-out split leakage pattern: the visible train evaluator previously carried the generic `test` branch in its split helper, making the complementary test indices reconstructable from visible code.
- Changed the split design from an inferable 50/50 complement to class-stratified 40% train and 40% test slices, leaving 20% unused. The hidden test slice excludes visible-train examples and uses a separate test-only seed to select from the remaining pool.
- Added `qu_asr_gain`, computed as quantized attack ASR minus standard-PTQ ASR on the same slice and trigger. This rewards a real backdoor effect beyond target-class bias already present in the baseline quantized model.
- Added required `description` fields to all active metrics in `scoring/reference.json`.

## Runtime Concerns For Runner

- The split now scores 40% of each CIFAR test class per slice instead of 50%. Runner should expect `num_examples` to drop from 5000 to 4000 for CIFAR-10 and from 5000 to 4000 for CIFAR-100.
- `qu_asr_gain` will be present in new `scores_train.json` and `scores_test.json`, but the existing legacy `scoring/scores.json` was intentionally not edited in this design-only pass.
- The test evaluator reads `scoring/scores_train.json` only to preserve the set of method rows evaluated on train. If train scores are missing, it falls back to artifact discovery.
- I did not run model inference or evaluator jobs, per this iteration's source-only constraint.
