# Official QuRA Repository Notes

The paper links to `https://github.com/cxx122/QuRA`. I verified the repository
on 2026-05-16 and cloned commit `5d39c4029e6ddc5f9a5132e2583efc42f84af994`
under `tmp/QuRA` for inspection only.

Implementation details that informed this reproduction:

- The official CV config `cv_4_4_bd.yaml` uses asymmetric per-channel 4-bit
  weights, asymmetric per-tensor 4-bit activations, calibration batch size 16,
  backdoor target 0, conflicting rate 0.03, binary rounding weight 0.01, beta
  range `[20, 2]`, and a minimal backdoor-loss threshold of 0.01.
- The released reconstruction code computes clean-loss influence using a full
  input Hessian term, not only a diagonal approximation. The local method now
  exposes `hessian_mode=full` as the default and keeps `diag` for ablations.
- Selected backdoor-aligned weights are initialized toward AdaRound values
  0.1/0.9 in the released code. They are not added to the total loss through
  the logged penalty term. The local method now uses a matching soft selected
  initialization by default; hard clamping remains available with
  `--freeze_selected`.
- The official scripts select GPUs by setting `CUDA_VISIBLE_DEVICES`, so they
  cannot be run directly in this benchmark environment. The local
  implementation keeps scheduler-managed GPU assignment intact.
