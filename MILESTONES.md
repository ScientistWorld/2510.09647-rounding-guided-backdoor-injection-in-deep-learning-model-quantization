# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-04-09] - method_runs
- Implemented QURA Algorithm 2 (rounding-guided backdoor injection)
- Fixed critical algorithmic bugs:
  - R_bd formula: corrected to 0.5*(1-sign(I_bd)) per Algorithm 2 line 4
  - P(w) formula: uses signed I_bd/I_acc instead of absolute values per Eq. 6
  - I_acc: simplified to gradient-only per paper's simplification
- CIFAR-10 data fully available at /home/user/data/cifar-10
- Container build attempted: docker-archive approach uses pre-downloaded AzureLinux image from MCR at /home/user/environment/azurelinux_python.tar (137MB)
- QURA code, training pipeline, and evaluation framework complete