# WANPM for Fokker鈥揚lanck equations 鈥?supplementary experiments (revision R1)

Source code and results of the numerical experiments added, re-run or carried out for the
revision of

> A. Q. He and W. Cai, *Neural Pushforward Sampler for Transient Multi-modal and Singular
> Distributions from Fokker鈥揚lanck Equations with Adversarial Training*, submitted to
> J. Comput. Phys. (JCOMP-D-26-00768; earlier version arXiv:2509.14575).

Everything here was produced in response to the referee reports of the first round. Experiments
that appear unchanged from the original submission (triple-well, 10/100-dimensional
Ornstein鈥揢hlenbeck, 20-dimensional double well) are not included.

## Where to find what

### Examples in the revised manuscript

| Manuscript | Experiment | Folder | Figure(s) produced |
|---|---|---|---|
| 搂4.1 tilted (asymmetric) double well | mixed plane-wave/Gaussian-envelope test-function bank, 蟽-annealing | `supplementary_experiments/R1_5_asymmetric_muon/` (`run_mixed_bank_standalone.py`, `plot_mixed_bank_standalone.py`, `mixed_bank_result.md`) | `results/mixed_bank_result.png` (Fig. 3) |
| 搂4.2 Kramers double well (phase space) | sin-majority mixed bank, Muon/AdamW, M = 2000; Adam control | `supplementary_experiments/R1_6_nongrad_metastable/` (`run_experiment_kramers_closer.py`, `plot_a1_*.py`, `results_kramers_closer/a1_m2000_result.md`) | `results_kramers_closer/a1_m2000_result.png` (Fig. 4), `a1_adam_vs_muon.png` |
| 搂4.3 singular distribution on a diagonal line | generic network, uniform base, plane waves only | `supplementary_experiments/R2_7_diag45_singular_density/` | `results/diag45_singular_density_plot.png` (Fig. 5) |
| 搂4.4 ring with rotational drift, 搂4.5 Monte Carlo integration | corrected drift, mixed bank with Gaussian bumps; full sweep documented in `RESULTS.md` | `supplementary_experiments/R2_3_ring_radial/` (`ring_lab.py`, `sweep.py`, `make_paper_figs.py`, `make_mc_fig.py`) | `figs/P1_wavelet_localized/*` (Figs. 6鈥?; run `P1_mex4_rbf2x10_s1`) |
| 搂4.7 time-dependent 1-D Ornstein鈥揢hlenbeck | re-run with the map x鈧€ + 鈭歵 F虄(t, r) | `supplementary_experiments/R2_5_ou_1d_time/` | `results/1D_time_OU_*.png` (Fig. 11) |
| 搂4.8 ablation of the 鈭歵 time parameterization | p 鈭?{0.25, 0.5, 1, 2}, near-delta initial condition | `supplementary_experiments/R1_7_sqrt_t_ablation/` | `results/ablation_plot.png` (Fig. 12) |
| 搂4.10 time-dependent ring | corrected drift, space鈥搕ime mixed bank, Euler鈥揗aruyama reference | `supplementary_experiments/R2_3_ring_radial/` (`ring_time.py`, `make_ring_time_fig.py`) | `figs/ring_time_h64/2dring_time_evolution_scatter.png` (Fig. 17) |
| 搂4.11 one-peak-to-two-peak problem vs. temporal normalizing flow; free-energy dissipation | WANPM run `E_adam_saddle`, t-NF run `E_tnf_L4spl`, finite-volume reference, free energy | `supplementary_experiments/R1_2_time_bifurcation/transient/` (`wanpm_transient.py`, `tnf.py`, `reference_fv.py`, `metrics_vs_fv.py`, `free_energy.py`, `make_compare_figure.py`, `make_loss_figure.py`; `RESULTS_SUMMARY.md`) | `results/E_compare_wanpm_tnf.png` (Fig. 18), `E_loss_comparison.png` (Fig. 19), `E_free_energy.png` (Fig. 20) |

### Examples removed from the manuscript (reproduced in the reply letter)

The three introductory steady-state examples of the original submission were re-run for the
reviewers and then removed from the revised manuscript for reasons of length. Their full text,
as it stood before removal, is in **`supplement/removed_examples.pdf`** (source
`removed_examples.tex`).

| Removed example | Reply letter | Folder | Figure(s) |
|---|---|---|---|
| former 搂4.1, 1-D steady Ornstein鈥揢hlenbeck, now with adversarial test functions | Reviewer 2, comment 1 | `supplementary_experiments/R2_1_adversarial_sec41/` (`run_final.py` figure, `run_seeds.py` five-seed study) | `results/1dst_N21_adv_dist.png`, `1dst_N21_adv_LvE.png` |
| former 搂4.2, 1-D symmetric double well, 蟽-annealing + Muon/AdamW | Reviewer 1, comment 4; Reviewer 3, major comment 5 | `supplementary_experiments/R3_M5_optimizer_comparison/` (`muon_only.py --paper` makes the figure; `run_experiment.py` optimizer sweep; `muon_controls.py` AdamW-split control) | `results/muon_only.png` |
| former 搂4.3, 2-D symmetric double well, 蟽-annealing + mixed bank | Reviewer 1, comment 4; Reviewer 3, minor comment 4 | `supplementary_experiments/R3_m4_double_peak_2d/` (`dp2d_lab.py`, `make_figs.py`, `summarize.py`; run shown `adam_mixed_M1000_s1`, controls `adam_noanneal_*`, `adam_sin_*`) | `figs/double_peak_2d_plot{1,2}_final.png` |

### Experiments reported only in the reply letter

| Reply letter | Experiment | Folder | Figure |
|---|---|---|---|
| Reviewer 2, comment 2 | motion of the test-function frequencies on the ring example | `supplementary_experiments/R2_3_ring_radial/r2_2_frequency_rerun.py`, `results_r2_2_rerun/` (first version: `supplementary_experiments/R2_2_test_function_frequency/`) | `results_r2_2_rerun/frequency_analysis.png` |
| Reviewer 3, major comment 4 | multilayer-perceptron test functions vs. plane waves on the 1-D double well | `supplementary_experiments/R3_M4b_mlp_test_functions/run_experiment_sec42.py`, `results_sec42/` | `results_sec42/mlp_vs_planewave_sec42_seed0.png` |
| Reviewer 3, major comment 5 | SGD / Adam / AdamW / Shampoo / SOAP / Muon on the 1-D double well; AdamW-with-Muon's-split control | `supplementary_experiments/R3_M5_optimizer_comparison/` | `results/*.png` |
| Reviewer 3, minor comment 3 | free-energy dissipation for the 1-D time-dependent Ornstein鈥揢hlenbeck process | `supplementary_experiments/R3_m3_free_energy_dissipation/` | `results/ou_free_energy.png` |

## Layout and conventions

* All experiments live under `supplementary_experiments/`, one folder per experiment, named after
  the reviewer comment that prompted it (`R1_5_...` = Reviewer 1, comment 5; `R3_m4_...` =
  Reviewer 3, minor comment 4).
* `supplementary_experiments/common_wanpm.py` is the shared implementation of the method (steady and
  time-dependent weak-adversarial pushforward training) that the other scripts import by relative
  path. `supplementary_experiments/R1_5_asymmetric_muon/wavelet_testfuncs.py` holds the plane-wave test
  functions with wavelet envelopes. Keep the folder layout when running the scripts. (A few code
  comments speak of "stage 1/2/3"; this is the order in which the experiments were carried out,
  and the folders named there are the ones listed above.)
* Each experiment folder keeps its scripts, a `results*/` folder with the metrics (`*.json`),
  figures (`*.png`), logs and, where small, the trained generator (`*.pt`) and the samples
  (`*.npz`). Sample dumps larger than about 1 MB and the finite-volume reference arrays were left
  out; they are regenerated by the scripts (`reference_fv.py` for the reference solution).
* Figures in the manuscript are the result PNGs, in a few cases with a relabeled axis or a
  cropped title (e.g. `T` 鈫?`尾鈦宦筦 in the Kramers figure, `f鈧? f鈧卄 鈫?`f鈧? f鈧俙 in the Monte Carlo
  figure).
* The exploratory sweeps that preceded the reported configurations (other well separations,
  test-function shapes, batch sizes) are summarized in the `RESULTS*.md` / `*_result.md` notes of
  the folders; only the runs used in the manuscript or the letter, and the controls quoted there,
  are included.

## Environment

Python 3.13.9, PyTorch 2.9.1 (CUDA 12.6), NumPy 2.3.5, SciPy 1.16.3, Matplotlib 3.10.7,
`pytorch_optimizer` 3.10.1 (for Muon, Shampoo and SOAP). All runs were made on one laptop GPU
(RTX 3050 Ti); run times are stated in the manuscript and in the result notes.
