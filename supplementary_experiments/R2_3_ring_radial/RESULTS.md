# Sec. 4.6 ring: radial marginal (manuscript Fig. 12, p. 31)

Date: 2026-09-30. Relates to Reviewer 2 comment 3 and Reviewer 3 minor comment 5.
Python: `D:\PythonProjects\WANPM_FP\venv\Scripts\python.exe`.

## 1. The "analytical" curve in Fig. 12 is not the solution of the stated problem

Sec. 4.6 states `V = (1/4)(r^2 - r0^2)^2` and `b_radial = -grad V = -2 (r^2 - r0^2) x`.
But `grad[(1/4)(r^2 - r0^2)^2] = (r^2 - r0^2) x`. The printed (and coded) drift is
the gradient of `(1/2)(r^2 - r0^2)^2`, twice the stated potential.

Write `b_radial = -kappa (r^2 - r0^2) x` and the generator's diffusion as `(a/2) Lap`.
The stationary density is `rho ~ exp(-kappa (r^2 - r0^2)^2 / (2a))`, i.e. `r^2 ~ N(r0^2, a/kappa)`.

| case | kappa | a | kappa/a | true radial std | radial peak |
|---|---|---|---|---|---|
| curve plotted in Fig. 11/12 (eq. `ring-rho-true`) | 1 | 1 | 1 | 0.258 | 1.6 |
| drift as coded, `(sigma^2/2) Lap` (all rebuttal reproductions) | 2 | 1 | 2 | 0.179 | 2.3 |
| drift as coded, `a_ij = sigma^2 delta_ij / 2` taken literally | 2 | 0.5 | 4 | 0.126 | 3.2 |

`verify_reference.py` confirms this twice without any neural network: an
Euler-Maruyama simulation of the SDE, and the weak-form residual on exact samples
(about 1e-5 for the matching density, 1e-2 to 1e-1 for the other two).

The learned histogram in the manuscript's Fig. 12 has peak 3.1, which matches the
third row. So the original run was essentially correct and the plotted reference
was too wide by a factor of two. The original training script is not in this
repository, so that attribution is inferred from the histogram, not read from code.
`figs/P3_manuscript_config/` reproduces the original figure with the correct curve.

Knock-on effects:

- The "learned std / true std = 0.78 to 0.84, structural over-concentration"
  conclusion of stage 1 (`R3_m5_ring_dim_r_sweep`) and stage 3 (`plot_ring.py`,
  `r_std_true = 0.2582` in `ring2d.json`) used the wrong reference. Against the
  correct density the same samples give 1.13 to 1.21: slightly too wide, because
  about 1% of the samples sit inside the ring. The core width (inter-quartile
  range) is within a few percent of the truth.
- Sec. 4.8 (time-dependent ring) uses the same coded drift with `a = sigma^2`.
  Its reported `r = 1.98 +- 0.19` agrees with the second row (0.179), not with the
  stated potential.
- Sec. 4.7 (Monte Carlo): the grid reference for `f_4` moves only from 0.1244 to
  0.1228 between the first and third rows, so that figure hardly changes.

## 2. What improves the fit once the reference is right

Problem used below ("P1"): `kappa = 1, a = 1`, i.e. drift `= -grad V` of the stated
potential. Its true density is exactly the formula already printed in the manuscript.
Manuscript configuration otherwise: `(4,64,2)` tanh, `K = 200`, `M = 1000`, 20,000
epochs, Adam `1e-2`, adversary SGD `1e-2`, loss unchanged. Mean +- sd over seeds.

| configuration | seeds | radial Hellinger | angular TV | 2-D Hellinger | mass at r<1 |
|---|---|---|---|---|---|
| exact samples (noise floor) | | 0.007 | 0.006 | 0.071 | 0.16% |
| manuscript configuration | 4 | 0.101 +- 0.014 | 0.062 | 0.215 +- 0.030 | 1.6% |
| + sigma-annealing (sigma 2 -> 1) | 2 | 0.150 | 0.116 | 0.277 | 2.0% |
| 120 sin + 80 Mexican-hat ridge wavelets | 4 | 0.087 +- 0.008 | 0.025 | 0.129 +- 0.007 | 1.6% |
| 120 sin + 80 Hermite-1 ridge wavelets | 4 | 0.085 +- 0.010 | 0.020 | 0.137 +- 0.012 | 1.4% |
| 120 sin + 40 Mexican-hat + 40 localized Gaussians (amplitude 10) | 4 | 0.071 +- 0.014 | 0.024 | 0.141 +- 0.003 | 0.9% |
| split learning rate (Muon 0.1 hidden, 1e-2 head), 200 sin | 4 | 0.053 +- 0.010 | 0.032 | 0.193 +- 0.026 | 0.6% |
| split learning rate + Hermite-1 wavelets | 4 | 0.056 +- 0.007 | 0.030 | 0.170 +- 0.005 | 0.5% |
| split learning rate, plain Adam (control) | 4 | 0.075 +- 0.013 | 0.060 | 0.242 +- 0.050 | 1.0% |

Original problem ("P3", `kappa = 2, a = 0.5`): manuscript configuration radial
Hellinger 0.131 +- 0.008 (3 seeds); split learning rate 0.094 +- 0.006.

Reading:

- **Wavelets** (ridge form `phi(w.x + b)`, scale 4 = width 0.25 = ring width, 40% of
  the bank) halve the angular error and bring the 2-D Hellinger distance from 0.21
  to 0.13. The learned density surface (Fig. 11) becomes smooth instead of spiky.
  They do not change the radial marginal: the residual radial error is stray mass
  inside the ring, and a ridge function cannot isolate the interior. Its best
  single-function signal-to-noise ratio for that leak is below 0.5 at `M = 1000`.
- **Localized Gaussian wavelets** `exp(-s^2 |x - c|^2 / 2)` can see the leak
  (signal-to-noise up to 10) but need an amplitude of about 10 to be heard over the
  plane waves. The gain is modest and seed-dependent. Amplitudes 20 and 40 are no better and hurt the angular marginal.
- **sigma-annealing does not help** and hurts the angular marginal. The ring is a
  connected support with no barrier to cross, so there is nothing for a curriculum to do.
- **Split learning rate** on the generator halves the radial Hellinger distance and
  the interior leak. It makes the 2-D density spikier (folds of the map). Muon beats
  the plain-Adam control here, but most of the gain is the split itself.
- Not useful: higher-frequency plane waves, `M = 4000`, lr decay, 60,000 epochs,
  `d = 16`. Morlet wavelets are harmful (loss about 5, ring destroyed).

## 3. Figure sets (drop-in replacements for the three Sec. 4.6 figures + Sec. 4.7)

| folder | run | note |
|---|---|---|
| `figs/final/` | `final_split_lr_wavelet` (`run_final.py`, one command, 212 s) | **chosen for the reply: split learning rate + 80 Hermite wavelets, seed 3** |
| `figs/P1_wavelet_localized/` | `P1_mex4_rbf2x10_s1` | best all-round single run; best of 4 seeds for the interior leak |
| `figs/P1_adam_wavelet/` | `P1_mex4_40_s1` | typical wavelet run: smooth surface, faint inner tail |
| `figs/P1_split_lr/` | `P1_base_muon_s1` | best radial marginal; spiky surface |
| `figs/P1_split_lr_wavelet/` | `P1_muon_herm4_40_s3` | as above with wavelets |
| `figs/P1_manuscript_config/` | `P1_base_s0` | unchanged method, consistent problem |
| `figs/P3_manuscript_config/` | `P3_base_s2` | original problem, corrected reference curve |
| `figs/P3_split_lr/` | `P3_muon_s1` | original problem, split learning rate |
| `figs/ring_radial_summary.png` | | before/after overview |

## 4. Two consistent ways to fix the manuscript

**Applied 2026-09-30: option A**, with the figures of `figs/P1_wavelet_localized/` (run `P1_mex4_rbf2x10_s1`). Sec. 4.8 was left unchanged.

**A. Keep the stated potential and density; correct the drift.** `b_radial = -(r^2 - r0^2) x`,
`a_ij = sigma^2 delta_ij`. Every analytical panel stays. Replace the learned panels
with a P1 figure set and update the numbers in the text. Sec. 4.8 then needs
`V = (1/2)(r^2 - r0^2)^2` or a rerun.

**B. Keep the runs; correct the formulas.** `V = (1/2)(r^2 - r0^2)^2` in Secs. 4.6 and
4.8, and `rho_true ~ exp(-2 (r^2 - r0^2)^2 / sigma^2)` in Sec. 4.6. Redraw the
analytical panels of Figs. 11 and 12 and the grid reference of Fig. 13
(`figs/P3_*` do this).

Either way the sentence at the end of Sec. 4.6 ("slightly higher peak density and
narrower radial width"), the matching bullet in the limitations list (main.tex l.215),
and the reply to Reviewer 2 comment 3 should be revised: the radial width was right.

## Files

- `ring_lab.py` problem, test-function bank (spec string `kind:count:scale[:amplitude]`), metrics, training.
- `verify_reference.py` the two network-free checks of Sec. 1.
- `bias_theory.py` weak-form residual and batch-bias term as the radial width is rescaled.
- `sweep.py` parallel sweeps (`--stage baseline|bank|anneal|knobs|combo|final|smooth|rbfamp|rbfamp2`, `--summary P1`).
- `make_paper_figs.py`, `make_mc_fig.py`, `plot_summary.py` figures.
- `results/<tag>.{json,npz,pt}` metrics, samples and generator weights of every run.
