# Results summary — transient bifurcation vs. temporal normalizing flow (R1, comment 2)

JCOMP-D-26-00768, "Neural Pushforward Samplers for Transient Distributions from
Fokker–Planck Equations with Weak Adversarial Training". Addresses Reviewer 1's
starred comment 2 (quantitative benchmark against a normalizing-flow-based FPE
solver) for the time-dependent case, using Feng, Zeng & Zhou's temporal
normalizing flow (t-NF), *"Solving Time Dependent Fokker-Planck Equations via
Temporal Normalizing Flow"*, Commun. Comput. Phys. 32 (2022) 401-423.

**Headline figure:** [`results/E_compare_wanpm_tnf.png`](results/E_compare_wanpm_tnf.png)
— reference / WANPM / t-NF, heatmaps only, case E (σ=1), at the four snapshot
times shown in the manuscript, t ∈ {0.05, 0.2, 0.5, 1}; the version with all five
snapshot times (t = 0.1 included) is `results/E_compare_wanpm_tnf_5times.png`. The
metrics tables below list all five times.

**Loss-curve figure:** [`results/E_loss_comparison.png`](results/E_loss_comparison.png)
— WANPM's training loss (left) vs. t-NF's L_r/L_ic (right), same run.

## The problem

d=2, dX = −∇V dt + σ dW, V = [(x₁−1)²+(x₂−1)²]·[(x₁+1)²+(x₂+1)²] (wells at
(±1,±1), saddle at the origin, V=4 there). Initial condition: one Gaussian
N(0, 0.5²I) sitting on the saddle, so the density must split into two peaks
over [0, T=1]. Case E: σ=1 (σ²=1, relative barrier 2ΔV/σ²=8). Chosen as a
*mild* setting deliberately — case D (σ=0.5, barrier 32) and the manuscript's
own σ=0.4 (barrier 50) are both harder and are not part of this comparison;
see `results/D_*` for the WANPM-only tuning study that preceded this. Ground
truth: independent-seed Euler–Maruyama Monte Carlo, dt=1e-4, two halves of 1M
particles each (`reference/ref_E.npz`), giving a same-method noise floor for
every metric (reported in brackets throughout).

## WANPM configuration (`wanpm_transient.py`, tag `E_adam_saddle`)

No σ-specific tuning — the recipe was frozen on case D (σ=0.5) and reused
here unchanged:

- Pushforward F(t,x₀,r) = x₀ + √t·F̃(t,x₀,r), F̃ a (11 → 128 → 128 → 2) tanh
  MLP (18,306 parameters), r ~ N(0,I₈) (Gaussian base — a uniform U(0,1)⁸
  base was tried and was clearly worse: `results/D_adam_saddle_unif.*`).
- Test-function bank K=500: 225 sin plane waves + 150 Gaussian wavelets
  (initial |w| scale 4, time-frequency κ~N(0,10²)) + 125 low-κ Gaussian
  wavelets centred within ±0.3 of the saddle (κ~N(0,1.5²)) — the third
  group is what removes the spurious transition-state density spike a
  plain sin/wavelet bank leaves behind; see `results/D_adam_diagnosis.*`
  for the signal-to-noise diagnostic that motivated it.
- Adam (lr 1e-3) on the generator; SGD (lr 1e-2, 5 ascent steps every 10
  epochs) on the adversary; σ² annealed 1.5 → 1.0 (10% warm-up, 55% cosine,
  35% hold); M=2000, 30,000 epochs, t ~ U(0,T) exactly (no ε-offset).
- 874 s wall-clock (RTX 3050 Ti).

## t-NF configuration (`tnf.py`, tag `E_tnf_L4spl`)

Implemented from the paper (no reference code available); validated against
the paper's own toy heat-equation example first (`tnf_sanity_heat.py`,
`results/heat_sanity.json`): relative L² 2.6e-3 and relative KL 9.8e-5 at
t=1, k=5, vs. the paper's own ~2e-3 / ~5e-5 — the implementation reproduces
the paper's published accuracy on the paper's own problem.

- 4 (actnorm + time-conditioned affine coupling) blocks, width 128, +
  50-node spline layer (paper's eq. 3.14-3.17, c=4, γ=1e-6) — 68,734
  parameters, comparable to or larger than every depth/width the paper uses.
- Adam (lr 1e-3); loss = L_r + 10·L_ic (λ_ic unspecified in the paper — 10
  is our choice, not yet swept); strong-form residual via third-order
  autograd of log p; N_r=500,000 (5000 spatial × 100 time), N_ic=50,000,
  batch 10,000.
- Algorithm 1 exactly: 5 adaptive rounds, N_e=50 epochs in round 0, ×1.5
  growth per round (50, 75, 112, 169, 253 → 659 epochs, 32,950 optimizer
  steps total), collocation points resampled from the model's own current
  density after round 0.
- 7,341 s wall-clock (GPU shared with another job for part of this run;
  alone this would be roughly 80 min) — about 8.4× WANPM's cost for a
  comparable step count (32,950 vs. 30,000), reflecting the per-step cost
  of third-order autograd on a larger network vs. WANPM's first-order,
  closed-form test-function derivatives.
- An earlier run with a bug (resampled points silently filtered to
  \|x\|<24, hiding leaked mass) is archived, not reported:
  `legacy/results/E_tnf_outlier_filter.*`.

## Metrics

Sample-based only (WANPM has no closed-form density). sw2 = sliced
Wasserstein-2 (256 projections); Hellinger on histograms; π₊ = fraction with
u=(x₁+x₂)/√2 > 0 (well population, true 0.50 for case E's symmetric IC);
saddle mass = P(‖x‖<0.3). **t-NF's spline layer has tail slope γ=1e-6, so a
handful of samples land at \|x\|~10⁵ and make raw sw2 meaningless (\>1000);
all t-NF numbers below are trimmed to the ~72-86% of samples with \|x\|_∞<4**
(kept fraction reported per row) — this is generous to t-NF, not a
cherry-pick, since it discards exactly the samples that would otherwise
dominate the distance in t-NF's favor-negative direction.

| t | WANPM sw2 | t-NF sw2 (trimmed) | noise floor | WANPM π₊ | t-NF π₊ | t-NF kept frac |
|---|---|---|---|---|---|---|
| 0.05 | 0.038 | 0.777 | 0.003 | 0.49 | 0.74 | 0.86 |
| 0.1  | 0.054 | 0.872 | 0.003 | 0.48 | 0.78 | 0.80 |
| 0.2  | 0.068 | 0.982 | 0.003 | 0.48 | 0.84 | 0.74 |
| 0.5  | 0.166 | 1.356 | 0.010 | 0.47 | 0.91 | 0.72 |
| 1.0  | 0.280 | 1.544 | 0.031 | 0.45 | 0.94 | 0.72 |

(True π₊ = 0.500 at every t, by symmetry of the initial condition.)

## Reading

**WANPM** captures the bifurcation at every snapshot: both wells appear with
close to 50/50 population throughout, 5-20× closer to the reference than
t-NF on the trimmed metric at every t. It is not perfect — a faint bridge of
excess density remains along the diagonal between the wells at late times
(visible in `E_adam_saddle.png`'s bottom-row marginal), a residual not yet
eliminated by the saddle-wavelet fix.

**t-NF** does not find the second well. From t=0.05 onward the learned
density is already lopsided toward (1,1) (π₊=0.74, true 0.50) and the bias
grows monotonically to π₊=0.94 by t=1 — a progressive single-well collapse,
not a wrong-shaped pair of wells. This holds at a network 3.8× larger than
WANPM's and a step count (32,950) comparable to WANPM's (30,000).

**Why, mechanistically** (see `E_loss_comparison.png`): t-NF's residual loss
L_r converges *within* every adaptive round, often to values far below
WANPM's noise floor (down to 1.3e-6 in round 2) — optimization is not
stalling. But each round's adaptive resampling (Algorithm 1: draw new
collocation points from the model's own current density) reopens L_r by
4-8 orders of magnitude at the start of the next round (e.g. 260 at the
start of round 3). The model is achieving local consistency on whatever
points its own (already-collapsed) density currently produces, which
reinforces the wrong shape rather than correcting it. This is a
self-reinforcing failure of the adaptive-sampling loop, not a failure to
optimize in the ordinary sense — worth stating precisely rather than as
"the loss doesn't converge".

## Caveats — not yet done

1. **n=1 for both methods.** No seed variation yet at σ=1; the sw2 gap
   (5-20×) is large relative to run-to-run noise seen at case D (~0.03 in
   absolute sw2), but this has not been checked directly for case E.
2. **λ_ic unswept.** The paper does not state a value; 10 was not tuned.
3. **t-NF's outlier fraction, not just trimmed metrics, belongs in the final
   write-up** — 72-86% kept is itself informative and should not be buried
   in a caption.
4. **The saddle bridge in WANPM's own late-time result** is a known,
   partially-open residual (see `results/D_adam_*_diagnosis.json`) and
   should be disclosed alongside the headline comparison, not omitted.
5. Only one σ (the mild end, chosen so t-NF would struggle) has this full
   comparison; cases A/B/C (WANPM-only) and the steady-state ADDA-KR arm
   (`../steady/`, not yet started) are separate from this note.

## Update 2026-10-02 — finite-volume reference replaces the Monte Carlo one in the paper

The manuscript figure and metrics now use a deterministic reference:
`reference_fv.py` solves the Fokker–Planck equation on [-3,3]², 300×300 cells,
Scharfetter–Gummel fluxes, Heun, dt=1e-5 (`reference/ref_E_fv.npz`: density at
the snapshot times, F(t), and two independent sets of 1M samples drawn from the
density). `make_compare_figure.py` shows that density in the top row (default
t-NF tag is now `E_tnf_L4spl`, the run the paper uses). `metrics_vs_fv.py`
re-scores the saved samples (`results/E_metrics_fv.json`):

| t | WANPM sw2 | t-NF sw2 (trimmed) | noise floor |
|---|---|---|---|
| 0.05 | 0.038 | 0.777 | 0.007 |
| 0.1  | 0.050 | 0.873 | 0.003 |
| 0.2  | 0.068 | 0.984 | 0.005 |
| 0.5  | 0.156 | 1.357 | 0.019 |
| 1.0  | 0.296 | 1.543 | 0.007 |

Against the Monte Carlo reference the same saved samples give 0.039–0.293 for
WANPM, so the reference change moves nothing beyond the third decimal. The
Monte Carlo files (`reference_mc.py`, `reference/ref_E.npz`) are kept; the
tables above this section are the original Monte Carlo-based numbers.

`free_energy.py` (Reviewer 3, minor comment 3) uses the same solver for the
reference F(t); `results/E_free_energy.png` shows WANPM against it.
