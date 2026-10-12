"""
R1.5 revisited with Muon: can the optimizer fix the asymmetric double well?

BACKGROUND.  Stage 2's R1_5 (`R1_5_asymmetric_double_well/`)
is the weakest result in the rebuttal set.  On the tilted potential

    V(x) = (x^2 - 1)^2 + eps*x,   b(x) = -V'(x) = -4x^3 + 4x - eps,
    eps = 0.1,  a = sigma^2 = 0.3

the true Boltzmann populations (quadrature) are 78.4% left / 21.6% right, and
the trained sampler returned 39.8% / 60.2% -- it put the MAJORITY of its mass
in the SHALLOWER well, i.e. it got the direction of the asymmetry wrong, after
11,134 s of training on a large configuration (d_base=8, hidden=(64,64),
K=3000, M=8000, 75k epochs, Adam lr 2e-2).

Stage 3 (`R3_M5_optimizer_comparison/`) then found that on the
manuscript's own Sec. 4.2 (SYMMETRIC) double well, swapping the generator
optimizer from Adam to Muon cut the Hellinger distance from 0.210 to 0.091.
This script asks whether that carries over to the asymmetric case, which is a
harder and qualitatively different failure: symmetric-case accuracy is about
the SHAPE of each peak (the 50/50 split is free by symmetry), whereas here the
quantity that fails is the RELATIVE MASS of two wells separated by a barrier.

DESIGN.  Adam and Muon are compared at an IDENTICAL configuration, so the
architecture change relative to stage 2 cannot be credited to Muon:

    pushforward   (8, 100, 1) tanh MLP, 8-dim Gaussian base   [Sec. 4.2 arch]
    test funcs    f^(k)(x) = sin(w^(k) x + b^(k)), K = 100
    16,000 epochs, batch M = 1000
    adversary     SGD lr 1e-2, 5 ascent steps every 10 epochs [paper's own]
    curriculum    sigma^2 annealed 1.5 -> 0.3 (10% warmup, 55% cosine anneal)

Both optimizers get their own LR sweep (Muon's optimum sits ~100x above
Adam's, so a shared LR would be meaningless), then the best non-collapsed
settings are re-run across seeds; only seed-averaged numbers are quoted.

CURRICULUM CONTROL.  The sigma-annealing curriculum is a suspect in its own
right here: at sigma^2 = 1.5 the eps=0.1 tilt is nearly invisible (the true
split at the START of the curriculum is only ~56:44, versus 78:22 at the
target), so the curriculum spends most of training near a near-symmetric
solution and the asymmetry has to be established late, by transporting mass
across a barrier.  `--mode anneal` re-runs the best Adam and Muon settings
with no curriculum, and with a curriculum starting from a milder sigma^2, to
separate "the optimizer cannot move the mass" from "the curriculum sends it
to the wrong place first".

METRIC.  As in stage 3, the minimax training loss is not comparable across
optimizers, so every run is scored externally against the analytical tilted
Boltzmann density rho ~ exp(-2V/a) computed by quadrature: the right-well
fraction error (the quantity R1.5 actually reports), plus Hellinger and total
variation over the density, and the transition-state (bridge) over-population.

Usage:
    python asym_muon.py --mode smoke     # 300-epoch sanity check
    python asym_muon.py --mode sweep     # per-optimizer LR grid, seed 0
    python asym_muon.py --mode seeds     # best settings x 3 seeds
    python asym_muon.py --mode anneal    # curriculum control
    python asym_muon.py --mode plot      # figures + summary from JSON
"""
import sys, os, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))

import numpy as np
import torch
from scipy.integrate import quad

import pytorch_optimizer as po
from common_wanpm import (SteadyPushforward, SteadyTestFunctions, steady_loss,
                          gaussian_base_sampler, mlp)
import torch.nn as nn
from wavelet_testfuncs import WaveletTestFunctions, MixedTestFunctions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTDIR = os.path.join(HERE, 'results')
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Problem -- identical to R1_5_asymmetric_double_well
# ---------------------------------------------------------------------------
EPS = 0.1
SIGMA2 = 0.3          # this is `a`, the diffusion coefficient (see common_wanpm)


def V(x):
    return (x ** 2 - 1) ** 2 + EPS * x


def drift_fn(x):
    return -4 * x ** 3 + 4 * x - EPS


def drift_fn_eps(x, eps):
    return -4 * x ** 3 + 4 * x - eps


# ---------------------------------------------------------------------------
# TILT-PRESERVING CURRICULUM.
#
# The standard sigma-annealing curriculum lowers the barrier by raising the
# temperature -- but the well-population ratio it is trying to teach is
#     P_right / P_left ~ exp(-2 (V(x_R) - V(x_L)) / a) = exp(-4 eps / a),
# so raising `a` washes the asymmetry out at exactly the same rate it flattens
# the barrier. At the curriculum's start (a = 1.5) the true split is 56:44,
# not 78:22: for the first two-thirds of training the model is being taught a
# nearly symmetric problem, and the asymmetry then has to be established late,
# by moving mass across a barrier that is by then already high.
#
# The fix is to make the tilt track the temperature, eps(a) = eps_target *
# a / a_target, which holds exp(-4 eps / a) -- and hence the target population
# split -- CONSTANT at 78:22 for the whole curriculum while the barrier still
# anneals down. eps(1.5) = 0.5, comfortably below the eps = 8/(3 sqrt 3) ~ 1.54
# at which the shallower well disappears entirely, so the problem stays
# bimodal throughout.
# ---------------------------------------------------------------------------
def shifted_base_sampler(shift):
    """N(shift*1, I) instead of N(0, I).

    WHY.  The generator is a tanh MLP, and tanh is ODD: with zero biases
    F(-r) = -F(r) exactly. The base N(0,I) is spherically symmetric, so r and
    -r are equally likely, and an odd F therefore forces P(x>0) = 0.5 EXACTLY.
    At standard initialisation the biases are small, so the map starts NEARLY
    odd -- which is the most plausible explanation for Muon sitting at
    P(x>0) = 0.500 +/- 0.003 with skew ~ 0.000 at every barrier height.

    NOTE what this does and does not do. For a one-hidden-layer net,
        F(z + mu) = W2 tanh(W1 z + (W1 mu + b1)) + b2,
    so shifting the base is EXACTLY a shift of the first-layer bias, which is
    already learnable. The representable family is therefore unchanged: this is
    purely a symmetry-breaking change to the INITIALISATION, not extra
    capacity. GELU is the stronger intervention, since it is not odd at all.
    """
    def _s(M, d, device):
        return torch.randn(M, d, device=device) + shift
    return _s


def eps_at_sigma2(a, tilt_preserving):
    return EPS * a / SIGMA2 if tilt_preserving else EPS


def unnorm_density(x, a=SIGMA2):
    return np.exp(-2 * V(x) / a)


def true_stats(a=SIGMA2):
    Z, _ = quad(unnorm_density, -4, 4, args=(a,), limit=400)
    fr, _ = quad(lambda t: unnorm_density(t, a) / Z, 0, 4, limit=400)
    return Z, float(fr)


Z_TRUE, TRUE_FRAC_RIGHT = true_stats()
TRUE_FRAC_LEFT = 1.0 - TRUE_FRAC_RIGHT


def true_pdf(x):
    return unnorm_density(x) / Z_TRUE


# ---------------------------------------------------------------------------
# Training configuration -- manuscript Sec. 4.2, verbatim except the potential
# ---------------------------------------------------------------------------
D_BASE = 8
HIDDEN = (100,)
K_TEST = 100
N_EPOCHS = 16000
BATCH_M = 1000
ADV_EVERY = 10
ADV_STEPS = 5
LR_TEST_PAPER = 1e-2
LR_GEN_PAPER = 2e-3          # Adam on theta; also Muon's internal AdamW lr

SIGMA2_START = 1.5           # stage-1/2/3 curriculum starting point
WARMUP_FRAC, ANNEAL_FRAC = 0.10, 0.55


def sigma2_at_epoch(epoch, n_epochs, anneal=True, s2_start=SIGMA2_START,
                    target=SIGMA2):
    if not anneal:
        return target
    w_end = int(WARMUP_FRAC * n_epochs)
    a_end = w_end + int(ANNEAL_FRAC * n_epochs)
    if epoch < w_end:
        return s2_start
    if epoch < a_end:
        p = (epoch - w_end) / max(1, a_end - w_end)
        cos_p = 0.5 * (1 - np.cos(np.pi * p))
        return s2_start + (target - s2_start) * cos_p
    return target


# ---------------------------------------------------------------------------
# Optimizers.  The Muon parameter split mirrors
# .../run_experiment.py::_split_muon_groups exactly: Muon on
# 2-D hidden weight matrices, its internal AdamW on all biases and on the
# output layer.  On this (8,100,1) net that is one matrix, net.0.weight (100x8).
# ---------------------------------------------------------------------------
def split_hidden_head(model):
    named = list(model.named_parameters())
    linear_idx = sorted({int(n.split('.')[1]) for n, _ in named})
    head_idx = linear_idx[-1]
    hidden, rest = [], []
    for n, p in named:
        idx = int(n.split('.')[1])
        (hidden if (p.ndim >= 2 and idx != head_idx) else rest).append(p)
    return hidden, rest


def make_gen_opt(name, model, lr, head_lr=LR_GEN_PAPER):
    if name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == 'Muon':
        hidden, rest = split_hidden_head(model)
        return po.Muon([dict(params=hidden, lr=lr, weight_decay=0.0, use_muon=True),
                        dict(params=rest, lr=head_lr, weight_decay=0.0, use_muon=False)],
                       weight_decay=0.0)
    if name == 'AdamW-split':
        # Stage-3 CONTROL A, carried over: same two-group lr split as Muon,
        # same betas as Muon's internal AdamW, orthogonalization removed.
        hidden, rest = split_hidden_head(model)
        return torch.optim.AdamW([dict(params=hidden, lr=lr),
                                  dict(params=rest, lr=head_lr)],
                                 betas=(0.9, 0.95), weight_decay=0.0)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Evaluation against the analytical tilted Boltzmann density
# ---------------------------------------------------------------------------
GRID_LO, GRID_HI, N_BINS, N_EVAL = -2.5, 2.5, 250, 200000


def true_moments(a=SIGMA2):
    """(mean, skew) of the true tilted Boltzmann density by quadrature."""
    Z, _ = quad(unnorm_density, -5, 5, args=(a,), limit=400)
    m1, _ = quad(lambda t: t * unnorm_density(t, a) / Z, -5, 5, limit=400)
    m2, _ = quad(lambda t: (t - m1) ** 2 * unnorm_density(t, a) / Z, -5, 5, limit=400)
    m3, _ = quad(lambda t: (t - m1) ** 3 * unnorm_density(t, a) / Z, -5, 5, limit=400)
    return float(m1), float(m3 / m2 ** 1.5)


def evaluate(x, a=SIGMA2):
    Z_a, true_fr = (Z_TRUE, TRUE_FRAC_RIGHT) if a == SIGMA2 else true_stats(a)
    edges = np.linspace(GRID_LO, GRID_HI, N_BINS + 1)
    c = 0.5 * (edges[:-1] + edges[1:])
    dx = edges[1] - edges[0]
    ph, _ = np.histogram(x, bins=edges, density=True)
    frac_in = float(((x >= GRID_LO) & (x <= GRID_HI)).mean())
    ph = ph * frac_in                       # mass outside the window is error
    pt = unnorm_density(c, a) / Z_a
    pt = pt / (pt.sum() * dx)

    true_m1, true_sk = true_moments(a)
    frac_right = float((x > 0).mean())
    bridge = float((np.abs(x) < 0.5).mean())
    bridge_true = float(pt[np.abs(c) < 0.5].sum() * dx)
    return {
        'hellinger': float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(ph * pt)) * dx))),
        'tv': float(0.5 * np.sum(np.abs(ph - pt)) * dx),
        'frac_right': frac_right,
        'frac_left': 1.0 - frac_right,
        'frac_right_err': abs(frac_right - true_fr),
        'direction_correct': bool(frac_right < 0.5),   # left well must dominate
        'bridge_learned': bridge,
        'bridge_true': bridge_true,
        'bridge_ratio': float(bridge / bridge_true) if bridge_true > 0 else float('nan'),
        'mean': float(x.mean()),
        'std': float(x.std()),
        'frac_in_window': frac_in,
        # P(x>0) is a THRESHOLD statistic and misses partial asymmetry: a run
        # can shift the density's mass noticeably toward the deeper well while
        # leaving P(x>0) at 0.50. The mean is the sensitive observable, so we
        # also report the fraction of the true mean shift that was recovered.
        # 1.0 = correct asymmetry, 0.0 = perfectly symmetric solution.
        'true_mean': true_m1,
        'true_skew': true_sk,
        'skew': float(((x - x.mean()) ** 3).mean() / max(x.std(), 1e-12) ** 3),
        'asymmetry_captured': float(x.mean() / true_m1) if true_m1 else float('nan'),
    }


# ---------------------------------------------------------------------------
# One training run
# ---------------------------------------------------------------------------
def steady_loss_unbiased(pushforward, testfuncs, M, drift_fn, sigma2_fn,
                         base_sampler, device):
    """Split-batch, BIAS-FREE version of common_wanpm.steady_loss.

    The manuscript's loss squares an M-sample mean, so what it minimises in
    expectation is

        E[ Ehat^2 ]  =  R^2  +  Var(Lf) / M ,

    and the second term -- the estimator bias Reviewer 3 raises in Major
    Comment 2 -- depends on the density being learned. `identifiability.py`
    shows that on this problem the bias term is ~600x LARGER than the R^2
    signal that distinguishes the correct 78:22 well split from a symmetric
    one at M = 1000, so the minimiser of the empirical loss is not the true
    density and no optimizer can recover the split.

    Splitting the (iid) batch into two independent halves and multiplying the
    two half-means gives an unbiased estimate of the same target,

        E[ EhatA * EhatB ]  =  R^2   exactly, for every M,

    at identical sampling and compute cost. The estimate can be negative,
    which is expected -- it is an unbiased estimate of a non-negative quantity
    near zero, not a bug.
    """
    r = base_sampler(M, pushforward.d_base, device)
    x = pushforward(r)
    Lf = testfuncs.Lf(x, drift_fn, sigma2_fn)       # (M, K)
    h = Lf.shape[0] // 2
    EA = Lf[:h].mean(dim=0)
    EB = Lf[h:2 * h].mean(dim=0)
    return (EA * EB).mean(), x


def steady_loss_ustat(pushforward, testfuncs, M, drift_fn, sigma2_fn,
                      base_sampler, device):
    """Unbiased estimate of R^2 using ALL off-diagonal pairs (a U-statistic).

    `steady_loss_unbiased` above removes the Var/M bias by multiplying two
    independent half-batch means, but it throws away most of the pairs and is
    correspondingly noisy -- in practice noisy enough that the result swings
    from one seed to the next.  Averaging over every ordered pair i != j
    instead,

        Rhat^2 = ( (sum_i Lf_i)^2 - sum_i Lf_i^2 ) / (M (M-1)) ,

    has the same expectation R^2 exactly, at the same O(M) cost, with far
    lower variance.  This is the estimator to prefer; the split-batch form is
    kept only because it is the most transparent demonstration that the bias,
    not the optimizer, is what pins the well split at 50:50.
    """
    r = base_sampler(M, pushforward.d_base, device)
    x = pushforward(r)
    Lf = testfuncs.Lf(x, drift_fn, sigma2_fn)       # (M, K)
    m = Lf.shape[0]
    s1 = Lf.sum(dim=0)
    s2 = (Lf ** 2).sum(dim=0)
    r2 = (s1 ** 2 - s2) / (m * (m - 1))
    return r2.mean(), x


def steady_loss_snr(pushforward, testfuncs, M, drift_fn, sigma2_fn,
                    base_sampler, device):
    """ADVERSARY-ONLY objective: the studentized residual Rhat^2 / Var(Lf).

    residual_theory.py shows the weak-form residual of a wrong well split is
        R = (a/2) rho*(0) f'(0) (lambda - lambda*) / [lambda*(1-lambda*)],
    so the test function enters ONLY through f'(0). What the adversary should
    therefore maximise is |f'(0)| measured against the sampling noise, i.e. the
    signal-to-noise ratio -- not the raw loss.

    Ascending the raw BIASED loss maximises R^2 + Var/M, which at M = 1000 is
    ~600x dominated by Var/M: a stronger adversary then chases HIGH-VARIANCE
    (high-frequency) test functions, which is exactly the wrong direction.
    Ascending the raw U-statistic maximises R^2 alone, which for a plane wave
    grows like w^2 with no variance penalty, so it runs to high frequency too.
    Normalising fixes both.
    """
    r = base_sampler(M, pushforward.d_base, device)
    x = pushforward(r)
    Lf = testfuncs.Lf(x, drift_fn, sigma2_fn)
    m = Lf.shape[0]
    s1 = Lf.sum(dim=0)
    s2 = (Lf ** 2).sum(dim=0)
    r2 = (s1 ** 2 - s2) / (m * (m - 1))
    var = (s2 - s1 ** 2 / m) / (m - 1)
    return (r2 / (var + 1e-12)).mean(), x


ESTIMATORS = {'biased': steady_loss,
              'unbiased': steady_loss_unbiased,
              'ustat': steady_loss_ustat,
              'snr': steady_loss_snr}


def run(gen_name, lr_gen, seed=0, n_epochs=N_EPOCHS, anneal=True,
        hidden=None, gen_act='tanh', base_shift=0.0,
        s2_start=SIGMA2_START, head_lr=LR_GEN_PAPER, tilt_preserving=False,
        estimator='biased', batch_m=BATCH_M,
        adv_steps=ADV_STEPS, adv_every=ADV_EVERY, adv_opt='SGD',
        lr_test=LR_TEST_PAPER, adv_objective='same', sigma2_target=SIGMA2,
        activation='sin', wavelet_w0=5.0, init_scale=1.0, init_centre=1.5,
        frac_wave=0.5,
        keep_samples=False, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    hid = HIDDEN if hidden is None else tuple(hidden)
    pf = SteadyPushforward(d_base=D_BASE, n_out=1, hidden=hid).to(device)
    if gen_act != 'tanh':
        # rebuild the MLP with a non-odd activation; common_wanpm.mlp takes the
        # activation class, so the shared library is not modified.
        act_cls = {'gelu': nn.GELU, 'silu': nn.SiLU, 'softplus': nn.Softplus}[gen_act]
        pf.net = mlp(D_BASE, 1, hid, activation=act_cls).to(device)
    sampler = (gaussian_base_sampler if base_shift == 0.0
               else shifted_base_sampler(base_shift))
    if activation == 'sin':
        tf = SteadyTestFunctions(n_in=1, K=K_TEST).to(device)
    elif activation.startswith('mix-'):
        # 'mix-gauss', 'mix-mexhat', ... : half plane waves (global coverage),
        # half wavelets (transition-state sensitivity).
        tf = MixedTestFunctions(n_in=1, K=K_TEST, activation=activation[4:],
                                w0=wavelet_w0, frac_wave=frac_wave,
                                init_scale=init_scale,
                                init_centre_spread=init_centre).to(device)
    else:
        tf = WaveletTestFunctions(n_in=1, K=K_TEST, activation=activation,
                                  w0=wavelet_w0, init_scale=init_scale,
                                  init_centre_spread=init_centre).to(device)

    gen_opt = make_gen_opt(gen_name, pf, lr_gen, head_lr=head_lr)
    if adv_opt == 'SGD':
        test_opt = torch.optim.SGD(tf.parameters(), lr=lr_test)
    elif adv_opt == 'SGDm':
        test_opt = torch.optim.SGD(tf.parameters(), lr=lr_test, momentum=0.9)
    elif adv_opt == 'Adam':
        test_opt = torch.optim.Adam(tf.parameters(), lr=lr_test)
    elif adv_opt == 'AdamW':
        test_opt = torch.optim.AdamW(tf.parameters(), lr=lr_test, weight_decay=1e-2)
    else:
        raise ValueError(adv_opt)

    cur = [sigma2_target]
    cur_eps = [EPS]

    def sigma2_cur(x):
        return cur[0] * torch.ones(x.shape[0], 1, device=x.device)

    def drift_cur(x):
        return drift_fn_eps(x, cur_eps[0])

    loss_impl = ESTIMATORS[estimator]
    adv_impl = ESTIMATORS['snr'] if adv_objective == 'normalized' else loss_impl

    def loss_fn():
        return loss_impl(pf, tf, M=batch_m, drift_fn=drift_cur,
                         sigma2_fn=sigma2_cur,
                         base_sampler=sampler, device=device)

    def adv_loss_fn():
        return adv_impl(pf, tf, M=batch_m, drift_fn=drift_cur,
                        sigma2_fn=sigma2_cur,
                        base_sampler=sampler, device=device)

    losses = np.empty(n_epochs)
    sig_log = np.empty(n_epochs)
    # track the right-well fraction through training: the whole question is
    # WHEN (and whether) the asymmetry gets established
    trace_every = max(1, n_epochs // 200)
    trace = []

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    diverged = False
    for epoch in range(n_epochs):
        cur[0] = sigma2_at_epoch(epoch, n_epochs, anneal, s2_start, sigma2_target)
        cur_eps[0] = eps_at_sigma2(cur[0], tilt_preserving)
        sig_log[epoch] = cur[0]

        if epoch % adv_every == 0 and epoch > 0:
            for _ in range(adv_steps):
                l, _ = adv_loss_fn()
                test_opt.zero_grad(); (-l).backward(); test_opt.step()

        l, _ = loss_fn()
        gen_opt.zero_grad(); l.backward(); gen_opt.step()
        lv = l.item()
        losses[epoch] = lv
        if not np.isfinite(lv):
            diverged = True
            losses[epoch:] = np.nan
            sig_log[epoch:] = np.nan
            break

        if epoch % trace_every == 0:
            with torch.no_grad():
                xs = pf(torch.randn(4000, D_BASE, device=device)).ravel()
                trace.append((epoch, float(cur[0]), float((xs > 0).float().mean())))
        if verbose and (epoch % 4000 == 0 or epoch == n_epochs - 1):
            print(f"    epoch {epoch:6d}/{n_epochs}  s2={cur[0]:.3f}  loss={lv:.4e}",
                  flush=True)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    dt = time.time() - t0

    rec = {'gen_opt': gen_name, 'lr_gen': lr_gen, 'head_lr': head_lr,
           'seed': seed, 'n_epochs': n_epochs, 'anneal': anneal,
           's2_start': s2_start if anneal else None,
           'tilt_preserving': tilt_preserving,
           'estimator': estimator, 'batch_m': batch_m,
           'adv_steps': adv_steps, 'adv_every': adv_every, 'adv_opt': adv_opt,
           'lr_test': lr_test, 'adv_objective': adv_objective,
           'sigma2_target': sigma2_target, 'activation': activation,
           'wavelet_w0': wavelet_w0, 'init_scale': init_scale,
           'frac_wave': frac_wave,
           'init_centre': init_centre,
           'true_frac_right': (TRUE_FRAC_RIGHT if sigma2_target == SIGMA2
                               else true_stats(sigma2_target)[1]),
           'train_time_s': dt, 'diverged': diverged,
           'frac_right_trace': trace}
    x = None
    if diverged:
        rec.update({k: float('nan') for k in
                    ('hellinger', 'tv', 'frac_right', 'frac_left',
                     'frac_right_err', 'bridge_learned', 'bridge_true',
                     'bridge_ratio', 'mean', 'std', 'frac_in_window',
                     'skew', 'asymmetry_captured', 'final_loss')})
        rec['direction_correct'] = False
    else:
        with torch.no_grad():
            r0 = sampler(N_EVAL, D_BASE, device)
            x = pf(r0).cpu().numpy().ravel()
            # Is the learned map ODD?  With a spherically symmetric Gaussian
            # base, r and -r are equally likely, so an exactly odd F forces an
            # exactly symmetric output density and hence P(x>0) = 0.5 EXACTLY.
            # The tanh MLP is nearly odd at initialisation (tanh is odd; only
            # the biases break it), so this measures whether training ever
            # breaks that symmetry. even_frac = ||F_even|| / ||F||, where
            # F_even(r) = (F(r) + F(-r))/2.
            # reflect about the BASE MEAN: T(r) = 2 mu - r is the involution
            # that preserves the base measure, so it reduces to r -> -r when
            # the base is centred. The component of F invariant under T is what
            # breaks the output density's symmetry (a T-anti-invariant F would
            # push forward to a density symmetric about 0).
            xm = pf(2.0 * base_shift - r0).cpu().numpy().ravel()
            f_even = 0.5 * (x + xm)
            f_odd = 0.5 * (x - xm)
            n_e = float(np.sqrt((f_even ** 2).mean()))
            n_o = float(np.sqrt((f_odd ** 2).mean()))
        rec.update(evaluate(x, a=sigma2_target))
        rec['even_norm'] = n_e
        rec['odd_norm'] = n_o
        rec['even_frac'] = n_e / max(np.hypot(n_e, n_o), 1e-12)
        rec['n_params'] = int(sum(q.numel() for q in pf.parameters()))
        rec['gen_act'] = gen_act
        rec['base_shift'] = base_shift
        rec['final_loss'] = float(np.nanmean(losses[-500:]))
    rec['loss_log_every50'] = losses[::50].tolist()
    if keep_samples:
        return rec, x, losses, sig_log
    return rec


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
SWEEP_GRID = {
    # Adam around the manuscript's own 2e-3, extended upward toward the
    # 2e-2 that stage 2's R1_5 actually used.
    'Adam': [1e-3, 2e-3, 5e-3, 1e-2, 2e-2],
    # Muon's usable band on the symmetric Sec. 4.2 problem was 1e-1..5e-1
    # (everything at and below 5e-2 collapsed); bracket that on both sides.
    'Muon': [2e-2, 5e-2, 1e-1, 2e-1, 5e-1],
}

# A run is "collapsed" when essentially all the mass is in one well. Note the
# asymmetry of the criterion: the TRUE answer here is already lopsided
# (frac_right = 0.216), so unlike the symmetric case we cannot call any
# imbalance a failure. We only reject near-total collapse.
COLLAPSE_LO, COLLAPSE_HI = 0.02, 0.98
SEEDS = [0, 1, 2]


def _save(recs, fname):
    with open(os.path.join(OUTDIR, fname), 'w') as fh:
        json.dump(recs, fh, indent=2)
    print(f"  -> wrote {fname} ({len(recs)} runs)", flush=True)


def _line(rec):
    if rec['diverged']:
        return f"DIVERGED  ({rec['train_time_s']:.0f}s)"
    tfr = rec.get('true_frac_right', TRUE_FRAC_RIGHT)
    return (f"frac_right={rec['frac_right']:.4f} (true {tfr:.4f}, "
            f"err={rec['frac_right_err']:.4f})  H={rec['hellinger']:.4f} "
            f"TV={rec['tv']:.4f}  bridge={rec['bridge_ratio']:.2f}x  "
            f"({rec['train_time_s']:.0f}s)"
            f"{'' if rec['direction_correct'] else '  [WRONG WELL]'}")


def mode_sweep(args):
    recs = []
    for name, lrs in SWEEP_GRID.items():
        for lr in lrs:
            print(f"[sweep] {name:6s} lr={lr:g} ...", flush=True)
            rec = run(name, lr, seed=0, n_epochs=args.epochs)
            print(f"         {_line(rec)}", flush=True)
            recs.append(rec)
            _save(recs, 'sweep.json')
    return recs


def _finalists(sweep, n=2):
    out = {}
    for name in SWEEP_GRID:
        cand = [r for r in sweep if r['gen_opt'] == name and not r['diverged']
                and np.isfinite(r['hellinger'])]
        good = [r for r in cand if COLLAPSE_LO < r['frac_right'] < COLLAPSE_HI]
        pool = good if good else cand
        if not pool:
            print(f"[seeds] {name}: nothing finite, skipping")
            continue
        # rank by the quantity R1.5 is judged on, not by Hellinger: a run can
        # look good in Hellinger while still getting the population split (and
        # even its direction) wrong.
        pool = sorted(pool, key=lambda r: r['frac_right_err'])[:n]
        out[name] = [r['lr_gen'] for r in pool]
        print(f"[seeds] {name:6s} finalist lrs = "
              f"{', '.join(f'{l:g}' for l in out[name])}"
              f"{'' if good else '  (every LR collapsed)'}")
    return out


def mode_seeds(args):
    path = os.path.join(OUTDIR, 'sweep.json')
    if not os.path.exists(path):
        raise SystemExit("run --mode sweep first")
    with open(path) as fh:
        sweep = json.load(fh)
    fin = _finalists(sweep)

    recs, samples = [], {}
    for name, lrs in fin.items():
        for lr in lrs:
            for seed in SEEDS:
                print(f"[seeds] {name:6s} lr={lr:g} seed={seed} ...", flush=True)
                rec, x, losses, sig = run(name, lr, seed=seed,
                                          n_epochs=args.epochs, keep_samples=True)
                print(f"         {_line(rec)}", flush=True)
                recs.append(rec)
                if x is not None:
                    samples[f"{name}_lr{lr:g}_s{seed}"] = x.astype(np.float32)
                    if seed == 0:
                        samples[f"{name}_lr{lr:g}_s{seed}_loss"] = \
                            losses[::25].astype(np.float32)
                        samples[f"{name}_lr{lr:g}_s{seed}_sigma2"] = \
                            sig[::25].astype(np.float32)
                _save(recs, 'seeds.json')
                np.savez_compressed(os.path.join(OUTDIR, 'seed_samples.npz'), **samples)
    return recs


def mode_anneal(args):
    """Curriculum control at each optimizer's best LR from --mode seeds."""
    path = os.path.join(OUTDIR, 'seeds.json')
    if not os.path.exists(path):
        raise SystemExit("run --mode seeds first")
    with open(path) as fh:
        seeds = json.load(fh)

    best = {}
    for name in SWEEP_GRID:
        by_lr = {}
        for r in seeds:
            if r['gen_opt'] == name and not r['diverged']:
                by_lr.setdefault(r['lr_gen'], []).append(r['frac_right_err'])
        if by_lr:
            best[name] = min(by_lr, key=lambda k: float(np.mean(by_lr[k])))
            print(f"[anneal] {name:6s} best lr (seed-mean) = {best[name]:g}")

    variants = [('no-curriculum', dict(anneal=False)),
                ('anneal-from-0.6', dict(anneal=True, s2_start=0.6)),
                ('anneal-from-1.5', dict(anneal=True, s2_start=1.5)),
                ('tilt-preserving', dict(anneal=True, s2_start=1.5,
                                         tilt_preserving=True))]
    recs, samples = [], {}
    for name, lr in best.items():
        for tag, kw in variants:
            for seed in SEEDS:
                print(f"[anneal] {name:6s} lr={lr:g} {tag:16s} seed={seed} ...",
                      flush=True)
                rec, x, _, _ = run(name, lr, seed=seed, n_epochs=args.epochs,
                                   keep_samples=True, **kw)
                rec['variant'] = tag
                print(f"          {_line(rec)}", flush=True)
                recs.append(rec)
                if x is not None:
                    samples[f"{name}_{tag}_s{seed}"] = x.astype(np.float32)
                _save(recs, 'anneal.json')
                np.savez_compressed(os.path.join(OUTDIR, 'anneal_samples.npz'),
                                    **samples)
    return recs


# Best LR per optimizer from the seed-0 sweep (ranked by frac_right_err among
# non-collapsed runs). Both sit on a flat plateau, so the exact pick is not
# load-bearing.
BEST_LR = {'Adam': 5e-3, 'Muon': 1e-1}


def mode_estimator(args):
    """THE FIX. Biased (manuscript) vs unbiased (split-batch) loss estimator,
    at identical compute, for both optimizers."""
    recs, samples = [], {}
    for name, lr in BEST_LR.items():
        for est in ('biased', 'unbiased'):
            for seed in SEEDS:
                print(f"[est] {name:5s} lr={lr:g} {est:9s} seed={seed} ...",
                      flush=True)
                rec, x, losses, sig = run(name, lr, seed=seed, estimator=est,
                                          n_epochs=args.epochs, keep_samples=True)
                print(f"       {_line(rec)}", flush=True)
                recs.append(rec)
                if x is not None:
                    samples[f"{name}_{est}_s{seed}"] = x.astype(np.float32)
                    if seed == 0:
                        samples[f"{name}_{est}_s{seed}_loss"] = \
                            losses[::25].astype(np.float32)
                _save(recs, 'estimator.json')
                np.savez_compressed(
                    os.path.join(OUTDIR, 'estimator_samples.npz'), **samples)
    return recs


def mode_ustat(args):
    """The low-variance unbiased estimator, both optimizers x 3 seeds.
    Written to its own file so it can run alongside --mode estimator."""
    recs, samples = [], {}
    for name, lr in BEST_LR.items():
        for seed in SEEDS:
            print(f"[ustat] {name:5s} lr={lr:g} seed={seed} ...", flush=True)
            rec, x, losses, sig = run(name, lr, seed=seed, estimator='ustat',
                                      n_epochs=args.epochs, keep_samples=True)
            print(f"        {_line(rec)}", flush=True)
            recs.append(rec)
            if x is not None:
                samples[f"{name}_ustat_s{seed}"] = x.astype(np.float32)
                if seed == 0:
                    samples[f"{name}_ustat_s{seed}_loss"] = \
                        losses[::25].astype(np.float32)
            _save(recs, 'ustat.json')
            np.savez_compressed(os.path.join(OUTDIR, 'ustat_samples.npz'), **samples)
    return recs


def mode_combo(args):
    """Both levers at once.

    identifiability.py says two separate things are wrong at M = 1000:
      (i)  the Var/M bias term dominates the R^2 signal by ~600x, so the loss
           is minimised somewhere other than the true density  -> fixed by the
           unbiased U-statistic estimator;
      (ii) even without the bias, the R^2 signal that distinguishes a 78:22
           split from 50:50 is tiny in absolute terms, so the gradient in that
           direction is buried in sampling noise -> only a larger M fixes that,
           since the noise on the estimator falls as 1/M.
    Fixing (i) alone left Muon pinned at 0.50. This runs (i) AND (ii) together,
    trading epochs for batch size at ~4x the baseline compute.
    """
    recs, samples = [], {}
    m, ep = args.combo_m, args.combo_epochs
    for name, lr in BEST_LR.items():
        for seed in SEEDS:
            print(f"[combo] {name:5s} lr={lr:g} M={m} epochs={ep} seed={seed} ...",
                  flush=True)
            rec, x, losses, sig = run(name, lr, seed=seed, estimator='ustat',
                                      batch_m=m, n_epochs=ep, keep_samples=True)
            print(f"         {_line(rec)}", flush=True)
            recs.append(rec)
            if x is not None:
                samples[f"{name}_combo_s{seed}"] = x.astype(np.float32)
            _save(recs, 'combo.json')
            np.savez_compressed(os.path.join(OUTDIR, 'combo_samples.npz'), **samples)
    return recs


# ---------------------------------------------------------------------------
# ADVERSARY STRENGTH.
#
# If the residual R only depends on the test function through f'(0)
# (residual_theory.py), then whether the adversary FINDS a good f matters, and
# the paper's setting -- SGD lr 1e-2, 5 ascent steps every 10 epochs -- may
# simply be too weak to get there. This sweeps ascent steps, adversary
# optimizer and adversary learning rate.
#
# Note which objective the adversary ascends:
#   'biased'      -> maximises R^2 + Var/M, which at M=1000 is ~600x dominated
#                    by Var/M, so a STRONGER adversary chases variance harder.
#   'ustat'       -> maximises R^2 alone; for a plane wave R ~ w cos(b), so it
#                    pushes to high frequency with no variance penalty.
#   'normalized'  -> maximises R^2/Var, the actual detection SNR. This is what
#                    the theory says the adversary should be doing.
# ---------------------------------------------------------------------------
ADV_SWEEP = [
    # (tag,                 estimator, adv_steps, adv_every, adv_opt, lr_test, adv_objective)
    ('paper-baseline',      'biased', 5,  10, 'SGD',  1e-2, 'same'),
    ('ustat-baseline',      'ustat',  5,  10, 'SGD',  1e-2, 'same'),
    # more ascent steps
    ('ustat-20steps',       'ustat',  20, 10, 'SGD',  1e-2, 'same'),
    ('ustat-50steps',       'ustat',  50, 10, 'SGD',  1e-2, 'same'),
    ('ustat-every1',        'ustat',  5,  1,  'SGD',  1e-2, 'same'),
    # adversary learning rate, up and down
    ('ustat-lr1e-3',        'ustat',  20, 10, 'SGD',  1e-3, 'same'),
    ('ustat-lr1e-1',        'ustat',  20, 10, 'SGD',  1e-1, 'same'),
    ('ustat-lr1',           'ustat',  20, 10, 'SGD',  1e0,  'same'),
    # better adversary optimizers
    ('ustat-Adam1e-2',      'ustat',  20, 10, 'Adam', 1e-2, 'same'),
    ('ustat-Adam1e-1',      'ustat',  20, 10, 'Adam', 1e-1, 'same'),
    ('ustat-SGDm1e-2',      'ustat',  20, 10, 'SGDm', 1e-2, 'same'),
    # theory-motivated: adversary maximises the SNR, not the raw loss
    ('ustat-normadv',       'ustat',  20, 10, 'Adam', 1e-2, 'normalized'),
    ('ustat-normadv-50',    'ustat',  50, 10, 'Adam', 1e-2, 'normalized'),
    ('biased-normadv',      'biased', 20, 10, 'Adam', 1e-2, 'normalized'),
]


def _gens(args):
    return {k: v for k, v in BEST_LR.items()
            if args.gen == 'all' or k == args.gen}


def mode_adversary2(args):
    recs, samples = [], {}
    for name, lr in _gens(args).items():
        for (tag, est, st, ev, aopt, alr, aobj) in ADV_SWEEP:
            for seed in SEEDS[:args.adv_seeds]:
                print(f"[adv] {name:5s} {tag:18s} seed={seed} ...", flush=True)
                rec, x, _, _ = run(name, lr, seed=seed, estimator=est,
                                   adv_steps=st, adv_every=ev, adv_opt=aopt,
                                   lr_test=alr, adv_objective=aobj,
                                   n_epochs=args.epochs, keep_samples=True)
                rec['variant'] = tag
                print(f"      {_line(rec)}", flush=True)
                recs.append(rec)
                if x is not None and seed == 0:
                    samples[f"{name}_{tag}_s0"] = x.astype(np.float32)
                _save(recs, 'adversary2.json')
                np.savez_compressed(os.path.join(OUTDIR, 'adversary2_samples.npz'),
                                    **samples)
    return recs


# ---------------------------------------------------------------------------
# BARRIER SWEEP -- the theory's sharpest testable prediction.
#
# residual_theory.py says the batch size needed to resolve the split is
#     sigma^2 = 1.2 -> M ~ 1.1e3      0.45 -> M ~ 2.3e4
#     sigma^2 = 0.9 -> M ~ 1.6e3      0.30 -> M ~ 6.1e5   <- the R1.5 setting
#     sigma^2 = 0.6 -> M ~ 5.4e3      0.20 -> M ~ 1.3e8
# so at M = 1000 the method should RECOVER the asymmetry for mild barriers and
# fail for stiff ones, with the crossover around sigma^2 ~ 0.9-1.2. If the
# learned split tracks the truth above the crossover and departs below it, the
# whole analysis is confirmed end to end -- and R1.5 becomes a characterised
# limitation with a known validity boundary rather than a bare failure.
# ---------------------------------------------------------------------------
BARRIER_S2 = [1.2, 0.9, 0.6, 0.45, 0.3]


def mode_barrier(args):
    recs, samples = [], {}
    for name, lr in _gens(args).items():
        for s2 in BARRIER_S2:
            # curriculum must start above the target; skip annealing when the
            # target is already at/above the usual start
            anneal = s2 < SIGMA2_START
            for seed in SEEDS[:args.adv_seeds]:
                print(f"[barrier] {name:5s} sigma2={s2:g} seed={seed} ...", flush=True)
                rec, x, _, _ = run(name, lr, seed=seed, estimator=args.barrier_est,
                                   sigma2_target=s2, anneal=anneal,
                                   n_epochs=args.epochs, keep_samples=True)
                print(f"          {_line(rec)}", flush=True)
                recs.append(rec)
                if x is not None and seed == 0:
                    samples[f"{name}_s2{s2:g}_s0"] = x.astype(np.float32)
                tag = ('barrier' if args.barrier_est == 'biased'
                       else 'barrier_' + args.barrier_est)
                _save(recs, tag + '.json')
                np.savez_compressed(os.path.join(OUTDIR, tag + '_samples.npz'),
                                    **samples)
    return recs


def mode_mscale(args):
    """How large must M be, at the actual R1.5 setting (sigma^2 = 0.3)?

    residual_theory.py predicts the split becomes resolvable around
    M ~ 6e5. The unbiased (U-statistic) loss is used throughout, since with
    the manuscript's biased loss the Var/M term is systematic and raising M
    alone cannot fix it. Epochs are traded against M; large batches are
    cheap on GPU, so the wall clock stays roughly flat.
    """
    plan = [(1000, 16000), (4000, 12000), (16000, 8000),
            (64000, 4000), (256000, 2000)]
    recs = []
    for name, lr in _gens(args).items():
        for m, ep in plan:
            for seed in SEEDS[:args.adv_seeds]:
                print(f"[mscale] {name:5s} M={m:>7d} epochs={ep:>6d} seed={seed} ...",
                      flush=True)
                rec = run(name, lr, seed=seed, estimator='ustat',
                          batch_m=m, n_epochs=ep)
                print(f"         {_line(rec)}  captured="
                      f"{100*rec['asymmetry_captured']:.1f}%", flush=True)
                recs.append(rec)
                _save(recs, 'mscale.json')
    return recs


MIX_CONFIGS = [
    # (activation, w0, init_scale, frac_wave)
    ('sin', 5.0, 1.0, 1.0),            # baseline: all plane waves
    ('mix-gauss', 5.0, 4.0, 0.5),
    ('mix-gauss', 5.0, 8.0, 0.5),      # narrower wavelets (nearer the optimum |w|~8)
    ('mix-gauss', 5.0, 4.0, 0.75),     # mostly plane waves, a quarter wavelets
    ('mix-mexhat', 5.0, 4.0, 0.5),     # mexhat alone escaped; does mixing rescue it?
    ('mix-morlet', 2.0, 4.0, 0.5),
]


NORMADV_CONFIGS = [
    # (activation, w0, init_scale, frac_wave, adv_objective, adv_steps, adv_opt, lr_test)
    ('sin',       5.0, 1.0, 1.0, 'same',       5,  'SGD',  1e-2),   # manuscript baseline
    ('mix-gauss', 5.0, 4.0, 0.5, 'same',       5,  'SGD',  1e-2),   # mixing only
    ('mix-gauss', 5.0, 4.0, 0.5, 'normalized', 5,  'SGD',  1e-2),   # + SNR adversary
    ('mix-gauss', 5.0, 4.0, 0.5, 'normalized', 20, 'Adam', 1e-2),   # + stronger SNR adversary
    ('gauss',     5.0, 4.0, 0.0, 'normalized', 20, 'Adam', 1e-2),   # wavelets only, SNR adversary
]


def mode_normadv(args):
    """Mixed bank PLUS an adversary that maximises SNR rather than raw loss.

    wavelet_placement.py shows the two reasons the 71x SNR advantage of a
    Gaussian wavelet does not reach the loss:
      (i)  the loss averages over K=100, so the bank's MEAN SNR is what counts
           and that is only 1.1% of the best single wavelet (0.79x the plane
           wave's best) -- addressed by mixing in global plane waves;
      (ii) the adversary maximises the raw loss, which is dominated by
           variance, and so drives wavelets AWAY from the transition state
           (fraction with |centre|<0.3 falls 0.21 -> 0.11 during training) --
           addressed by the 'normalized' adversary objective R^2/Var, which is
           the detection SNR itself.
    """
    recs, samples = [], {}
    for name, lr in _gens(args).items():
        for (act, w0, iscale, fw, aobj, ast, aopt, alr) in NORMADV_CONFIGS:
            for seed in SEEDS[:args.adv_seeds]:
                tag = f"{act}_fw{fw:g}_{aobj[:4]}_{aopt}{ast}"
                print(f"[nadv] {name:5s} {tag:28s} seed={seed} ...", flush=True)
                rec, x, _, _ = run(name, lr, seed=seed, estimator='ustat',
                                   activation=act, wavelet_w0=w0,
                                   init_scale=iscale, frac_wave=fw,
                                   adv_objective=aobj, adv_steps=ast,
                                   adv_opt=aopt, lr_test=alr,
                                   batch_m=args.mix_m, n_epochs=args.epochs,
                                   keep_samples=True)
                rec['variant'] = tag
                print(f"       {_line(rec)}  captured="
                      f"{100*rec['asymmetry_captured']:.1f}%  "
                      f"in_win={rec['frac_in_window']:.3f}", flush=True)
                recs.append(rec)
                if x is not None and seed == 0:
                    samples[f"{name}_{tag}_s0"] = x.astype(np.float32)
                _save(recs, 'normadv.json')
                np.savez_compressed(os.path.join(OUTDIR, 'normadv_samples.npz'),
                                    **samples)
    return recs


CAPACITY_HIDDEN = [(100,), (400,), (100, 100), (64, 64, 64), (256, 256)]


SYMBREAK_CONFIGS = [
    # (gen_act, base_shift)
    ('tanh', 0.0),    # baseline: odd activation, centred base
    ('gelu', 0.0),    # non-odd activation
    ('tanh', 1.0),    # odd activation, shifted base (init-only symmetry break)
    ('gelu', 1.0),    # both
]


def mode_symbreak(args):
    """Break the generator's odd symmetry: GELU activation and/or a shifted base.

    tanh is odd, so with small biases the map starts nearly odd, and a
    spherically symmetric base then forces P(x>0) = 0.5 exactly. This tests
    whether removing that symmetry lets the (tiny) tilt signal pick a side.
    Run in the best-known configuration: 75/25 plane-wave/Gaussian-wavelet
    bank, Adam + U-statistic, M = 1000.
    """
    recs = []
    for name, lr in _gens(args).items():
        for (act, shift) in SYMBREAK_CONFIGS:
            for seed in SEEDS[:args.adv_seeds]:
                tag = f"{act}_shift{shift:g}"
                print(f"[sym] {name:5s} {tag:16s} seed={seed} ...", flush=True)
                rec = run(name, lr, seed=seed, estimator='ustat',
                          activation='mix-gauss', init_scale=4.0, frac_wave=0.75,
                          gen_act=act, base_shift=shift, n_epochs=args.epochs)
                rec['variant'] = tag
                print(f"      {_line(rec)}  captured="
                      f"{100*rec['asymmetry_captured']:.1f}%  "
                      f"sym_frac={rec['even_frac']:.4f}", flush=True)
                recs.append(rec)
                _save(recs, 'symbreak.json')
    return recs


def mode_capacity(args):
    """Is the 1001-parameter generator big enough?

    Run the best-known configuration (75/25 plane-wave/Gaussian-wavelet bank,
    Adam + U-statistic) across generator architectures spanning 1e3 to 1e5
    parameters. Also records `even_frac`, the fraction of the learned map that
    is EVEN: with a spherically symmetric Gaussian base an exactly odd map
    forces P(x>0) = 0.5 exactly, and a tanh MLP is nearly odd at
    initialisation, so this tests whether capacity or symmetry is the binding
    constraint.
    """
    recs = []
    for name, lr in _gens(args).items():
        for hid in CAPACITY_HIDDEN:
            for seed in SEEDS[:args.adv_seeds]:
                tag = 'x'.join(map(str, hid))
                print(f"[cap] {name:5s} hidden={tag:14s} seed={seed} ...", flush=True)
                rec = run(name, lr, seed=seed, estimator='ustat',
                          activation='mix-gauss', init_scale=4.0, frac_wave=0.75,
                          hidden=hid, n_epochs=args.epochs)
                rec['variant'] = tag
                print(f"      {_line(rec)}  captured="
                      f"{100*rec['asymmetry_captured']:.1f}%  "
                      f"params={rec['n_params']}  even_frac={rec['even_frac']:.4f}",
                      flush=True)
                recs.append(rec)
                _save(recs, 'capacity.json')
    return recs


def mode_mixed(args):
    """Half global plane waves, half localized wavelets.

    A purely localized bank lets the generator escape the test functions'
    support entirely (mexhat/morlet: only 2-6% of samples left in the window,
    std 9-39 vs a true 0.97). Keeping plane waves in the bank should close that
    escape route while the wavelets supply the transition-state sensitivity."""
    recs, samples = [], {}
    for name, lr in _gens(args).items():
        for (act, w0, iscale, fw) in MIX_CONFIGS:
            for seed in SEEDS[:args.adv_seeds]:
                tag = f"{act}_s{iscale:g}_fw{fw:g}"
                print(f"[mix] {name:5s} {tag:24s} seed={seed} ...", flush=True)
                rec, x, _, _ = run(name, lr, seed=seed, estimator='ustat',
                                   activation=act, wavelet_w0=w0,
                                   init_scale=iscale, frac_wave=fw,
                                   batch_m=args.mix_m, n_epochs=args.epochs,
                                   keep_samples=True)
                rec['variant'] = tag
                print(f"      {_line(rec)}  captured="
                      f"{100*rec['asymmetry_captured']:.1f}%  "
                      f"in_win={rec['frac_in_window']:.3f}", flush=True)
                recs.append(rec)
                if x is not None and seed == 0:
                    samples[f"{name}_{tag}_s0"] = x.astype(np.float32)
                _save(recs, 'mixed.json')
                np.savez_compressed(os.path.join(OUTDIR, 'mixed_samples.npz'),
                                    **samples)
    return recs


ACT_CONFIGS = [
    # (activation, w0, init_scale)  -- init_scale sets the initial wavelet
    # width ~ 1/|w|. wavelet_snr.py puts the OPTIMUM at |w| ~ 8 (width 0.125)
    # centred on the transition state, so a wavelet initialised at |w| ~ 1 has
    # a long way to travel; 4 starts it much closer.
    ('sin', 5.0, 1.0),
    ('gauss', 5.0, 4.0),
    ('mexhat', 5.0, 4.0),
    ('morlet', 2.0, 4.0),
]
ACT_BATCHES = [(1000, 16000), (16000, 8000)]


def mode_activation(args):
    """Swap the test-function activation phi for a localized (wavelet) one.

    wavelet_snr.py predicts, by quadrature, that the batch size needed to
    resolve the well split drops from 6.0e5 (plane wave) to 8.5e3 (gauss),
    1.6e4 (mexhat) or 1.8e4 (morlet w0=2) -- a 35-70x reduction that brings the
    problem inside reach. This tests that prediction by actually training.

    The unbiased U-statistic loss is used throughout: the SNR argument is about
    sampling noise, and under the manuscript's biased loss the Var/M term is a
    systematic offset that no amount of SNR would remove.
    """
    recs, samples = [], {}
    for name, lr in _gens(args).items():
        for (act, w0, iscale) in ACT_CONFIGS:
            for (m, ep) in ACT_BATCHES:
                for seed in SEEDS[:args.adv_seeds]:
                    print(f"[act] {name:5s} {act:7s} M={m:>6d} seed={seed} ...",
                          flush=True)
                    rec, x, _, _ = run(name, lr, seed=seed, estimator='ustat',
                                       activation=act, wavelet_w0=w0,
                                       init_scale=iscale, batch_m=m,
                                       n_epochs=ep, keep_samples=True)
                    rec['variant'] = f"{act}_M{m}"
                    print(f"      {_line(rec)}  captured="
                          f"{100*rec['asymmetry_captured']:.1f}%", flush=True)
                    recs.append(rec)
                    if x is not None and seed == 0:
                        samples[f"{name}_{act}_M{m}_s0"] = x.astype(np.float32)
                    _save(recs, 'activation.json')
                    np.savez_compressed(
                        os.path.join(OUTDIR, 'activation_samples.npz'), **samples)
    return recs


def mode_batch(args):
    """Confirmatory: with the BIASED loss, does raising M alone move the split
    the way identifiability.py predicts?  Epochs are cut as M grows to keep the
    wall clock bounded; the sample budget M*epochs still rises 8x."""
    plan = [(1000, 16000), (4000, 8000), (16000, 4000), (64000, 2000)]
    recs = []
    for name in ('Muon',):
        for m, ep in plan:
            for seed in SEEDS[:2]:
                print(f"[batch] {name:5s} M={m} epochs={ep} seed={seed} ...",
                      flush=True)
                rec = run(name, BEST_LR[name], seed=seed, batch_m=m,
                          n_epochs=ep, estimator='biased')
                print(f"        {_line(rec)}", flush=True)
                recs.append(rec)
                _save(recs, 'batch.json')
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True,
                    choices=['smoke', 'sweep', 'seeds', 'anneal', 'estimator',
                             'ustat', 'combo', 'adversary2', 'barrier', 'mscale', 'activation', 'mixed', 'normadv', 'capacity', 'symbreak',
                             'batch', 'plot'])
    ap.add_argument('--epochs', type=int, default=N_EPOCHS)
    ap.add_argument('--combo-m', type=int, default=8000)
    ap.add_argument('--combo-epochs', type=int, default=8000)
    ap.add_argument('--adv-seeds', type=int, default=1)
    ap.add_argument('--mix-m', type=int, default=1000)
    ap.add_argument('--gen', default='all', choices=['all', 'Adam', 'Muon'])
    ap.add_argument('--barrier-est', default='biased',
                    choices=['biased', 'ustat'])
    args = ap.parse_args()

    print(f"device: {device}"
          f"{'  ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")
    print(f"tilted double well: eps={EPS}, a=sigma^2={SIGMA2}")
    print(f"true populations (quadrature): left={TRUE_FRAC_LEFT:.4f}  "
          f"right={TRUE_FRAC_RIGHT:.4f}")
    _, fr_start = true_stats(SIGMA2_START)
    print(f"  (at the curriculum start sigma^2={SIGMA2_START}: "
          f"left={1-fr_start:.4f} right={fr_start:.4f})")
    print(f"config: d_base={D_BASE}, hidden={HIDDEN}, K={K_TEST}, "
          f"M={BATCH_M}, epochs={args.epochs}")

    if args.mode == 'smoke':
        for name, lrs in SWEEP_GRID.items():
            lr = lrs[len(lrs) // 2]
            rec = run(name, lr, seed=0, n_epochs=300)
            print(f"  {name:6s} lr={lr:g}  {_line(rec)}  "
                  f"-> est {rec['train_time_s']*N_EPOCHS/300/60:.1f} min for "
                  f"{N_EPOCHS} epochs", flush=True)
    elif args.mode == 'sweep':
        mode_sweep(args)
    elif args.mode == 'seeds':
        mode_seeds(args)
    elif args.mode == 'anneal':
        mode_anneal(args)
    elif args.mode == 'estimator':
        mode_estimator(args)
    elif args.mode == 'ustat':
        mode_ustat(args)
    elif args.mode == 'combo':
        mode_combo(args)
    elif args.mode == 'adversary2':
        mode_adversary2(args)
    elif args.mode == 'barrier':
        mode_barrier(args)
    elif args.mode == 'mscale':
        mode_mscale(args)
    elif args.mode == 'activation':
        mode_activation(args)
    elif args.mode == 'mixed':
        mode_mixed(args)
    elif args.mode == 'capacity':
        mode_capacity(args)
    elif args.mode == 'symbreak':
        mode_symbreak(args)
    elif args.mode == 'normadv':
        mode_normadv(args)
    elif args.mode == 'batch':
        mode_batch(args)
    elif args.mode == 'plot':
        import make_plots
        make_plots.main(OUTDIR)


if __name__ == '__main__':
    main()
