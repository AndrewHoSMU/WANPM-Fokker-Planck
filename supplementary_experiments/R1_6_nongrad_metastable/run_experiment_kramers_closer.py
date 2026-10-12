"""
R1.6 Kramers metastable example -- CLOSER WELLS + mixed test-function bank.

Companion to run_experiment_kramers.py (kept as-is). That script runs the
classic underdamped/Kramers double well in phase space (x, v),

    dx = v dt,
    dv = (-gamma v - U'(x)) dt + sqrt(2 gamma T) dW,   U(x) = (x^2 - 1)^2,

with gamma=1, T=0.25. It is genuinely non-gradient in the full (x,v) phase
space and hypoelliptic (diffusion enters only the v-equation). The steady
state factorises exactly,

    rho(x,v) ~ exp(-H/T),  H = U(x) + v^2/2   =>   rho_x(x) ~ exp(-U(x)/T),
                                                   rho_v(v)  = N(0, T).

At U(x)=(x^2-1)^2 (wells at x=+-1, separation 2.0) the relative barrier is
2 dU / T = 8 and the connectivity rho*(0)/rho*_peak ~ 0.018 -- a nearly
disconnected instance. run_experiment_kramers.py fails there: the learned
x-marginal has a large spurious spike at the transition state x=0 and
under-resolved wells (results_kramers/kramers_plot.png).

This is the SAME transition-state over-population artifact diagnosed in the
R1.5 asymmetric-well study (R1_5_asymmetric_muon/RESULTS.md),
whose two working fixes were:
  (1) a milder instance with the wells closer together, solved with the
      method completely unmodified;
  (2) a mixed plane-wave / Gaussian-wavelet test-function bank -- mostly
      plane waves for well-posedness, a minority of wavelets for
      transition-state sensitivity.

This script applies that recipe here. The potential is generalised to
U(x) = (x^2 - a^2)^2 (wells at x = +-a, still SYMMETRIC), and a is swept.
The Kramers phase-space dynamics are unchanged and remain non-gradient for
any U. No U-statistic, no Muon: plain Adam exactly as in the current
Kramers script.

  a       separation 2a   2 dU / T   connectivity rho*(0)/rho*_peak
  1.0     2.00            8.00       0.018   (current -- fails)
  0.85    1.70            4.18       0.124
  0.70    1.40            1.92       0.383   (headline -- mirrors the R1.5
                                             "closer wells" instance)

Two configs, 3 seeds each:
  'manuscript'  DiagTestFunctions (sin, K=300) + the manuscript's own biased
                squared-mean loss + Adam -- i.e. nothing changed from
                run_experiment_kramers.py except the value of a.
  'mixed'       DiagMixedTestFunctions: 75% sin plane waves + 25% Gaussian
                wavelets localised on the transition-state region, K=300,
                SAME biased loss, same optimizers and schedule. Only the
                test-function activation class changes.

Usage:
    python run_experiment_kramers_closer.py --selftest
    python run_experiment_kramers_closer.py --a-well 0.7 [--epochs N] [--seeds S]
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm

from common_wanpm import SteadyPushforward, gaussian_base_sampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTDIR = os.path.join(os.path.dirname(__file__), 'results_kramers_closer')
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# physics -- identical to run_experiment_kramers.py except U is parametrised
# ---------------------------------------------------------------------------
GAMMA = 1.0
T_TARGET = 0.25
D_BASE = 8
K_TEST = 300
BATCH_M = 1000
LR_GEN = 1e-2
LR_TEST = 1e-2
ADV_EVERY = 10
ADV_STEPS = 1          # matches run_experiment_kramers.py exactly (1 ascent step / 10 epochs)

T_START = 1.0
WARMUP_EPOCHS, ANNEAL_EPOCHS, FINETUNE_EPOCHS = 3000, 20000, 12000
N_EPOCHS_DEFAULT = WARMUP_EPOCHS + ANNEAL_EPOCHS + FINETUNE_EPOCHS

A_WELL = 0.7          # well half-separation: wells at x = +-A_WELL (set via set_a)


def Ufn(x):
    return (x ** 2 - A_WELL ** 2) ** 2


def Uprime(x):
    return 4 * x ** 3 - 4 * A_WELL ** 2 * x


def drift_kramers(xv):
    x, v = xv[:, 0:1], xv[:, 1:2]
    bx = v
    bv = -GAMMA * v - Uprime(x)
    return torch.cat([bx, bv], dim=1)


def T_at_epoch(epoch, n_epochs):
    """cosine T-anneal, scaled so the three phases keep their proportions
    (10% / 62.5% / 37.5%-ish) for any total epoch count."""
    warm = int(round(WARMUP_EPOCHS / N_EPOCHS_DEFAULT * n_epochs))
    anneal = int(round(ANNEAL_EPOCHS / N_EPOCHS_DEFAULT * n_epochs))
    if epoch < warm:
        return T_START
    if epoch < warm + anneal:
        progress = (epoch - warm) / max(anneal, 1)
        cos_progress = 0.5 * (1 - np.cos(np.pi * progress))
        return T_START + (T_TARGET - T_START) * cos_progress
    return T_TARGET


# ---------------------------------------------------------------------------
# true marginals (quadrature)
# ---------------------------------------------------------------------------
def _unnorm_x(x):
    return np.exp(-((x ** 2 - A_WELL ** 2) ** 2) / T_TARGET)


TRUE = {}


def _recompute_true():
    Zx, _ = quad(_unnorm_x, -4, 4, limit=400)
    frac_right, _ = quad(lambda t: _unnorm_x(t) / Zx, 0, 4, limit=400)
    # peak height (at x = +-a) and valley height (x=0) of the normalised x-marginal
    peak = _unnorm_x(A_WELL) / Zx
    valley = _unnorm_x(0.0) / Zx
    TRUE.update(Zx=Zx, frac_right=float(frac_right),
                connectivity=float(valley / peak),
                rel_barrier=float(2 * (A_WELL ** 4) / T_TARGET),
                v_std=float(np.sqrt(T_TARGET)))


def set_a(a_new):
    global A_WELL
    A_WELL = float(a_new)
    _recompute_true()


_recompute_true()


def true_x_pdf(x_grid):
    return _unnorm_x(x_grid) / TRUE['Zx']


# ---------------------------------------------------------------------------
# test functions
# ---------------------------------------------------------------------------
class DiagTestFunctions(nn.Module):
    """Plane-wave (sin) ridge test functions, general diagonal diffusion.
    Identical to run_experiment_kramers.py::DiagTestFunctions."""

    def __init__(self, n_in, K, w_scale=1.0, seed=None):
        super().__init__()
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        self.w = nn.Parameter(torch.randn(n_in, K, generator=gen) * w_scale)
        self.b = nn.Parameter(torch.rand(K, generator=gen) * 2 * np.pi)

    def phase(self, x):
        return x @ self.w + self.b

    def Lf(self, x, drift_fn, a_diag_fn):
        phase = self.phase(x)
        cos_p = torch.cos(phase)
        sin_p = torch.sin(phase)
        b_dot_w = drift_fn(x) @ self.w
        a_diag = a_diag_fn(x)
        weighted_w_sq = a_diag @ (self.w ** 2)
        return -b_dot_w * cos_p + 0.5 * weighted_w_sq * sin_p


def _phi_sin(z):
    s, c = torch.sin(z), torch.cos(z)
    return s, c, -s


def _phi_gauss(z):
    """Non-oscillatory Gaussian bump -- the R1.5 study's winning wavelet."""
    g = torch.exp(-z ** 2 / 2)
    return g, -z * g, (z ** 2 - 1) * g


def _phi_mexhat(z):
    """Mexican hat / Ricker: (1-z^2) exp(-z^2/2). Even, one internal
    zero-crossing (oscillatory in the R1.5 sense)."""
    g = torch.exp(-z ** 2 / 2)
    return (1 - z ** 2) * g, (z ** 3 - 3 * z) * g, (-z ** 4 + 6 * z ** 2 - 3) * g


def _phi_morlet(z, w0=5.0):
    """Windowed plane wave: exp(-z^2/2) cos(w0 z). Oscillatory."""
    g = torch.exp(-z ** 2 / 2)
    c, s = torch.cos(w0 * z), torch.sin(w0 * z)
    return (g * c,
            g * (-z * c - w0 * s),
            g * ((z ** 2 - 1 - w0 ** 2) * c + 2 * w0 * z * s))


def _phi_sech2(z):
    """Logistic/hyperbolic-secant bump: sech^2(z) = 1 - tanh^2(z). Heavier
    (exponential, not Gaussian) tails; matches the generator's own tanh
    nonlinearity."""
    t = torch.tanh(z)
    s2 = 1 - t ** 2                      # sech^2(z)
    phi = s2
    d1 = -2 * t * s2
    d2 = 4 * s2 - 6 * s2 ** 2             # = 4 sech^2 - 6 sech^4
    return phi, d1, d2


def _phi_herm1(z):
    """First-Hermite bump: z*exp(-z^2/2) -- ODD (antisymmetric) about its
    centre, unlike the other (even) localized activations here."""
    g = torch.exp(-z ** 2 / 2)
    return z * g, (1 - z ** 2) * g, (z ** 3 - 3 * z) * g


_PHI = {'sin': _phi_sin, 'gauss': _phi_gauss, 'mexhat': _phi_mexhat,
        'morlet': _phi_morlet, 'sech2': _phi_sech2, 'herm1': _phi_herm1}
LOCAL_KINDS = ('gauss', 'mexhat', 'morlet', 'sech2', 'herm1')  # non-plane-wave choices


class _RidgeBank(nn.Module):
    """One homogeneous bank of ridge test functions f = phi(w.x + b), with a
    general diagonal diffusion tensor a(x) = diag(a_1, ..., a_n):

        Lf = -(b_drift . w) phi'(z)  -  (1/2) (sum_i a_ii w_i^2) phi''(z),
        z  = w . x + b.

    Same sign convention as DiagTestFunctions (phi=sin, phi'=cos, phi''=-sin
    reproduces it exactly). Only phi changes between banks."""

    def __init__(self, n_in, K, kind, init_scale=4.0, centre_spread=1.0,
                 v_coupling=0.5, seed=None):
        super().__init__()
        self.kind = kind
        self._phi = _PHI[kind]
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        if kind == 'sin':
            w = torch.randn(n_in, K, generator=gen)
            b = torch.rand(K, generator=gen) * 2 * np.pi
        elif kind in LOCAL_KINDS:
            # |w1| ~ init_scale sets the wavelet WIDTH ~ 1/|w1| in x;
            # centre -b/w1 spread over [-centre_spread, centre_spread];
            # mild velocity coupling w2.
            mag = init_scale * (0.5 + torch.rand(K, generator=gen))
            sign = torch.where(torch.rand(K, generator=gen) < 0.5, -1.0, 1.0)
            w1 = mag * sign
            w2 = v_coupling * torch.randn(K, generator=gen)
            w = torch.stack([w1, w2], dim=0)
            centre = (torch.rand(K, generator=gen) * 2 - 1) * centre_spread
            b = -(w1 * centre)
        else:
            raise ValueError(kind)
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(b)

    def phase(self, x):
        return x @ self.w + self.b

    def Lf(self, x, drift_fn, a_diag_fn):
        z = self.phase(x)
        _, d1, d2 = self._phi(z)
        b_dot_w = drift_fn(x) @ self.w
        a_diag = a_diag_fn(x)
        weighted_w_sq = a_diag @ (self.w ** 2)
        return -b_dot_w * d1 - 0.5 * weighted_w_sq * d2


class DiagMixedTestFunctions(nn.Module):
    """`frac_wave` of K functions are global sin plane waves (keep the weak
    form well-posed everywhere); the rest are Gaussian wavelets localised on
    the transition-state region (supply the sensitivity that suppresses the
    spurious x=0 spike). Diagonal-diffusion Lf; the manuscript's ridge form
    throughout, only phi differs. Mirrors
    R1_5_asymmetric_muon/wavelet_testfuncs.py::MixedTestFunctions,
    adapted from isotropic a=sigma^2 I to a general diagonal a(x)."""

    def __init__(self, n_in, K, frac_wave=0.75, activation='gauss', init_scale=4.0,
                 centre_spread=1.0, v_coupling=0.5, seed=None):
        super().__init__()
        self.n_plane = int(round(frac_wave * K))
        self.n_loc = K - self.n_plane
        self.plane = _RidgeBank(n_in, self.n_plane, 'sin',
                                seed=seed) if self.n_plane else None
        self.local = _RidgeBank(n_in, self.n_loc, activation, init_scale=init_scale,
                                centre_spread=centre_spread, v_coupling=v_coupling,
                                seed=None if seed is None else seed + 1) if self.n_loc else None
        self.tag = f'mixed({activation},{frac_wave:g})'

    def Lf(self, x, drift_fn, a_diag_fn):
        parts = [m.Lf(x, drift_fn, a_diag_fn)
                 for m in (self.plane, self.local) if m is not None]
        return torch.cat(parts, dim=1)


class DiagMultiMixedTestFunctions(nn.Module):
    """`frac_wave` of K functions are global sin plane waves; the rest are
    split as evenly as possible across ALL of `local_kinds` (default: every
    localized activation tried here -- gauss/mexhat/morlet/sech2/herm1), each
    with its own sub-bank of _RidgeBank. Diagonal-diffusion Lf throughout."""

    def __init__(self, n_in, K, frac_wave=0.6,
                 local_kinds=('gauss', 'mexhat', 'morlet', 'sech2', 'herm1'),
                 init_scale=4.0, centre_spread=1.0, v_coupling=0.5, seed=None):
        super().__init__()
        self.n_plane = int(round(frac_wave * K))
        n_loc_total = K - self.n_plane
        n_kinds = len(local_kinds)
        base_n = n_loc_total // n_kinds
        rem = n_loc_total - base_n * n_kinds
        sizes = [base_n + (1 if i < rem else 0) for i in range(n_kinds)]
        self.plane = _RidgeBank(n_in, self.n_plane, 'sin', seed=seed) if self.n_plane else None
        self.locals = nn.ModuleList()
        s = seed
        for kind, k in zip(local_kinds, sizes):
            if k <= 0:
                continue
            s = None if seed is None else s + 1
            self.locals.append(_RidgeBank(n_in, k, kind, init_scale=init_scale,
                                          centre_spread=centre_spread,
                                          v_coupling=v_coupling, seed=s))
        self.tag = f"multimix({'+'.join(local_kinds)},{frac_wave:g})"
        self.sizes = dict(zip(local_kinds, sizes))

    def Lf(self, x, drift_fn, a_diag_fn):
        mods = ([self.plane] if self.plane is not None else []) + list(self.locals)
        return torch.cat([m.Lf(x, drift_fn, a_diag_fn) for m in mods], dim=1)


def diag_steady_loss(pf, tf, M, drift_fn, a_diag_fn, base_sampler, device):
    """The manuscript's own biased squared-sample-mean loss (no U-statistic)."""
    r = base_sampler(M, pf.d_base, device)
    xv = pf(r)
    Lf = tf.Lf(xv, drift_fn, a_diag_fn)
    Ehat = Lf.mean(dim=0)
    return (Ehat ** 2).mean(), xv


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
# mixed-bank hyperparameters (overridable from the CLI for tuning passes)
MIX_FRAC_WAVE = 0.75          # fraction of the K functions that are sin plane waves
MIX_ACTIVATION = 'gauss'      # local-half activation: gauss/mexhat/morlet/sech2/herm1
MIX_INIT_SCALE = 4.0          # |w1| ~ this -> wavelet width ~ 1/|w1|
MIX_CENTRE_FACTOR = 1.5       # wavelet x-centres spread over [-f*a, f*a]
MULTIMIX_KINDS = LOCAL_KINDS  # which activations the 'multimix' config splits evenly
                              # (pass --multimix-kinds sin,gauss,... to include sin
                              # as an equal partner rather than the majority plane-wave)


def make_tf(config, seed):
    if config == 'manuscript':
        return DiagTestFunctions(n_in=2, K=K_TEST, seed=seed).to(device)
    if config == 'mixed':
        return DiagMixedTestFunctions(
            2, K_TEST, frac_wave=MIX_FRAC_WAVE, activation=MIX_ACTIVATION,
            init_scale=MIX_INIT_SCALE, centre_spread=MIX_CENTRE_FACTOR * A_WELL,
            v_coupling=0.5, seed=seed).to(device)
    if config == 'multimix':
        return DiagMultiMixedTestFunctions(
            2, K_TEST, frac_wave=MIX_FRAC_WAVE, local_kinds=MULTIMIX_KINDS,
            init_scale=MIX_INIT_SCALE, centre_spread=MIX_CENTRE_FACTOR * A_WELL,
            v_coupling=0.5, seed=seed).to(device)
    raise ValueError(config)


# ---------------------------------------------------------------------------
# generator optimizer -- mirrors R1_5_asymmetric_muon/asym_muon.py
# exactly: Muon on every 2-D hidden weight matrix, its internal AdamW (at the
# manuscript's own generator lr) on all biases and the output layer.
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


def make_gen_opt(name, model, lr, head_lr=LR_GEN):
    if name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == 'Muon':
        import pytorch_optimizer as po
        hidden, rest = split_hidden_head(model)
        return po.Muon([dict(params=hidden, lr=lr, weight_decay=0.0, use_muon=True),
                        dict(params=rest, lr=head_lr, weight_decay=0.0, use_muon=False)],
                       weight_decay=0.0)
    raise ValueError(name)


def run(config, seed, n_epochs, verbose=False, optimizer='Adam', lr_gen=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    pf = SteadyPushforward(d_base=D_BASE, n_out=2, hidden=(32, 32)).to(device)
    tf = make_tf(config, seed)
    gen_opt = make_gen_opt(optimizer, pf, lr_gen if lr_gen is not None else LR_GEN)
    test_opt = torch.optim.SGD(tf.parameters(), lr=LR_TEST)

    losses = np.empty(n_epochs)
    t0 = time.time()
    diverged = False
    for ep in range(n_epochs):
        cur_T = float(T_at_epoch(ep, n_epochs))
        cur_avv = float(2 * GAMMA * cur_T)

        def a_diag_fn(xv, _avv=cur_avv):
            return torch.tensor([0.0, _avv], dtype=torch.float32,
                                device=xv.device).unsqueeze(0).expand(xv.shape[0], -1)

        def loss_fn():
            return diag_steady_loss(pf, tf, BATCH_M, drift_kramers, a_diag_fn,
                                    gaussian_base_sampler, device)

        if ep % ADV_EVERY == 0 and ep > 0:
            for _ in range(ADV_STEPS):
                l, _ = loss_fn()
                test_opt.zero_grad(); (-l).backward(); test_opt.step()
        l, _ = loss_fn()
        gen_opt.zero_grad(); l.backward(); gen_opt.step()
        lv = l.item()
        losses[ep] = lv
        if not np.isfinite(lv):
            diverged = True
            break
        if verbose and (ep % 4000 == 0 or ep == n_epochs - 1):
            print(f"    epoch {ep:6d}/{n_epochs}  T={cur_T:.3f}  loss={lv:.4e}", flush=True)
    dt = time.time() - t0

    rec = {'config': config, 'seed': seed, 'n_epochs': n_epochs,
           'train_time_s': dt, 'diverged': diverged}
    xv = None
    if diverged:
        rec.update({k: float('nan') for k in
                    ('hellinger', 'tv', 'frac_right', 'ts_ratio', 'ts_ratio_true',
                     'v_std', 'x_std')})
    else:
        with torch.no_grad():
            xv = pf(torch.randn(200000, D_BASE, device=device)).cpu().numpy()
        rec.update(evaluate(xv))
    return rec, xv, losses


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
GRID_LO, GRID_HI, N_BINS = -2.5, 2.5, 250


def evaluate(xv):
    x, v = xv[:, 0], xv[:, 1]
    edges = np.linspace(GRID_LO, GRID_HI, N_BINS + 1)
    c = 0.5 * (edges[:-1] + edges[1:])
    dx = edges[1] - edges[0]
    ph, _ = np.histogram(x, bins=edges, density=True)
    frac_in = float(((x >= GRID_LO) & (x <= GRID_HI)).mean())
    ph = ph * frac_in
    pt = true_x_pdf(c)
    pt = pt / (pt.sum() * dx)

    # transition-state ratio: learned density at x=0 relative to at the wells
    def _band(arr, lo, hi):
        m = (arr >= lo) & (arr < hi)
        return m.mean() / (hi - lo)          # crude density estimate
    half = 0.15
    dens0 = _band(x, -half, half)
    densL = _band(x, -A_WELL - half, -A_WELL + half)
    densR = _band(x, A_WELL - half, A_WELL + half)
    ts_ratio = float(dens0 / max(0.5 * (densL + densR), 1e-9))
    t0 = _unnorm_x(0.0)
    tw = _unnorm_x(A_WELL)
    ts_ratio_true = float(t0 / tw)

    return {
        'hellinger': float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(ph * pt)) * dx))),
        'tv': float(0.5 * np.sum(np.abs(ph - pt)) * dx),
        'frac_right': float((x > 0).mean()),
        'frac_right_true': TRUE['frac_right'],
        'ts_ratio': ts_ratio,
        'ts_ratio_true': ts_ratio_true,
        'v_std': float(v.std()),
        'v_std_true': TRUE['v_std'],
        'x_std': float(x.std()),
        'frac_in_window': frac_in,
    }


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def _selftest():
    torch.set_default_dtype(torch.float64)
    print("[selftest] phi' / phi'' vs autograd (float64):")
    z = torch.randn(300, dtype=torch.float64, requires_grad=True)
    for name, fn in _PHI.items():
        f, d1, d2 = fn(z)
        g1 = torch.autograd.grad(f.sum(), z, create_graph=True)[0]
        g2 = torch.autograd.grad(g1.sum(), z)[0]
        print(f"  {name:6s} max|phi'-ad|={float((d1-g1).abs().max()):.2e}  "
              f"max|phi''-ad|={float((d2-g2).abs().max()):.2e}")
        assert (d1 - g1).abs().max() < 1e-9 and (d2 - g2).abs().max() < 1e-9

    # DiagMixedTestFunctions with frac_wave=1 must reproduce DiagTestFunctions
    torch.manual_seed(3)
    ref = DiagTestFunctions(2, 32, seed=3).double()
    mix = DiagMixedTestFunctions(2, 32, frac_wave=1.0, seed=3).double()
    xv = torch.randn(64, 2, dtype=torch.float64)
    drift = lambda t: torch.cat([t[:, 1:2], -t[:, 1:2] - (4 * t[:, 0:1] ** 3 - 4 * 0.49 * t[:, 0:1])], dim=1)
    adiag = lambda t: torch.tensor([0.0, 0.5], dtype=torch.float64).unsqueeze(0).expand(t.shape[0], -1)
    d = (ref.Lf(xv, drift, adiag) - mix.Lf(xv, drift, adiag)).abs().max().item()
    print(f"[selftest] mixed(frac_wave=1) vs DiagTestFunctions: max|dLf|={d:.2e}")
    assert d < 1e-9
    torch.set_default_dtype(torch.float32)
    print("[selftest] OK")


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def make_figure(samples, tag):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    xg = np.linspace(GRID_LO, GRID_HI, 500)
    pt = true_x_pdf(xg)

    ax = axes[0]
    if 'manuscript_s0' in samples:
        xv = samples['manuscript_s0']
        ax.scatter(xv[::20, 0], xv[::20, 1], s=2, alpha=0.15, color='#b23a48')
    ax.plot([-A_WELL, A_WELL], [0, 0], 'kx', ms=12, mew=3, label='true minima')
    ax.set_xlim(GRID_LO, GRID_HI); ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel('x'); ax.set_ylabel('v')
    ax.set_title(f'Phase-space samples (manuscript, seed 0)\n'
                 f'2a={2*A_WELL:.2f}, connectivity {TRUE["connectivity"]:.3f}')
    ax.legend(fontsize=8)

    ax = axes[1]
    for k, col, lab in (('manuscript_s0', '#b23a48', 'sin (manuscript)'),
                        ('mixed_s0', '#2c6e49', '75/25 mix-gauss')):
        if k in samples:
            ax.hist(samples[k][:, 0], bins=150, range=(GRID_LO, GRID_HI),
                    density=True, color=col, alpha=0.5, label=f'learned: {lab}')
    ax.plot(xg, pt, 'k-', lw=2.2, label=r'true $\rho_x\propto e^{-U/T}$')
    ax.axvline(0, color='0.5', ls=':', lw=1)
    ax.set_xlabel('x'); ax.set_ylabel('density')
    ax.set_title('x-marginal (double-well Boltzmann)')
    ax.legend(fontsize=8.5)

    ax = axes[2]
    vs = TRUE['v_std']
    for k, col, lab in (('manuscript_s0', '#b23a48', 'manuscript'),
                        ('mixed_s0', '#2c6e49', 'mixed')):
        if k in samples:
            ax.hist(samples[k][:, 1], bins=150, range=(-3 * vs, 3 * vs),
                    density=True, color=col, alpha=0.5, label=f'learned v: {lab}')
    vg = np.linspace(-3 * vs, 3 * vs, 300)
    ax.plot(vg, norm.pdf(vg, 0, vs), 'k-', lw=2.2, label=r'true $N(0,T)$')
    ax.set_xlabel('v'); ax.set_ylabel('density')
    ax.set_title('v-marginal (Maxwell-Boltzmann)')
    ax.legend(fontsize=8.5)

    fig.suptitle(f'Kramers metastable, closer wells: $U(x)=(x^2-{A_WELL}^2)^2$, '
                 f'$\\gamma$={GAMMA}, $T$={T_TARGET}  '
                 f'(non-gradient in $(x,v)$; $2\\Delta U/T$={TRUE["rel_barrier"]:.1f})',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    p = os.path.join(OUTDIR, f'kramers_closer_{tag}.png')
    plt.savefig(p, dpi=150); plt.close()
    print(f"  wrote {p}")


# ---------------------------------------------------------------------------
def main():
    global MIX_FRAC_WAVE, MIX_CENTRE_FACTOR, MIX_INIT_SCALE, MIX_ACTIVATION, MULTIMIX_KINDS, K_TEST, BATCH_M
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--a-well', type=float, default=A_WELL)
    ap.add_argument('--epochs', type=int, default=N_EPOCHS_DEFAULT)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--configs', default='manuscript,mixed')
    ap.add_argument('--k-test', type=int, default=K_TEST,
                    help='total number of test functions K (split per --frac-wave / '
                         '--multimix-kinds as usual)')
    ap.add_argument('--batch-m', type=int, default=BATCH_M,
                    help='collocation points M per gradient step')
    ap.add_argument('--frac-wave', type=float, default=MIX_FRAC_WAVE,
                    help='fraction of the bank that is sin plane waves (mixed config). '
                         'Set to 0 for multimix so sin is just another equal partner.')
    ap.add_argument('--activation', default=MIX_ACTIVATION, choices=list(LOCAL_KINDS),
                    help='wavelet-half activation shape')
    ap.add_argument('--multimix-kinds', default=','.join(LOCAL_KINDS),
                    help="comma-separated activations the 'multimix' config splits "
                         "evenly, e.g. 'sin,gauss,mexhat,morlet,sech2,herm1' to give "
                         "sin equal weight instead of the default majority plane-wave.")
    ap.add_argument('--centre-factor', type=float, default=MIX_CENTRE_FACTOR,
                    help='wavelet x-centres spread over [-f*a, f*a]')
    ap.add_argument('--init-scale', type=float, default=MIX_INIT_SCALE)
    ap.add_argument('--optimizer', default='Adam', choices=['Adam', 'Muon'])
    ap.add_argument('--lr-gen', type=float, default=None,
                    help='generator lr (Muon lr on hidden matrices; Adam lr otherwise). '
                         'Default: LR_GEN (1e-2) for Adam, same for Muon unless set --'
                         ' Muon typically needs ~10-50x that.')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--seed-start', type=int, default=0,
                    help='first seed value (seeds run are range(start, start+seeds))')
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    K_TEST = args.k_test
    BATCH_M = args.batch_m
    MIX_FRAC_WAVE = args.frac_wave
    MIX_ACTIVATION = args.activation
    MIX_CENTRE_FACTOR = args.centre_factor
    MIX_INIT_SCALE = args.init_scale
    MULTIMIX_KINDS = tuple(k.strip() for k in args.multimix_kinds.split(','))
    for k in MULTIMIX_KINDS:
        if k not in _PHI:
            raise ValueError(f"unknown activation {k!r} in --multimix-kinds")

    set_a(args.a_well)
    tag = args.tag or f"a{A_WELL:g}"
    configs = args.configs.split(',')

    print(f"device: {device}")
    print(f"KRAMERS CLOSER WELLS: U(x) = (x^2 - {A_WELL}^2)^2,  gamma={GAMMA}, T={T_TARGET}")
    print(f"  well separation {2*A_WELL:.2f}  (vs 2.00 current)")
    print(f"  relative barrier 2*dU/T = {TRUE['rel_barrier']:.2f}   "
          f"connectivity rho*(0)/rho*_peak = {TRUE['connectivity']:.4f}")
    print(f"  true frac(x>0) = {TRUE['frac_right']:.4f} (0.5 by symmetry)   "
          f"true v std = {TRUE['v_std']:.4f}")
    print(f"  epochs={args.epochs}  seeds={args.seeds}  configs={configs}"
          f"  optimizer={args.optimizer}"
          f"{f' lr_gen={args.lr_gen:g}' if args.lr_gen is not None else ''}")
    print()

    recs, samples, loss_curves = [], {}, {}
    for config in configs:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            print(f"[{config}] a={A_WELL:g} seed={seed} ...", flush=True)
            rec, xv, losses = run(config, seed, args.epochs, verbose=True,
                                  optimizer=args.optimizer, lr_gen=args.lr_gen)
            rec['optimizer'] = args.optimizer
            rec['lr_gen'] = args.lr_gen
            recs.append(rec)
            if rec['diverged']:
                print("    DIVERGED", flush=True)
            else:
                print(f"    frac_right={rec['frac_right']:.4f}  H={rec['hellinger']:.4f}  "
                      f"TV={rec['tv']:.4f}  ts_ratio={rec['ts_ratio']:.3f} "
                      f"(true {rec['ts_ratio_true']:.3f})  v_std={rec['v_std']:.4f}  "
                      f"({rec['train_time_s']:.0f}s)", flush=True)
            if xv is not None and seed == args.seed_start:
                samples[f"{config}_s{seed}"] = xv.astype(np.float32)
            loss_curves[f"{config}_s{seed}"] = losses.astype(np.float32)
            with open(os.path.join(OUTDIR, f'results_{tag}.json'), 'w') as fh:
                json.dump({'a_well': A_WELL, 'gamma': GAMMA, 'T': T_TARGET,
                           'connectivity': TRUE['connectivity'],
                           'rel_barrier': TRUE['rel_barrier'],
                           'true_frac_right': TRUE['frac_right'],
                           'true_v_std': TRUE['v_std'], 'records': recs}, fh, indent=2)
            np.savez_compressed(os.path.join(OUTDIR, f'samples_{tag}.npz'), **samples)
            np.savez_compressed(os.path.join(OUTDIR, f'loss_curves_{tag}.npz'), **loss_curves)

    _summary(recs, tag)
    make_figure(samples, tag)


def _summary(recs, tag):
    import collections
    g = collections.defaultdict(list)
    for r in recs:
        if not r['diverged']:
            g[r['config']].append(r)
    print()
    print("=" * 78)
    print(f"a={A_WELL:g}  connectivity={TRUE['connectivity']:.4f}  "
          f"2dU/T={TRUE['rel_barrier']:.2f}")
    print(f"{'config':12s} {'H':>16s} {'TV':>16s} {'ts_ratio':>16s} {'v_std':>12s}")
    for k, v in g.items():
        H = np.array([r['hellinger'] for r in v])
        TVv = np.array([r['tv'] for r in v])
        tr = np.array([r['ts_ratio'] for r in v])
        vs = np.array([r['v_std'] for r in v])
        print(f"{k:12s} {H.mean():7.4f}+/-{H.std():<7.4f} "
              f"{TVv.mean():7.4f}+/-{TVv.std():<7.4f} "
              f"{tr.mean():7.3f}+/-{tr.std():<7.3f} {vs.mean():6.4f}+/-{vs.std():<5.4f}")
    print(f"{'true':12s} {0.0:16.4f} {0.0:16.4f} "
          f"{recs[0].get('ts_ratio_true', float('nan')):16.3f} {TRUE['v_std']:12.4f}")


if __name__ == '__main__':
    main()
