"""
Phase 1: re-run main.tex's low-dimensional STEADY-STATE examples with Muon.

Covers Sec. 4.1 (1D Ornstein-Uhlenbeck, frozen test functions), Sec. 4.3
(2D double peak) and Sec. 4.5 (2D ring).  Sec. 4.2 is already covered by
run_experiment.py / simple_compare.py.  The time-dependent examples
(4.4, 4.7, 4.8, 4.9, 4.10) are phase 2.

Each example is run with
    - Adam at the manuscript's own learning rate          (baseline)
    - Muon over a learning-rate sweep                     (Muon's optimum sits
      ~100x above Adam's, so it cannot be plugged in blind)
and scored against the analytical solution.

Muon parameter routing on a multi-hidden-layer network: every 2-D hidden
weight matrix goes to Muon; biases and the output layer go to the internal
AdamW at the manuscript's own generator learning rate.  Sec. 4.2 had a single
hidden matrix, so these deeper networks are the first real test of whether
that convention generalises.

NOTATION NOTE.  common_wanpm's Lf is  -b.w cos(phase) + (1/2) a |w|^2 sin(phase),
i.e. its `sigma2_fn` returns the diffusion coefficient a, and the stationary
density is rho ~ exp(-2V/a).  We therefore set a = sigma^2 so that rho_true
matches the manuscript's own formula rho ~ exp(-2V/sigma^2) in 4.3 and 4.5.
For 4.1 the manuscript's eq:1d_fpe, its simplified form and its stated answer
N(2,1) are mutually inconsistent; we target the stated answer, which requires
a = 2 with V = (x-2)^2/2.

Usage:
    python lowdim_muon.py --example ou1d|dp2d|ring2d|all --mode sweep
    python lowdim_muon.py --example ... --mode plot
"""
import os, json, time, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad

import run_experiment as R
import muon_controls as MC
from common_wanpm import (SteadyPushforward, SteadyTestFunctions, steady_loss,
                          gaussian_base_sampler, uniform_base_sampler)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'results_lowdim')
os.makedirs(OUTDIR, exist_ok=True)
device = R.device


# ---------------------------------------------------------------------------
# Problem definitions
# ---------------------------------------------------------------------------
class Example:
    def __init__(self, key, title, n_dim, d_base, hidden, K, M, epochs,
                 adam_lr, drift, a_coef, sampler, adversarial=True,
                 anneal_from=None, adv_every=10, adv_steps=5, lr_test=1e-2):
        self.__dict__.update(locals()); del self.self


def _ou1d_drift(x):
    return -(x - 2.0)


def _dp2d_drift(x):
    x1, x2 = x[:, 0:1], x[:, 1:2]
    p = (x1 + 1) ** 2 + (x2 + 1) ** 2
    m = (x1 - 1) ** 2 + (x2 - 1) ** 2
    b1 = -2 * (x1 - 1) * p - 2 * (x1 + 1) * m
    b2 = -2 * (x2 - 1) * p - 2 * (x2 + 1) * m
    return torch.cat([b1, b2], dim=1)


def _ring_drift(x, r0=2.0, omega=2.0):
    x1, x2 = x[:, 0:1], x[:, 1:2]
    r2 = x1 ** 2 + x2 ** 2
    return torch.cat([-2 * x1 * (r2 - r0 ** 2) - omega * x2,
                      -2 * x2 * (r2 - r0 ** 2) + omega * x1], dim=1)


EXAMPLES = {
    # Sec. 4.1 -- (1,10,10,1), uniform base, test functions FROZEN
    'ou1d': Example('ou1d', 'Sec. 4.1  1D Ornstein-Uhlenbeck', 1, 1, (10, 10),
                    100, 1000, 10000, 1e-2, _ou1d_drift, 2.0,
                    uniform_base_sampler(0.0, 1.0), adversarial=False),
    # Sec. 4.3 -- (16,32,32,2), sigma=0.4.  Relative barrier 2*4/0.16 = 50;
    # stage-1 R1_4 showed this collapses without a curriculum, so anneal.
    'dp2d': Example('dp2d', 'Sec. 4.3  2D double peak', 2, 16, (32, 32),
                    200, 1000, 100000, 1e-2, _dp2d_drift, 0.16,
                    gaussian_base_sampler, anneal_from=1.5),
    # Sec. 4.5 -- (4,64,2), sigma=1
    'ring2d': Example('ring2d', 'Sec. 4.5  2D ring', 2, 4, (64,),
                      200, 1000, 20000, 1e-2, _ring_drift, 1.0,
                      gaussian_base_sampler),
}

MUON_LRS = [3e-2, 1e-1, 2e-1, 5e-1]
RING_DBASE = [4, 16, 64]     # author asked for a higher base dimension


# ---------------------------------------------------------------------------
def make_opt(name, model, lr, adam_lr):
    if name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == 'Muon':
        hidden, rest = MC.split_hidden_head(model)
        import pytorch_optimizer as po
        return po.Muon([dict(params=hidden, lr=lr, weight_decay=0.0, use_muon=True),
                        dict(params=rest, lr=adam_lr, weight_decay=0.0, use_muon=False)],
                       weight_decay=0.0)
    raise ValueError(name)


def sigma2_schedule(ex, epoch):
    if ex.anneal_from is None:
        return ex.a_coef
    w = int(0.10 * ex.epochs); a = w + int(0.55 * ex.epochs)
    if epoch < w:
        return ex.anneal_from
    if epoch < a:
        p = 0.5 * (1 - np.cos(np.pi * (epoch - w) / max(1, a - w)))
        return ex.anneal_from + (ex.a_coef - ex.anneal_from) * p
    return ex.a_coef


def train(ex, opt_name, lr, seed=0, d_base=None):
    d_base = d_base or ex.d_base
    torch.manual_seed(seed); np.random.seed(seed)
    pf = SteadyPushforward(d_base=d_base, n_out=ex.n_dim, hidden=ex.hidden).to(device)
    tf = SteadyTestFunctions(n_in=ex.n_dim, K=ex.K).to(device)
    gen_opt = make_opt(opt_name, pf, lr, ex.adam_lr)
    test_opt = torch.optim.SGD(tf.parameters(), lr=ex.lr_test)

    cur = [ex.a_coef]

    def sigma2_fn(x):
        return cur[0] * torch.ones(x.shape[0], 1, device=x.device)

    def loss_fn():
        return steady_loss(pf, tf, M=ex.M, drift_fn=ex.drift, sigma2_fn=sigma2_fn,
                           base_sampler=ex.sampler, device=device)

    losses = np.empty(ex.epochs); diverged = False
    t0 = time.time()
    for epoch in range(ex.epochs):
        cur[0] = sigma2_schedule(ex, epoch)
        if ex.adversarial and epoch % ex.adv_every == 0 and epoch > 0:
            for _ in range(ex.adv_steps):
                l, _ = loss_fn()
                test_opt.zero_grad(); (-l).backward(); test_opt.step()
        l, _ = loss_fn()
        gen_opt.zero_grad(); l.backward(); gen_opt.step()
        losses[epoch] = l.item()
        if not np.isfinite(losses[epoch]):
            diverged = True; losses[epoch:] = np.nan; break
    dt = time.time() - t0

    with torch.no_grad():
        x = pf(torch.randn(200000, d_base, device=device) if ex.sampler is gaussian_base_sampler
               else ex.sampler(200000, d_base, device)).cpu().numpy()
    return x, losses, dt, diverged


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics_1d(ex, x):
    x = x.ravel()
    lo, hi, nb = -1.0, 5.0, 240
    edges = np.linspace(lo, hi, nb + 1); c = 0.5 * (edges[:-1] + edges[1:])
    dx = edges[1] - edges[0]
    pt = np.exp(-(c - 2.0) ** 2 / 2.0) / np.sqrt(2 * np.pi)
    pt = pt / (pt.sum() * dx)
    ph, _ = np.histogram(x, bins=edges, density=True)
    ph = ph * float(((x >= lo) & (x <= hi)).mean())
    return {'hellinger': float(np.sqrt(max(0., 1 - np.sum(np.sqrt(ph * pt)) * dx))),
            'mean': float(x.mean()), 'std': float(x.std()),
            'mean_true': 2.0, 'std_true': 1.0}


def _true2d(ex, X, Y):
    if ex.key == 'dp2d':
        V = ((X - 1) ** 2 + (Y - 1) ** 2) * ((X + 1) ** 2 + (Y + 1) ** 2)
    else:
        V = 0.25 * (X ** 2 + Y ** 2 - 4.0) ** 2
    p = np.exp(-2 * V / ex.a_coef)
    return p / p.sum()


def metrics_2d(ex, x):
    lim = 2.5 if ex.key == 'dp2d' else 3.5
    nb = 120
    edges = np.linspace(-lim, lim, nb + 1); c = 0.5 * (edges[:-1] + edges[1:])
    H, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=[edges, edges])
    ph = H / max(1, H.sum())
    X, Y = np.meshgrid(c, c, indexing='ij')
    pt = _true2d(ex, X, Y)
    out = {'hellinger': float(np.sqrt(max(0., 1 - np.sum(np.sqrt(ph * pt)))))}
    if ex.key == 'dp2d':
        s = (x[:, 0] + x[:, 1]) / np.sqrt(2)
        out['frac_upper'] = float((s > 0).mean())
        out['well_balance_err'] = abs(out['frac_upper'] - 0.5)
        out['saddle_frac'] = float((np.hypot(x[:, 0], x[:, 1]) < 0.3).mean())
    else:
        r = np.hypot(x[:, 0], x[:, 1])
        # true radial law: p(r) ~ r exp(-(r^2-r0^2)^2 / (2 sigma^2))
        rg = np.linspace(0, 4.5, 3000)
        w = rg * np.exp(-(rg ** 2 - 4.0) ** 2 / (2 * ex.a_coef)); w /= w.sum()
        rm = float((rg * w).sum()); rs = float(np.sqrt(((rg - rm) ** 2 * w).sum()))
        out.update({'r_mean': float(r.mean()), 'r_std': float(r.std()),
                    'r_mean_true': rm, 'r_std_true': rs,
                    'r_std_ratio': float(r.std() / rs)})
        th = np.arctan2(x[:, 1], x[:, 0])
        hh, _ = np.histogram(th, bins=36, range=(-np.pi, np.pi))
        hh = hh / hh.sum()
        out['angular_tv'] = float(0.5 * np.abs(hh - 1 / 36).sum())
    return out


def score(ex, x):
    return metrics_1d(ex, x) if ex.n_dim == 1 else metrics_2d(ex, x)


# ---------------------------------------------------------------------------
def run_example(key):
    ex = EXAMPLES[key]
    print(f"\n=== {ex.title} ===")
    print(f"    arch ({ex.d_base},{','.join(map(str, ex.hidden))},{ex.n_dim})  "
          f"K={ex.K} M={ex.M} epochs={ex.epochs} a={ex.a_coef}"
          f"{'  [sigma-annealed]' if ex.anneal_from else ''}"
          f"{'' if ex.adversarial else '  [test fns frozen]'}")
    recs, samples = [], {}

    combos = [('Adam', ex.adam_lr, ex.d_base)]
    combos += [('Muon', lr, ex.d_base) for lr in MUON_LRS]
    if key == 'ring2d':
        combos += [('Muon', lr, d) for d in RING_DBASE[1:] for lr in MUON_LRS]
        combos += [('Adam', ex.adam_lr, d) for d in RING_DBASE[1:]]

    for opt, lr, d in combos:
        tag = f"{opt} lr={lr:g}" + (f" d={d}" if d != ex.d_base else "")
        print(f"  {tag:28s} ...", end='', flush=True)
        x, losses, dt, div = train(ex, opt, lr, d_base=d)
        if div:
            print(f"  DIVERGED ({dt:.0f}s)", flush=True)
            recs.append(dict(example=key, opt=opt, lr=lr, d_base=d,
                             diverged=True, train_time_s=dt))
        else:
            m = score(ex, x)
            extra = (f" r_std_ratio={m['r_std_ratio']:.3f}" if key == 'ring2d'
                     else (f" bal={m['well_balance_err']:.3f}" if key == 'dp2d'
                           else f" mean={m['mean']:.3f} std={m['std']:.3f}"))
            print(f"  H={m['hellinger']:.4f}{extra}  ({dt:.0f}s)", flush=True)
            recs.append(dict(example=key, opt=opt, lr=lr, d_base=d,
                             diverged=False, train_time_s=dt, **m))
            samples[tag] = x.astype(np.float32)
        with open(os.path.join(OUTDIR, f'{key}.json'), 'w') as fh:
            json.dump(recs, fh, indent=2)
    np.savez_compressed(os.path.join(OUTDIR, f'{key}_samples.npz'), **samples)
    return recs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--example', default='all',
                    choices=['ou1d', 'dp2d', 'ring2d', 'all'])
    args = ap.parse_args()
    print(f"device: {device}")
    keys = ['ou1d', 'ring2d', 'dp2d'] if args.example == 'all' else [args.example]
    for k in keys:
        run_example(k)
    print("\ndone")
