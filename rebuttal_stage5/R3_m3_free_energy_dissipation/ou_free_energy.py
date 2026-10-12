"""
Free-energy dissipation along the learned {rho_t} (Reviewer 3, minor comment 3), on the
1-D time-dependent Ornstein-Uhlenbeck problem of manuscript Sec. 4.9:

    dX = -theta (X - mu) dt + sigma dW,   theta=1, mu=0, sigma=1, T=1,   rho_0 = N(3, 0.5^2).

With V = theta (x - mu)^2 / 2 and D = sigma^2/2,

    F[rho] = int V rho dx + D int rho log rho dx,       F[rho_t] - F_inf = D * KL(rho_t || rho_inf),

and rho_t = N(m(t), v(t)) exactly, so F is available in closed form:

    F(t) = (theta/2) (v + (m - mu)^2) - (D/2) log(2 pi e v),     F_inf = -D log sqrt(2 pi D / theta),

decaying to F_inf at rate 2 theta (log-Sobolev). The learned F is estimated from samples of the
pushforward: int V rho is a sample mean, int rho log rho a histogram plug-in estimate (Miller-Madow
corrected, O(h^2) binning bias removed by Richardson extrapolation in the bin width) -- the same
estimator as transient/free_energy.py. It is checked on exact Gaussian samples first.

    python ou_free_energy.py                 # trains, saves results/ou_free_energy.{pt,json,npz,png}
    python ou_free_energy.py --eval-only     # re-evaluates the saved model
"""
import sys, os, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', 'rebuttal_stage1')))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common_wanpm import (TimeDepPushforward, TimeDepTestFunctions, time_dep_loss,
                          gaussian_base_sampler, train_adversarial)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTDIR = os.path.join(HERE, 'results')
os.makedirs(OUTDIR, exist_ok=True)

THETA, MU, SIGMA, T = 1.0, 0.0, 1.0, 1.0
X0_MEAN, X0_STD = 3.0, 0.5
D = SIGMA ** 2 / 2
F_INF = -D * 0.5 * np.log(2 * np.pi * D / THETA)


def drift_fn(t, x):
    return -THETA * (x - MU)


def sigma2_fn(t, x):
    return (SIGMA ** 2) * torch.ones(x.shape[0], 1, device=x.device)


def rho0_sampler(n, device):
    return X0_MEAN + X0_STD * torch.randn(n, 1, device=device)


def analytical_mv(t):
    m = MU + (X0_MEAN - MU) * np.exp(-THETA * t)
    v = X0_STD ** 2 * np.exp(-2 * THETA * t) + SIGMA ** 2 / (2 * THETA) * (1 - np.exp(-2 * THETA * t))
    return m, v


def analytical_F(t):
    m, v = analytical_mv(t)
    return 0.5 * THETA * (v + (m - MU) ** 2) - 0.5 * D * np.log(2 * np.pi * np.e * v)


# ---------------------------------------------------------------------------
# sample-based estimator (1-D version of transient/free_energy.py)
# ---------------------------------------------------------------------------
def _neg_entropy(counts, h):
    c = counts[counts > 0].double()
    N = c.sum()
    p = c / N
    # plug-in int rho log rho, minus the Miller-Madow bias (occupied bins - 1) / 2N
    return (p * torch.log(p / h)).sum().item() - (len(c) - 1) / (2 * N.item())


def free_energy_from_sampler(draw, n, lo=-5.0, hi=9.0, h=0.02, chunk=1_000_000):
    """draw(k) -> (k, 1) samples on device."""
    nb = 2 * int(round((hi - lo) / (2 * h)))
    h = (hi - lo) / nb
    counts = torch.zeros(nb, dtype=torch.long, device=device)
    s1 = s2 = 0.0
    n_out = 0
    for _ in range(n // chunk):
        x = draw(chunk)[:, 0].double()
        s1 += x.sum().item(); s2 += (x ** 2).sum().item()
        idx = torch.floor((x - lo) / h).long()
        ok = (idx >= 0) & (idx < nb)
        n_out += int((~ok).sum())
        counts += torch.bincount(idx[ok], minlength=nb)
    N = (n // chunk) * chunk
    mean = s1 / N
    var = s2 / N - mean ** 2
    EV = 0.5 * THETA * (var + (mean - MU) ** 2)
    s_h = _neg_entropy(counts, h)
    s_2h = _neg_entropy(counts.view(nb // 2, 2).sum(1), 2 * h)
    neg_ent = (4 * s_h - s_2h) / 3
    return dict(mean=mean, var=var, EV=EV, neg_entropy=neg_ent, F=EV + D * neg_ent, frac_outside=n_out / N)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--plot-only', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=20000)
    ap.add_argument('--K', type=int, default=100)
    ap.add_argument('--M', type=int, default=1000)
    ap.add_argument('--M0', type=int, default=500)
    ap.add_argument('--d-base', type=int, default=2)
    ap.add_argument('--hidden', type=int, nargs='+', default=[32, 32])
    ap.add_argument('--lr-gen', type=float, default=1e-3)
    ap.add_argument('--lr-test', type=float, default=1e-2)
    ap.add_argument('--n', type=int, default=4_000_000, help='samples per time for the estimate')
    ap.add_argument('--n-times', type=int, default=101)
    ap.add_argument('--tag', default='ou_free_energy')
    args = ap.parse_args()
    if args.plot_only:
        plot(args.tag); return

    pt_path = os.path.join(OUTDIR, f'{args.tag}.pt')
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.eval_only:
        ck = torch.load(pt_path, map_location=device)
        a, train_time, loss_log = ck['args'], ck['train_time_s'], ck['loss']
    else:
        a = vars(args)
    pf = TimeDepPushforward(1, a['d_base'], tuple(a['hidden']), power=0.5).to(device)
    if args.eval_only:
        pf.load_state_dict(ck['pf'])
    else:
        tf = TimeDepTestFunctions(1, args.K).to(device)

        def loss_fn():
            return time_dep_loss(pf, tf, M=args.M, M0=args.M0, MT=args.M0, T=T, rho0_sampler=rho0_sampler,
                                 drift_fn=drift_fn, sigma2_fn=sigma2_fn,
                                 base_sampler=gaussian_base_sampler, device=device)
        t0 = time.time()
        loss_log = train_adversarial(loss_fn, pf.parameters(), tf.parameters(), n_epochs=args.epochs,
                                     lr_gen=args.lr_gen, lr_test=args.lr_test, log_every=2000)
        train_time = time.time() - t0
        print(f'training time: {train_time:.0f}s', flush=True)
        torch.save({'pf': pf.state_dict(), 'args': a, 'train_time_s': train_time, 'loss': loss_log}, pt_path)
    pf.eval()

    ts = np.linspace(0.0, T, args.n_times)
    F_true = analytical_F(ts)

    # estimator check: exact Gaussian samples through the same estimator
    est_err = 0.0
    for t in ts[::10]:
        m, v = analytical_mv(t)
        r = free_energy_from_sampler(lambda k: m + np.sqrt(v) * torch.randn(k, 1, device=device), args.n)
        est_err = max(est_err, abs(r['F'] - analytical_F(t)))
    print(f'estimator check on exact Gaussian samples: max |F_est - F_exact| = {est_err:.1e}', flush=True)

    @torch.no_grad()
    def draw_at(t):
        return lambda k: pf(t * torch.ones(k, 1, device=device), rho0_sampler(k, device),
                            torch.randn(k, a['d_base'], device=device))
    rows = [free_energy_from_sampler(draw_at(float(t)), args.n) for t in ts]
    F = np.array([r['F'] for r in rows])
    m_true, v_true = analytical_mv(ts)
    mean_err = float(np.abs(np.array([r['mean'] for r in rows]) - m_true).max())
    var_err = float(np.abs(np.array([r['var'] for r in rows]) - v_true).max())
    dF = np.diff(F)
    k = ts >= 0.0
    rate = float(-np.polyfit(ts[k], np.log(F[k] - F_INF), 1)[0])
    print(f'WANPM: F(0)={F[0]:.4f} (exact {F_true[0]:.4f})  F(T)={F[-1]:.4f} (exact {F_true[-1]:.4f})  '
          f'F_inf={F_INF:.4f}\n  max |F - F_exact| = {np.abs(F - F_true).max():.4f}  '
          f'largest increase between consecutive times = {max(dF.max(), 0.0):.1e}  '
          f'fitted decay rate of F - F_inf = {rate:.3f}\n'
          f'  max |mean err| = {mean_err:.4f}  max |var err| = {var_err:.4f}', flush=True)

    np.savez(os.path.join(OUTDIR, f'{args.tag}.npz'), t=ts, F=F, F_true=F_true,
             mean=np.array([r['mean'] for r in rows]), var=np.array([r['var'] for r in rows]),
             loss=np.asarray(loss_log, dtype=np.float32))
    with open(os.path.join(OUTDIR, f'{args.tag}.json'), 'w') as fh:
        json.dump(dict(args=a, train_time_s=train_time, D=D, F_inf=F_INF, estimator_err=est_err,
                       max_abs_F_err=float(np.abs(F - F_true).max()),
                       max_increase=float(max(dF.max(), 0.0)), n_increases=int((dF > 0).sum()),
                       fitted_rate=rate, max_mean_err=mean_err, max_var_err=var_err,
                       wanpm=[dict(t=float(t), F_exact=float(f), **r) for t, f, r in zip(ts, F_true, rows)]),
                  fh, indent=1)
    plot(args.tag)


def plot(tag):
    d = np.load(os.path.join(OUTDIR, f'{tag}.npz'))
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for h, shift in zip(ax, (0.0, F_INF)):
        h.plot(d['t'], d['F_true'] - shift, 'k-', lw=1.6, label='analytical')
        h.plot(d['t'], d['F'] - shift, '--', color='C3', lw=1.8, label='WANPM (histogram estimate)')
        h.set_xlabel('$t$')
    ax[0].axhline(F_INF, color='gray', ls='--', lw=1, label=r'$\mathcal{F}_\infty$')
    ax[0].set_ylabel(r'$\mathcal{F}[\rho_t]$'); ax[0].legend(fontsize=8)
    ax[1].set_yscale('log')
    ax[1].set_ylabel(r'$\mathcal{F}[\rho_t]-\mathcal{F}_\infty$')
    plt.tight_layout()
    path = os.path.join(OUTDIR, f'{tag}.png')
    plt.savefig(path, dpi=150); plt.close()
    print('wrote', path)


if __name__ == '__main__':
    main()
