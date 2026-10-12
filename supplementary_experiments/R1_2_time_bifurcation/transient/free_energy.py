"""
Free-energy dissipation along the learned {rho_t} (Reviewer 3, minor comment 3), on the
transient one-peak -> two-peak problem (manuscript Sec. 4.4, case E, sigma=1).

For dX = -grad V dt + sigma dW (generator (sigma^2/2) Laplacian), with D = sigma^2/2,

    F[rho] = int V rho dx + D int rho log rho dx,     dF/dt = -int rho |grad V + D grad log rho|^2 <= 0,
    F[rho] - F_inf = D * KL(rho || rho_inf),          F_inf = -D log Z,  Z = int exp(-V/D) dx.

Two curves are plotted:
  * WANPM      samples of the saved pushforward at a dense time grid; int V rho is a plain
               sample mean, int rho log rho a 2-D histogram plug-in estimate (Miller-Madow
               corrected, O(h^2) binning bias removed by Richardson extrapolation in the bin width)
  * PDE        deterministic reference: the finite-volume solve of reference_fv.py
Samples drawn from the PDE solution, and the Euler-Maruyama samples of reference_mc.py (5 snapshot
times, two independent sets each), are pushed through the SAME histogram estimator as a check of
the estimator against the PDE solve (printed and stored in the json, not plotted).

    python free_energy.py                      # results/E_free_energy.{json,npz,png}
    python free_energy.py --plot-only
"""
import os, json, time, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import problem as P
import metrics as Mx
import reference_fv as FV
import wanpm_transient as W
from common_wanpm import TimeDepPushforward

device = W.device
OUTDIR = W.OUTDIR


# ---------------------------------------------------------------------------
# sample-based estimator
# ---------------------------------------------------------------------------
class FreeEnergyAccumulator:
    """Streams samples (torch, on device) into E[V] and a 2-D histogram on [-L, L]^2."""

    def __init__(self, L, h):
        self.L, self.nb = L, 2 * int(round(L / h))
        self.h = 2 * L / self.nb
        self.counts = torch.zeros(self.nb * self.nb, dtype=torch.long, device=device)
        self.sumV, self.n, self.n_out = 0.0, 0, 0

    def add(self, x):
        self.sumV += P.V_torch(x).double().sum().item()
        self.n += x.shape[0]
        idx = torch.floor((x + self.L) / self.h).long()
        ok = ((idx >= 0) & (idx < self.nb)).all(1)
        self.n_out += int((~ok).sum())
        idx = idx[ok]
        self.counts += torch.bincount(idx[:, 0] * self.nb + idx[:, 1], minlength=self.nb * self.nb)

    @staticmethod
    def _neg_entropy(counts, h):
        c = counts[counts > 0].double()
        N = c.sum()
        p = c / N
        # plug-in int rho log rho, minus the Miller-Madow bias (occupied bins - 1) / 2N
        return (p * torch.log(p / h ** 2)).sum().item() - (len(c) - 1) / (2 * N.item())

    def result(self, D):
        # binning smooths rho and lowers int rho log rho by O(h^2); remove the leading term by
        # Richardson extrapolation between bin widths h and 2h (2x2 blocks of the same histogram)
        nb = self.nb
        s_h = self._neg_entropy(self.counts, self.h)
        s_2h = self._neg_entropy(self.counts.view(nb // 2, 2, nb // 2, 2).sum((1, 3)), 2 * self.h)
        neg_ent = (4 * s_h - s_2h) / 3
        EV = self.sumV / self.n
        return dict(EV=EV, neg_entropy=neg_ent, F=EV + D * neg_ent, bin_bias=D * (s_h - neg_ent),
                    frac_outside=self.n_out / self.n)


@torch.no_grad()
def wanpm_free_energy(pf, case, d_base, t, D, n, L, h, chunk=500_000):
    acc = FreeEnergyAccumulator(L, h)
    rho0 = P.rho0_sampler(case, device)
    for _ in range(n // chunk):
        tt = t * torch.ones(chunk, 1, device=device)
        acc.add(pf(tt, rho0(chunk), W.sample_base(chunk, d_base)))
    return acc.result(D)


def samples_free_energy(x, D, L, h):
    acc = FreeEnergyAccumulator(L, h)
    acc.add(torch.as_tensor(x, dtype=torch.float32, device=device))
    return acc.result(D)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='E_adam_saddle', help='saved WANPM run (results/<tag>.pt)')
    ap.add_argument('--n', type=int, default=4_000_000, help='WANPM samples per time')
    ap.add_argument('--n-times', type=int, default=101)
    ap.add_argument('--bin', type=float, default=0.04, help='histogram bin width')
    ap.add_argument('--L', type=float, default=3.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--plot-only', action='store_true')
    args = ap.parse_args()

    ck = torch.load(os.path.join(OUTDIR, f'{args.tag}.pt'), map_location=device)
    a = ck['args']
    case = a['case']
    out = f'{case}_free_energy'
    if args.plot_only:
        plot(out); return
    W.BASE = a['base']
    D = 0.5 * P.CASES[case]['sigma'] ** 2
    torch.manual_seed(args.seed)
    pf = TimeDepPushforward(2, a['d_base'], tuple(a['hidden']), power=a['power']).to(device)
    pf.load_state_dict(ck['pf']); pf.eval()

    t0 = time.time()
    pde = FV.solve(case, L=args.L, n=300)
    pde_coarse = FV.solve(case, L=args.L, n=200)
    grid_err = float(np.abs(pde['F'] - pde_coarse['F']).max())
    print(f'PDE reference: F(0)={pde["F"][0]:.4f} F(T)={pde["F"][-1]:.4f} F_inf={pde["F_inf"]:.4f} '
          f'mass={pde["mass"]:.6f}  max|F_h=0.02 - F_h=0.03|={grid_err:.1e}  [{time.time()-t0:.0f}s]', flush=True)

    # estimator check: samples of known free energy (drawn from the PDE solution, and the
    # Euler-Maruyama particles), two independent sets each, through the same estimator
    check = {}
    for kind in ('fv', 'mc'):
        times, ra, rb = Mx.load_reference(case, kind)
        check[kind] = []
        for i, t in enumerate(times):
            Fa, Fb = (samples_free_energy(r[i], D, args.L, args.bin)['F'] for r in (ra, rb))
            F_pde = float(np.interp(t, pde['t'], pde['F']))
            check[kind].append(dict(t=float(t), F_a=Fa, F_b=Fb, F_pde=F_pde))
            print(f'  {kind} samples t={t:<5} F={Fa:.4f} / {Fb:.4f}   PDE {F_pde:.4f}', flush=True)
    est_err = {k: max(abs(F_ - c['F_pde']) for c in v for F_ in (c['F_a'], c['F_b'])) for k, v in check.items()}
    print(f'  max |F_est - F_pde|: {est_err}', flush=True)

    ts = np.linspace(0.0, P.T_FINAL, args.n_times)
    rows = [wanpm_free_energy(pf, case, a['d_base'], float(t), D, args.n, args.L, args.bin) for t in ts]
    F = np.array([r['F'] for r in rows])
    bin_err = float(max(abs(r['bin_bias']) for r in rows))
    i_min = int(F.argmin())
    print(f'WANPM: F(0)={F[0]:.4f} F(T)={F[-1]:.4f}  min F={F[i_min]:.4f} at t={ts[i_min]:.2f}  '
          f'rise after the minimum {F[-1] - F[i_min]:.2e}  removed binning bias <= {bin_err:.1e}  '
          f'[{time.time()-t0:.0f}s]', flush=True)

    np.savez(os.path.join(OUTDIR, f'{out}.npz'), t=ts, F=F, EV=np.array([r['EV'] for r in rows]),
             neg_entropy=np.array([r['neg_entropy'] for r in rows]), t_pde=pde['t'], F_pde=pde['F'])
    with open(os.path.join(OUTDIR, f'{out}.json'), 'w') as fh:
        json.dump(dict(args=vars(args), model_tag=args.tag, case=case, D=D, F_inf=pde['F_inf'],
                       pde_grid_err=grid_err, wanpm_bin_err=bin_err, estimator_check=check,
                       estimator_err=est_err, max_abs_diff_to_pde=float(np.abs(F - np.interp(ts, pde['t'], pde['F'])).max()),
                       wanpm_t_min=float(ts[i_min]), wanpm_rise_after_min=float(F[-1] - F[i_min]),
                       wanpm_frac_outside=float(max(r['frac_outside'] for r in rows)),
                       wanpm=[dict(t=float(t), **r) for t, r in zip(ts, rows)]), fh, indent=1)
    plot(out)


def plot(out):
    d = np.load(os.path.join(OUTDIR, f'{out}.npz'))
    with open(os.path.join(OUTDIR, f'{out}.json')) as fh:
        js = json.load(fh)
    F_inf = js['F_inf']
    fig, ax = plt.subplots(figsize=(7.9, 4.0))
    ax.plot(d['t_pde'], d['F_pde'], 'k-', lw=1.6, label='reference (finite-volume PDE solve)')
    ax.plot(d['t'], d['F'], '-', color='C3', lw=1.6, label='WANPM (histogram estimate)')
    ax.axhline(F_inf, color='gray', ls='--', lw=1, label=r'$\mathcal{F}_\infty=-\frac{\sigma^2}{2}\log Z$')
    # fonts sized for inclusion at \textwidth (390pt), which shrinks the 7.9in canvas by ~0.68
    ax.set_xlabel('$t$', fontsize=15); ax.set_ylabel(r'$\mathcal{F}[\rho_t]$', fontsize=15)
    ax.tick_params(labelsize=13); ax.legend(fontsize=12.5)
    plt.tight_layout()
    path = os.path.join(OUTDIR, f'{out}.png')
    plt.savefig(path, dpi=150); plt.close()
    print('wrote', path)


if __name__ == '__main__':
    main()
