"""
Regenerate the Sec. 4.7 Monte Carlo figure (mc_integration_convergence.png)
from a saved ring run, with the grid reference computed from the stationary
density of the problem the run was trained on.

    python make_mc_fig.py --tag <run tag> [--out figs]
"""
import os, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import ring_lab as RL

ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True)
ap.add_argument('--out', default='figs')
args = ap.parse_args()
rec = json.load(open(os.path.join(RL.OUTDIR, f'{args.tag}.json')))
c = rec['cfg']; var_u = c['a'] / c['kappa']
out = os.path.join(RL.HERE, args.out); os.makedirs(out, exist_ok=True)

F = {'f$_4$(x,y) = exp(-((x-1)$^2$ + (y-1)$^2$))': lambda x, y: np.exp(-(x - 1) ** 2 - (y - 1) ** 2),
     'f$_5$(x,y) = cos(x)·sin(y)': lambda x, y: np.cos(x) * np.sin(y)}


def grid_reference(f, v):
    g = np.linspace(-4, 4, 501)                        # Simpson needs an odd count
    X, Y = np.meshgrid(g, g, indexing='ij')
    rho = np.exp(-(X ** 2 + Y ** 2 - RL.R0 ** 2) ** 2 / (2 * v))
    Z = simpson(simpson(rho, x=g), x=g)
    return simpson(simpson(f(X, Y) * rho, x=g), x=g) / Z


pf = RL.mlp(c['d_base'], tuple(c['hidden']), 2)
pf.load_state_dict(torch.load(os.path.join(RL.OUTDIR, f'{args.tag}.pt')))
torch.manual_seed(2026)
Ns = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
for a_, (name, f) in zip(ax, F.items()):
    ref = grid_reference(f, var_u)
    est, err = [], []
    for N in Ns:
        with torch.no_grad():
            x = pf(torch.randn(N, c['d_base'])).numpy()
        v = f(x[:, 0], x[:, 1]); est.append(v.mean()); err.append(2 * v.std() / np.sqrt(N))
    a_.errorbar(Ns, est, yerr=err, fmt='o-', capsize=4, alpha=0.75, label='MC (±2σ)')
    a_.axhline(ref, color='r', ls='--', lw=2, label='Grid reference')
    a_.set_xscale('log'); a_.set_xlabel('Number of Samples (N)'); a_.set_ylabel('Integral Estimate')
    a_.set_title(name, fontweight='bold'); a_.grid(alpha=0.3); a_.legend()
    print(f"{name}: grid reference {ref:.5f} | MC at N=1e5: {est[-1]:.5f} +- {err[-1]:.5f} "
          f"| reference under the manuscript's plotted density: {grid_reference(f, 1.0):.5f}")
plt.tight_layout(); plt.savefig(os.path.join(out, 'mc_integration_convergence.png'), dpi=120)
print('wrote', os.path.join(out, 'mc_integration_convergence.png'))
