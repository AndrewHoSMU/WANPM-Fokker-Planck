"""
Sec. 4.8 figure: 2 x 6 scatter plots, learned pushforward (top) against direct
Monte Carlo simulation of the SDE (bottom), at t = 0, 0.1, ..., 0.5.

    python make_ring_time_fig.py --tag ringt_h64 [--out figs/ring_time]
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ring_lab as RL
import ring_time as RT

ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True)
ap.add_argument('--out', default='figs/ring_time')
ap.add_argument('--n', type=int, default=5000)
args = ap.parse_args()
out = os.path.join(RL.HERE, args.out); os.makedirs(out, exist_ok=True)
rec = json.load(open(os.path.join(RL.OUTDIR, f'{args.tag}.json')))
S = np.load(os.path.join(RL.OUTDIR, f'{args.tag}.npz'))

tt = np.linspace(0, 2 * np.pi, 400)
fig, ax = plt.subplots(2, 6, figsize=(16, 5.9), sharex=True, sharey=True)
rows = [('learned', 'Neural pushforward', 'royalblue'), ('mc', 'Direct simulation', 'darkorange')]
for i, (key, name, col) in enumerate(rows):
    for j, t in enumerate(RT.TIMES):
        a = ax[i, j]; x = S[f'{key}_{t:.1f}'][:args.n]; st = rec[key][j]
        a.scatter(x[:, 0], x[:, 1], s=0.8, alpha=0.45, color=col, rasterized=True)
        a.plot(RL.R0 * np.cos(tt), RL.R0 * np.sin(tt), 'r--', lw=1.6)
        a.set_xlim(-3.3, 3.3); a.set_ylim(-3.3, 3.3); a.set_aspect('equal'); a.grid(alpha=0.3)
        a.text(0.03, 0.97, f"mean: ({st['mean_x']:.2f}, {st['mean_y']:.2f})\n"
                           f"$r$: {st['r_mean']:.2f} $\\pm$ {st['r_std']:.2f}",
               transform=a.transAxes, va='top', ha='left', fontsize=9.5,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        if i == 0:
            a.set_title(f't = {t:.1f}', fontsize=15, fontweight='bold')
        if i == 1:
            a.set_xlabel('x', fontsize=12)
        if j == 0:
            a.set_ylabel(f'{name}\ny', fontsize=13)
plt.tight_layout()
path = os.path.join(out, '2dring_time_evolution_scatter.png')
plt.savefig(path, dpi=200); plt.close()
print('wrote', path)

# training loss (for reference; not used in the paper)
plt.figure(figsize=(9, 3.6)); plt.semilogy(S['losses'], lw=0.7)
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(out, 'ring_time_loss.png'), dpi=120); plt.close()
