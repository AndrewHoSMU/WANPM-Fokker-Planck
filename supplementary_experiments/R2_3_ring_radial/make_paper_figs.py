"""
Regenerate the three Sec. 4.6 ring figures of the manuscript from a saved run:

    ring_potential_training.png         (Fig. 10: scatter, heat map, loss)
    ring_potential_comparison.png       (Fig. 11: analytical vs learned surface)
    ring_potential_radial_angular.png   (Fig. 12: radial and angular marginals)

    python make_paper_figs.py --tag <run tag> [--out figs]

The analytical curves use the stationary density of the problem the run was
actually trained on:  rho ~ exp(-(r^2 - r0^2)^2 / (2 a/kappa)).
"""
import os, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ring_lab as RL

ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True)
ap.add_argument('--out', default='figs')
ap.add_argument('--n', type=int, default=2_000_000)
ap.add_argument('--bins', type=int, default=60, help='grid for the 3-D surface plot')
args = ap.parse_args()

rec = json.load(open(os.path.join(RL.OUTDIR, f'{args.tag}.json')))
c = rec['cfg']; var_u = c['a'] / c['kappa']
out = os.path.join(RL.HERE, args.out); os.makedirs(out, exist_ok=True)
losses = np.load(os.path.join(RL.OUTDIR, f'{args.tag}.npz'))['losses']

pf = RL.mlp(c['d_base'], tuple(c['hidden']), 2)
pf.load_state_dict(torch.load(os.path.join(RL.OUTDIR, f'{args.tag}.pt')))
torch.manual_seed(123)
with torch.no_grad():
    x = torch.cat([pf(torch.randn(200000, c['d_base'])) for _ in range(args.n // 200000)]).numpy()
r = np.hypot(x[:, 0], x[:, 1]); th = np.arctan2(x[:, 1], x[:, 0])
m = RL.metrics(x[:200000], var_u)
print({k: round(v, 4) for k, v in m.items()})
print('sample mean', x.mean(0))

tt = np.linspace(0, 2 * np.pi, 400)

# ---- Fig. 12: radial + angular ---------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
rg = np.linspace(0, 4, 800)
ax[0].hist(r, bins=80, range=(0, 4), density=True, color='skyblue', edgecolor='k',
           alpha=0.8, label='Learned')
ax[0].plot(rg, RL.true_radial(rg, var_u), 'r-', lw=2.2, label='Analytical')
ax[0].axvline(RL.R0, color='g', ls='--', lw=2, label=f'Target r$_0$={RL.R0:.1f}')
ax[0].set_xlabel('Radius r'); ax[0].set_ylabel('Probability Density p(r)')
ax[0].set_title('Radial Distribution'); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].hist(th, bins=36, range=(-np.pi, np.pi), density=True, color='coral',
           edgecolor='k', alpha=0.7)
ax[1].axhline(1 / (2 * np.pi), color='r', ls='--', lw=2, label='Uniform (2π)$^{-1}$')
ax[1].set_xlabel('Angle θ (radians)'); ax[1].set_ylabel('Probability Density p(θ)')
ax[1].set_title('Angular Distribution (should be uniform)')
ax[1].set_ylim(0, 0.205); ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(out, 'ring_potential_radial_angular.png'), dpi=150)
plt.close()

# ---- Fig. 10: scatter + loss (heat map dropped: same information as the scatter) ----
fig, ax = plt.subplots(1, 2, figsize=(18, 5.6), gridspec_kw=dict(width_ratios=[1, 2.1]))
ax[0].scatter(x[:5000, 0], x[:5000, 1], s=1.5, alpha=0.5)
ax[0].plot(RL.R0 * np.cos(tt), RL.R0 * np.sin(tt), 'r--', lw=1.8, label=f'Target ring r={RL.R0:.1f}')
ax[0].set_xlim(-3.5, 3.5); ax[0].set_ylim(-3.5, 3.5); ax[0].set_aspect('equal')
ax[0].set_xlabel('x'); ax[0].set_ylabel('y'); ax[0].set_title('Learned Distribution (Scatter)')
ax[0].legend(loc='upper right'); ax[0].grid(alpha=0.3)
ax[1].semilogy(losses, lw=0.8)
ax[1].set_xlim(0, len(losses))
ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Loss'); ax[1].set_title('Training Loss Convergence')
ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(out, 'ring_potential_training.png'), dpi=150)
plt.close()

# ---- Fig. 11: analytical vs learned surfaces --------------------------------
nb = args.bins; lim = 3.5
ed = np.linspace(-lim, lim, nb + 1); cc = 0.5 * (ed[:-1] + ed[1:])
X, Y = np.meshgrid(cc, cc, indexing='ij')
pt = np.exp(-(X ** 2 + Y ** 2 - RL.R0 ** 2) ** 2 / (2 * var_u)); pt /= pt.sum() * (ed[1] - ed[0]) ** 2
ph, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=[ed, ed], density=True)
zmax = 1.1 * max(pt.max(), ph.max())
print(f'surface plot: {nb}x{nb} bins, analytical max {pt.max():.3f}, learned max {ph.max():.3f}')
fig = plt.figure(figsize=(16, 6.5))
for i, (Z, title) in enumerate([(pt, f'Analytical Distribution (r$_0$={RL.R0:.1f}, σ=1.0)'),
                                (ph, 'Learned Distribution')]):
    a3 = fig.add_subplot(1, 2, i + 1, projection='3d')
    s = a3.plot_surface(X, Y, Z, cmap='viridis', vmin=0, vmax=zmax, linewidth=0,
                        antialiased=True, rcount=nb, ccount=nb)
    a3.set_zlim(0, zmax); a3.set_xlabel('x'); a3.set_ylabel('y'); a3.set_zlabel('Probability Density')
    a3.set_title(title); a3.view_init(elev=32, azim=-55)
    fig.colorbar(s, ax=a3, shrink=0.55, pad=0.08)
plt.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.04, wspace=0.05)
plt.savefig(os.path.join(out, 'ring_potential_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'wrote 3 figures to {out}')
