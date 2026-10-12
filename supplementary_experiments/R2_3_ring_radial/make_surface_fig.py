"""
Fig. 11-style 3-D comparison with the learned density shown as a Gaussian
kernel density estimate (KDE) of the generated samples.

    python make_surface_fig.py --tag final_split_lr_wavelet --kde 0.10 [--n 20000000]

The learned density is  rho_h = (samples histogram on a fine grid) * N(0, h^2 I).
The analytical panel is the exact density (no smoothing).  --kde 0 gives the raw
histogram on a --bins x --bins grid instead.
"""
import os, json, argparse
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ring_lab as RL

ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True)
ap.add_argument('--out', default='figs/final')
ap.add_argument('--n', type=int, default=20_000_000)
ap.add_argument('--kde', type=float, default=0.10, help='Gaussian bandwidth h (0 = raw histogram)')
ap.add_argument('--bins', type=int, default=60, help='grid for the raw histogram when --kde 0')
args = ap.parse_args()

rec = json.load(open(os.path.join(RL.OUTDIR, f'{args.tag}.json')))
c = rec['cfg']; var_u = c['a'] / c['kappa']
out = os.path.join(RL.HERE, args.out); os.makedirs(out, exist_ok=True)
pf = RL.mlp(c['d_base'], tuple(c['hidden']), 2)
pf.load_state_dict(torch.load(os.path.join(RL.OUTDIR, f'{args.tag}.pt')))

lim = 3.5
nb = 280 if args.kde > 0 else args.bins
ed = np.linspace(-lim, lim, nb + 1); cc = 0.5 * (ed[:-1] + ed[1:]); dx = ed[1] - ed[0]
H = np.zeros((nb, nb)); torch.manual_seed(123)
with torch.no_grad():
    for _ in range(max(1, args.n // 500000)):
        x = pf(torch.randn(500000, c['d_base'])).numpy()
        H += np.histogram2d(x[:, 0], x[:, 1], bins=[ed, ed])[0]
ph = H / H.sum() / dx ** 2
if args.kde > 0:
    ph = gaussian_filter(ph, args.kde / dx, mode='constant')
X, Y = np.meshgrid(cc, cc, indexing='ij')
pt = np.exp(-(X ** 2 + Y ** 2 - RL.R0 ** 2) ** 2 / (2 * var_u)); pt /= pt.sum() * dx ** 2
ring = pt > 0.5 * pt.max()
print(f'h={args.kde:g}, N={args.n:,}: analytical max {pt.max():.3f}, learned max {ph.max():.3f}, '
      f'rms relative deviation on the ring {np.sqrt(np.mean((ph[ring] / pt[ring] - 1) ** 2)):.3f}')

zmax = 1.1 * max(pt.max(), ph.max())
title_l = 'Learned Distribution' + (f' (KDE, h={args.kde:g})' if args.kde > 0 else '')
fig = plt.figure(figsize=(16, 6.5))
for i, (Z, title) in enumerate([(pt, f'Analytical Distribution (r$_0$={RL.R0:.1f}, σ=1.0)'), (ph, title_l)]):
    a3 = fig.add_subplot(1, 2, i + 1, projection='3d')
    s = a3.plot_surface(X, Y, Z, cmap='viridis', vmin=0, vmax=zmax, linewidth=0,
                        antialiased=True, rcount=100, ccount=100)
    a3.set_zlim(0, zmax); a3.set_xlabel('x'); a3.set_ylabel('y'); a3.set_zlabel('Probability Density')
    a3.set_title(title); a3.view_init(elev=32, azim=-55)
    fig.colorbar(s, ax=a3, shrink=0.55, pad=0.08)
plt.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.04, wspace=0.05)
name = f'ring_potential_comparison_kde{args.kde:.2f}.png' if args.kde > 0 else 'ring_potential_comparison.png'
plt.savefig(os.path.join(out, name), dpi=150, bbox_inches='tight'); plt.close()
print('wrote', os.path.join(args.out, name))
