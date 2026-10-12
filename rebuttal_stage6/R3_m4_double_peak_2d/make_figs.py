"""Figures for the Sec. 4.3 re-run (same layout as the manuscript's Figs. 5 and 6).

    python make_figs.py --tag mixed_s0      # writes figs/double_peak_2d_*.png
"""
import os, json, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='mixed_s0')
ap.add_argument('--suffix', default='')
args = ap.parse_args()
# the paper includes these at well under their drawn width, so draw the text large
plt.rcParams.update({'font.size': 15, 'axes.titlesize': 16, 'legend.fontsize': 13})
FIG = os.path.join(HERE, 'figs'); os.makedirs(FIG, exist_ok=True)

D = np.load(os.path.join(HERE, 'results', f'{args.tag}.npz'))
x, losses, avals = D['x'], D['losses'], D['a']
rec = json.load(open(os.path.join(HERE, 'results', f'{args.tag}.json')))
A, LIM = 0.16, 1.5


def V(X, Y):
    return ((X - 1) ** 2 + (Y - 1) ** 2) * ((X + 1) ** 2 + (Y + 1) ** 2)


# --- Fig. 5 layout: scatter + density heatmap --------------------------------
# two square panels of equal size; the colourbar hangs off the right panel
fig, ax = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
for a_ in ax:
    a_.set_box_aspect(1)
n_sc = 10000
ax[0].scatter(x[:n_sc, 0], x[:n_sc, 1], s=1, alpha=0.5)
ax[0].scatter([-1, 1], [-1, 1], c='red', marker='x', s=100, linewidths=2, label='Target peaks')
ax[0].set_xlim(-1.6, 1.6); ax[0].set_ylim(-1.6, 1.6)
ax[0].set_xlabel('x'); ax[0].set_ylabel('y'); ax[0].grid(alpha=0.3); ax[0].legend(loc='upper left')
ax[0].set_title(f'Learned Distribution (Scatter, {n_sc:,} samples)')
ed = np.linspace(-LIM, LIM, 51)
H, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=[ed, ed], density=True)
im = ax[1].imshow(H.T, origin='lower', extent=[-LIM, LIM, -LIM, LIM], cmap='viridis', aspect='auto')
ax[1].scatter([-1, 1], [-1, 1], c='red', marker='x', s=100, linewidths=2, label='Target peaks')
ax[1].set_xlabel('x'); ax[1].set_ylabel('y'); ax[1].legend(loc='upper right')
ax[1].set_title('Learned Distribution (Density)')
fig.colorbar(im, cax=ax[1].inset_axes([1.04, 0, 0.05, 1]), label='Density')
p1 = os.path.join(FIG, f'double_peak_2d_plot1{args.suffix}.png'); plt.savefig(p1, dpi=150); plt.close()

# --- Fig. 6 layout: analytical vs learned surface ----------------------------
ed = np.linspace(-LIM, LIM, 76); c = 0.5 * (ed[:-1] + ed[1:]); h = ed[1] - ed[0]
X, Y = np.meshgrid(c, c, indexing='ij')
gf = np.linspace(-2.5, 2.5, 2001); Xf, Yf = np.meshgrid(gf, gf, indexing='ij')
Z = np.exp(-2 * V(Xf, Yf) / A).sum() * (gf[1] - gf[0]) ** 2          # normalising constant
# analytical density averaged over each histogram cell, so both panels are binned alike
sub = (np.arange(8) + 0.5) / 8 - 0.5
PT = np.zeros_like(X)
for dx in sub:
    for dy in sub:
        PT += np.exp(-2 * V(X + dx * h, Y + dy * h) / A)
PT /= (len(sub) ** 2 * Z)
PL, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=[ed, ed])
PL = PL / len(x) / h ** 2
zmax = 1.05 * max(PT.max(), PL.max())
fig = plt.figure(figsize=(15, 6))
for i, (P, ttl) in enumerate(((PT, r'Analytical Distribution ($\sigma$=0.4)'), (PL, 'Learned Distribution'))):
    a3 = fig.add_subplot(1, 2, i + 1, projection='3d')
    s = a3.plot_surface(X, Y, P, cmap='viridis', alpha=0.8, vmin=0, vmax=zmax, rcount=75, ccount=75)
    a3.set_zlim(0, zmax); a3.set_xlabel('x'); a3.set_ylabel('y'); a3.set_title(ttl)
    fig.colorbar(s, ax=a3, shrink=0.5, label='Probability Density')
plt.tight_layout()
p2 = os.path.join(FIG, f'double_peak_2d_plot2{args.suffix}.png'); plt.savefig(p2, dpi=150); plt.close()

# --- diagnostic: profile along the minimum-energy path + loss ----------------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
s = (x[:, 0] + x[:, 1]) / np.sqrt(2)
sg = np.linspace(-2.2, 2.2, 1500)
# marginal of the analytical density along the diagonal coordinate s
tq = np.linspace(-1.5, 1.5, 1201)
ps = np.array([np.exp(-2 * V((si + tq) / np.sqrt(2), (si - tq) / np.sqrt(2)) / A).sum() for si in sg])
ps /= np.trapezoid(ps, sg)
for a_, logy in ((ax[0], False), (ax[1], True)):
    a_.hist(s, bins=220, range=(-2.2, 2.2), density=True, color='#7ba7d1', alpha=0.8, label='learned')
    a_.plot(sg, ps, 'k-', lw=1.5, label='analytical')
    a_.set_xlabel(r'$s=(x_1+x_2)/\sqrt{2}$'); a_.legend()
ax[0].set_ylabel('marginal density'); ax[0].set_title('Marginal along the diagonal')
ax[1].set_yscale('log'); ax[1].set_ylim(1e-4, 10); ax[1].set_title('Same, logarithmic scale')
ep = np.arange(len(losses))
ax[2].semilogy(ep, losses, lw=0.4, color='#9ec3e6')
k = 500; ma = np.convolve(losses, np.ones(k) / k, mode='valid')
ax[2].semilogy(ep[k - 1:], ma, color='#1f4e79', lw=1.5, label='loss (500-epoch mean)')
ax[2].set_xlabel('epoch'); ax[2].set_ylabel(r'$L_{\rm total}$')
a2 = ax[2].twinx(); a2.plot(ep, avals, 'r--', lw=1.2); a2.set_ylabel(r'$\sigma^2$ (annealed)', color='r')
ax[2].set_title('Training loss and annealing schedule'); ax[2].legend(loc='upper right')
plt.tight_layout()
p3 = os.path.join(FIG, f'double_peak_2d_profile{args.suffix}.png'); plt.savefig(p3, dpi=150); plt.close()
print('wrote', p1, p2, p3, sep='\n  ')
print(f"H={rec['hellinger']:.4f} stray={rec['stray_frac']:.2e} peak learned/analytical = {PL.max():.2f}/{PT.max():.2f}")
