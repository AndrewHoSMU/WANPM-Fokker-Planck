"""
Sec. 4.5 ring: Adam vs Muon across base dimension, and why r_std_ratio alone
is not a sufficient quality metric.

The stage-1 experiment R3_m5 swept the base dimension d with Adam, observed
learned std(r) / true std(r) flat at 0.78-0.84, and concluded that enlarging d
does not cure the over-concentration.  The sweep here shows that ratio can be
driven to 1.0 -- and past it, to 1.15 -- simply by making the solution worse:
the runs that hit ratio ~ 1 do so by breaking the rotational symmetry that the
true solution possesses exactly, and their Hellinger distance is the worst in
the set.  Angular total variation, plotted alongside, tracks Hellinger closely
and exposes this; the radial ratio does not.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'results_lowdim')

R0, SIGMA2 = 2.0, 1.0
LIM = 3.6

recs = json.load(open(os.path.join(OUTDIR, 'ring2d.json')))
S = np.load(os.path.join(OUTDIR, 'ring2d_samples.npz'))


def tag_of(r):
    t = f"{r['opt']} lr={r['lr']:g}"
    return t + (f" d={r['d_base']}" if r['d_base'] != 4 else "")


def true_radial(rg):
    w = rg * np.exp(-(rg ** 2 - R0 ** 2) ** 2 / (2 * SIGMA2))
    return w / np.trapezoid(w, rg)


PANELS = [
    ('Adam lr=0.01',        'Adam, d=4\n(manuscript Sec. 4.5)'),
    ('Adam lr=0.01 d=64',   'Adam, d=64\n(best Hellinger overall)'),
    ('Muon lr=0.03 d=64',   'Muon lr=3e-2, d=64\n(best Muon)'),
    ('Muon lr=0.1 d=64',    'Muon lr=1e-1, d=64\n(r_std_ratio = 1.012)'),
    ('Muon lr=0.5 d=64',    'Muon lr=5e-1, d=64\n(r_std_ratio = 1.148)'),
]

fig = plt.figure(figsize=(20, 9.2))
gs = fig.add_gridspec(2, 5, height_ratios=[1.15, 1], hspace=0.40, wspace=0.28)

by_tag = {tag_of(r): r for r in recs}
edges = np.linspace(-LIM, LIM, 141)

for i, (tag, title) in enumerate(PANELS):
    ax = fig.add_subplot(gs[0, i])
    x = S[tag]
    ax.hist2d(x[:, 0], x[:, 1], bins=[edges, edges], cmap='magma')
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(R0 * np.cos(th), R0 * np.sin(th), color='#33ff99', lw=1.4,
            ls='--', alpha=0.9)
    r = by_tag[tag]
    ax.set_title(f"{title}\nH={r['hellinger']:.3f}  ratio={r['r_std_ratio']:.3f}  "
                 f"angTV={r['angular_tv']:.3f}", fontsize=9.5)
    ax.set_aspect('equal'); ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
    ax.set_xlabel('$x_1$')
    if i == 0:
        ax.set_ylabel('$x_2$')

# radial marginals
ax = fig.add_subplot(gs[1, :2])
rg = np.linspace(0, 4.2, 600)
ax.plot(rg, true_radial(rg), 'k-', lw=2.6, label='analytical', zorder=3)
for tag, title in PANELS:
    x = S[tag]
    rr = np.hypot(x[:, 0], x[:, 1])
    h, e = np.histogram(rr, bins=140, range=(0, 4.2), density=True)
    ax.plot(0.5 * (e[:-1] + e[1:]), h, lw=1.7,
            label=f"{title.splitlines()[0]}  ({by_tag[tag]['hellinger']:.3f})")
ax.set_xlabel('$r$'); ax.set_ylabel('radial density $p(r)$')
ax.set_title('Radial marginal', fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(0.6, 3.6)

# the metric disagreement
ax = fig.add_subplot(gs[1, 2:4])
for r in recs:
    c = '#1f77b4' if r['opt'] == 'Adam' else '#6a3d9a'
    m = {4: 'o', 16: 's', 64: '^'}[r['d_base']]
    ax.scatter(r['r_std_ratio'], r['hellinger'], c=c, marker=m, s=64,
               alpha=0.85, edgecolor='k', linewidth=0.5)
ax.axvline(1.0, color='k', ls='--', lw=1.6)
ax.text(1.005, ax.get_ylim()[1] * 0.97, ' "correct" radial width',
        fontsize=8.5, va='top')
ax.axvspan(0.78, 0.84, color='orange', alpha=0.18)
ax.text(0.81, 0.55, 'R3_m5 band\n(0.78-0.84)', fontsize=8, ha='center', color='#a35200')
ax.set_xlabel('learned std($r$) / true std($r$)')
ax.set_ylabel('Hellinger distance')
ax.set_title('The two metrics disagree: driving the radial ratio to 1.0\n'
             'makes the Hellinger distance WORSE', fontsize=10.5)
ax.grid(alpha=0.3)
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([], [], marker='o', ls='', color='#1f77b4', label='Adam'),
    Line2D([], [], marker='o', ls='', color='#6a3d9a', label='Muon'),
    Line2D([], [], marker='o', ls='', color='gray', label='d=4'),
    Line2D([], [], marker='s', ls='', color='gray', label='d=16'),
    Line2D([], [], marker='^', ls='', color='gray', label='d=64')],
    fontsize=8, ncol=2)

# angular TV tracks Hellinger
ax = fig.add_subplot(gs[1, 4])
xs = [r['angular_tv'] for r in recs]; ys = [r['hellinger'] for r in recs]
cs = ['#1f77b4' if r['opt'] == 'Adam' else '#6a3d9a' for r in recs]
ax.scatter(xs, ys, c=cs, s=60, alpha=0.85, edgecolor='k', linewidth=0.5)
z = np.polyfit(xs, ys, 1); xx = np.linspace(min(xs), max(xs), 20)
ax.plot(xx, np.polyval(z, xx), 'k--', lw=1.4,
        label=f'corr = {np.corrcoef(xs, ys)[0,1]:.2f}')
ax.set_xlabel('angular total variation')
ax.set_ylabel('Hellinger distance')
ax.set_title('Angular TV DOES track quality', fontsize=10.5)
ax.legend(fontsize=8.5); ax.grid(alpha=0.3)

fig.suptitle('Sec. 4.5 ring: Adam vs Muon over base dimension $d \\in \\{4,16,64\\}$  '
             '(green dashed circle = true ring radius $r_0=2$)\n'
             'Muon does not improve the ring; and the radial-width ratio used by '
             'R3_m5 can be driven to 1.0 by degrading the solution',
             fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.92])
out = os.path.join(OUTDIR, 'ring_muon_comparison.png')
plt.savefig(out, dpi=140)
plt.close()
print(f"wrote {out}")
