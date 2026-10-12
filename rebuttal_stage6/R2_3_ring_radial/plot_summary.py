"""
One-figure summary of the ring radial-marginal study.

  (a) the manuscript's original problem (drift -2(r^2-r0^2)x, a = sigma^2/2),
      manuscript configuration: the learned histogram against BOTH the curve
      plotted in the manuscript's Fig. 12 and the true stationary density;
  (b) the self-consistent problem (drift = -grad V of the stated V, a = sigma^2),
      manuscript configuration;
  (c) same problem, generator trained with the two-group learning-rate split;
  (d) radial Hellinger and angular TV for every configuration tried.

    python plot_summary.py --a P3_base_s0 --b P1_base_s0 --c P1_base_muon_s1
"""
import os, json, glob, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ring_lab as RL

ap = argparse.ArgumentParser()
ap.add_argument('--a', default='P3_base_s0')
ap.add_argument('--b', default='P1_base_s0')
ap.add_argument('--c', default='P1_base_muon_s1')
ap.add_argument('--out', default='figs')
args = ap.parse_args()
os.makedirs(os.path.join(RL.HERE, args.out), exist_ok=True)


def load(tag):
    rec = json.load(open(os.path.join(RL.OUTDIR, f'{tag}.json')))
    x = np.load(os.path.join(RL.OUTDIR, f'{tag}.npz'))['x']
    return rec, np.hypot(x[:, 0], x[:, 1])


rg = np.linspace(0, 4, 1200)
fig, ax = plt.subplots(1, 4, figsize=(23, 5.2), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.25]))
titles = ['(a) original problem, manuscript configuration',
          '(b) consistent problem, manuscript configuration',
          '(c) consistent problem, split learning rate']
for i, tag in enumerate([args.a, args.b, args.c]):
    rec, r = load(tag); c = rec['cfg']; var_u = c['a'] / c['kappa']
    ax[i].hist(r, bins=80, range=(0, 4), density=True, color='skyblue', edgecolor='k',
               alpha=0.8, label='learned')
    ax[i].plot(rg, RL.true_radial(rg, var_u), 'r-', lw=2.4, label='true stationary density')
    if i == 0:
        ax[i].plot(rg, RL.true_radial(rg, 1.0), color='0.35', ls='--', lw=2,
                   label='curve plotted in manuscript Fig. 12')
    ax[i].axvline(RL.R0, color='g', ls=':', lw=1.5)
    ax[i].set_xlim(0.4, 3.2); ax[i].set_xlabel('radius $r$'); ax[i].grid(alpha=0.3)
    ax[i].set_title(f"{titles[i]}\nradial Hellinger {rec['radial_hellinger']:.3f},  "
                    f"IQR ratio {rec['r_iqr_ratio']:.2f},  mass at $r<1$: {100*rec['frac_inside']:.2f}%",
                    fontsize=10)
    ax[i].legend(fontsize=8.5, loc='upper left')
ax[0].set_ylabel('radial density $p(r)$')

# (d) every P1 configuration: radial Hellinger vs angular TV
groups = {}
for f in glob.glob(os.path.join(RL.OUTDIR, 'P1_*_s*.json')):
    rr = json.load(open(f))
    if rr.get('diverged'):
        continue
    groups.setdefault(rr['tag'].rsplit('_s', 1)[0], []).append(rr)


def style(name):
    n = name[3:]
    if 'morl' in n or n == 'multi':
        return None
    split = ('muon' in n) or ('adamsplit' in n)
    wave = any(k in n for k in ('g2', 'g4', 'g8', 'mex', 'herm', 'sech', 'rbf'))
    if n == 'base':
        return ('k', '*', 260, 'manuscript configuration')
    if 'an' in n.split('_')[-1] and n.split('_')[-1][:2] == 'an':
        return ('#d95f02', 'v', 70, '+ $\\sigma$-annealing')
    if split and wave:
        return ('#7570b3', 'D', 70, 'split learning rate + wavelets')
    if split:
        return ('#1b9e77', 's', 80, 'split learning rate (Muon / Adam)')
    if wave:
        return ('#e7298a', 'o', 70, 'plane waves + wavelets')
    return ('0.55', 'x', 50, 'other knobs ($M$, $d$, depth, epochs, $|w|$)')


seen = set()
for name, rs in sorted(groups.items()):
    st = style(name)
    if st is None:
        continue
    hx = np.mean([q['radial_hellinger'] for q in rs]); ay = np.mean([q['angular_tv'] for q in rs])
    ax[3].scatter(hx, ay, c=st[0], marker=st[1], s=st[2], edgecolor='k', linewidth=0.6,
                  label=None if st[3] in seen else st[3], zorder=3)
    seen.add(st[3])
rng = np.random.default_rng(0)
fl = [RL.metrics(RL.sample_true(200000, 1.0, rng), 1.0) for _ in range(3)]
ax[3].scatter(np.mean([q['radial_hellinger'] for q in fl]), np.mean([q['angular_tv'] for q in fl]),
              c='gold', marker='P', s=150, edgecolor='k', label='exact samples (noise floor)', zorder=4)
ax[3].set_xlabel('radial Hellinger distance'); ax[3].set_ylabel('angular total variation')
ax[3].set_title('(d) all configurations on the consistent problem\n(seed-averaged; lower-left is better)',
                fontsize=10)
ax[3].set_xlim(0, 0.16); ax[3].set_ylim(0, 0.125)
ax[3].grid(alpha=0.3); ax[3].legend(fontsize=8.2, loc='upper left')
plt.tight_layout()
out = os.path.join(RL.HERE, args.out, 'ring_radial_summary.png')
plt.savefig(out, dpi=140); plt.close()
print('wrote', out)
