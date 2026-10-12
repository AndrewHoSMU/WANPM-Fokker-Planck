"""Three-row heatmap comparison: finite-volume reference / WANPM / t-NF, case E (sigma=1).
No marginal-distribution row -- heatmaps only, per user request. The top row is the density
computed by reference_fv.py (its cells coincide with the histogram bins of the two sample rows).

    python make_compare_figure.py --tnf-tag E_tnf_L4spl --wanpm-tag E_adam_saddle

The manuscript figure (Sec. 4.11) shows four of the five snapshot times, t in {0.05, 0.2, 0.5, 1}
(default of --times); pass --times 0.05,0.1,0.2,0.5,1 for all five (results/E_compare_wanpm_tnf_5times.png).
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import reference_fv as FV

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results')

ap = argparse.ArgumentParser()
ap.add_argument('--case', default='E')
ap.add_argument('--wanpm-tag', default='E_adam_saddle')
ap.add_argument('--tnf-tag', default='E_tnf_L4spl')
ap.add_argument('--times', default='0.05,0.2,0.5,1', help='comma-separated snapshot times to show')
ap.add_argument('--out', default=None)
args = ap.parse_args()

times, xc, rho = FV.load_density(args.case)
keep = [float(s) for s in args.times.split(',')]
sel = [j for j, t in enumerate(times) if any(abs(float(t) - k) < 1e-9 for k in keep)]
times = [times[j] for j in sel]
rho = [rho[j] for j in sel]
wanpm = np.load(os.path.join(OUT, f'{args.wanpm_tag}.npz'))
tnf = np.load(os.path.join(OUT, f'{args.tnf_tag}.npz'))

rows = [('Finite volume\n(reference)', None),
        ('WANPM', wanpm),
        ('t-NF', tnf)]

n = len(times)
fig, ax = plt.subplots(3, n, figsize=(3.0 * n, 8.6))
e = np.linspace(-2.2, 2.2, 221)
h_fv = xc[1] - xc[0]
inside = np.abs(xc) < 2.2
ext = [xc[inside][0] - h_fv / 2, xc[inside][-1] + h_fv / 2] * 2
for i, (lab, src) in enumerate(rows):
    for j, t in enumerate(times):
        h = ax[i, j]
        if src is None:
            h.imshow(rho[j][np.ix_(inside, inside)].T, origin='lower', extent=ext, cmap='viridis',
                     norm=matplotlib.colors.PowerNorm(0.5), interpolation='nearest')
        else:
            s = src[f'x_{float(t)}']
            h.hist2d(s[:, 0], s[:, 1], bins=[e, e], cmap='viridis',
                     norm=matplotlib.colors.PowerNorm(0.5))
        h.set_aspect('equal')
        h.plot([1, -1], [1, -1], 'r+', ms=8)
        h.set_xticks([]); h.set_yticks([])
        if i == 0:
            h.set_title(f't = {t:g}', fontsize=12)
        if j == 0:
            h.set_ylabel(lab, fontsize=12)

fig.suptitle(r'Transient double-well bifurcation, $\sigma=1$: WANPM vs. temporal normalizing flow',
             fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = args.out or os.path.join(OUT, f'{args.case}_compare_wanpm_tnf.png')
plt.savefig(out, dpi=140)
print('wrote', out)
