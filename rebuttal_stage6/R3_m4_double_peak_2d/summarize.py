"""One table for every Sec. 4.3 run, scored with dp2d_lab.metrics (includes the
stage-3 100k-epoch plane-wave runs for reference)."""
import os, glob, json
import numpy as np
import dp2d_lab as L

rows = []
for p in sorted(glob.glob(os.path.join(L.OUTDIR, '*.json'))):
    r = json.load(open(p)); c = r['cfg']
    rows.append((r['tag'], c['bank'].count(',') > 0, c['M'], c['epochs'], r))
old = os.path.join(L.HERE, '..', '..', 'rebuttal_stage3', 'R3_M5_optimizer_comparison',
                   'results_lowdim', 'dp2d_samples.npz')
if os.path.exists(old):
    S = np.load(old)
    for k in S.files:
        rows.append((f'stage3 {k}', False, 1000, 100000, L.metrics(S[k])))
print(f"{'run':26s} {'bank':6s} {'M':>5s} {'epochs':>7s} {'H':>6s} {'TV':>6s} {'upper':>6s} {'stray':>8s} {'std along/across (true %.3f/%.3f)' % (rows[0][4]['well_std_along_true'], rows[0][4]['well_std_across_true'])}")
for tag, mixed, M, E, r in rows:
    sd = '/'.join(f"{r.get(k + '_std_along', float('nan')):.3f},{r.get(k + '_std_across', float('nan')):.3f}" for k in ('upper', 'lower'))
    print(f"{tag:26s} {'mixed' if mixed else 'sin':6s} {M:5d} {E:7d} {r['hellinger']:6.3f} {r['tv']:6.3f} "
          f"{r['frac_upper']:6.3f} {r['stray_frac']:8.2e} {sd}")
