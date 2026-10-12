"""Re-score the saved WANPM and t-NF samples against the finite-volume reference (reference_fv.py)
instead of the Euler-Maruyama one. Same metrics.py, same saved samples (results/<tag>.npz); the
noise floor is one set of finite-volume samples against an independent second set. t-NF is also
scored after discarding samples with |x|_inf >= 4 (its spline tails), as in RESULTS_SUMMARY.md.

    python metrics_vs_fv.py                 # results/E_metrics_fv.json
"""
import os, json, argparse
import numpy as np
import metrics as Mx

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results')

ap = argparse.ArgumentParser()
ap.add_argument('--case', default='E')
ap.add_argument('--wanpm-tag', default='E_adam_saddle')
ap.add_argument('--tnf-tag', default='E_tnf_L4spl')
args = ap.parse_args()

wanpm = np.load(os.path.join(OUT, f'{args.wanpm_tag}.npz'))
tnf = np.load(os.path.join(OUT, f'{args.tnf_tag}.npz'))
res = {}
for kind in ('fv', 'mc'):
    times, a, b = Mx.load_reference(args.case, kind)
    rows = {}
    for i, t in enumerate(times):
        xw, xt = wanpm[f'x_{float(t)}'], tnf[f'x_{float(t)}']
        keep = (np.abs(xt) < 4).all(1)
        rows[str(float(t))] = dict(wanpm=Mx.compare(xw, a[i]), tnf=Mx.compare(xt, a[i]),
                                   tnf_trimmed=Mx.compare(xt[keep], a[i]), tnf_kept_frac=float(keep.mean()),
                                   floor=Mx.compare(b[i], a[i]))
    res[kind] = rows
    print(f'reference: {kind}')
    print('  t      WANPM sw2  pi+     t-NF sw2 (trimmed)  pi+ (all / trimmed)  kept    floor sw2')
    for t, r in rows.items():
        print(f'  {t:<6} {r["wanpm"]["sw2"]:.4f}     {r["wanpm"]["pi_plus"]:.3f}   {r["tnf_trimmed"]["sw2"]:.4f}'
              f'              {r["tnf"]["pi_plus"]:.3f} / {r["tnf_trimmed"]["pi_plus"]:.3f}        '
              f'{r["tnf_kept_frac"]:.3f}   {r["floor"]["sw2"]:.4f}')
with open(os.path.join(OUT, f'{args.case}_metrics_fv.json'), 'w') as fh:
    json.dump(dict(wanpm_tag=args.wanpm_tag, tnf_tag=args.tnf_tag, **res), fh, indent=1)
