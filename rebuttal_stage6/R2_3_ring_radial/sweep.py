"""
Parallel sweep driver for ring_lab.  Each (tag, cfg) is one training run; results
are cached in results/<tag>.json so re-running a stage only fills in the gaps.

    python sweep.py --stage baseline --workers 6
    python sweep.py --summary baseline
"""
import os, sys, json, argparse, subprocess, itertools
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# the three candidate problems (see verify_reference.py)
P1 = dict(kappa=1.0, a=1.0)     # drift = -grad V of the stated V; true = manuscript formula
P2 = dict(kappa=2.0, a=1.0)     # drift as coded, generator (sigma^2/2) Laplacian
P3 = dict(kappa=2.0, a=0.5)     # drift as coded, a_ij = sigma^2/2 literally
PROB = {'P1': P1, 'P2': P2, 'P3': P3}

MULTI = 'sin:120:1,gauss:16:4,mexhat:16:4,morlet:16:4,sech2:16:4,herm1:16:4'


def stage_configs(stage):
    runs = []

    def add(tag, seeds=(0, 1), **cfg):
        for s in seeds:
            runs.append((f'{tag}_s{s}', dict(cfg, seed=s)))

    if stage == 'baseline':
        for pn, p in PROB.items():
            add(f'{pn}_base', seeds=(0, 1, 2), **p)

    elif stage == 'bank':        # which test functions?  (P1, default training)
        B = {'sin2': 'sin:200:2', 'sin3': 'sin:200:3',
             'sin1+sin3': 'sin:100:1,sin:100:3',
             'g4_25': 'sin:150:1,gauss:50:4', 'g4_40': 'sin:120:1,gauss:80:4',
             'g2_40': 'sin:120:1,gauss:80:2', 'g8_40': 'sin:120:1,gauss:80:8',
             'mex4_40': 'sin:120:1,mexhat:80:4', 'herm4_40': 'sin:120:1,herm1:80:4',
             'sech4_40': 'sin:120:1,sech2:80:4', 'morl4_40': 'sin:120:1,morlet:80:4',
             'multi': MULTI}
        for k, b in B.items():
            add(f'P1_{k}', bank=b, **P1)

    elif stage == 'anneal':      # sigma-annealing curriculum (P1)
        for a0 in (2.0, 4.0, 9.0):
            add(f'P1_base_an{a0:g}', anneal_from=a0, **P1)
            add(f'P1_g4_40_an{a0:g}', bank='sin:120:1,gauss:80:4', anneal_from=a0, **P1)

    elif stage == 'knobs':       # ordinary training knobs on the manuscript bank (P1)
        add('P1_base_M4000', M=4000, **P1)
        add('P1_base_lrdecay', lr_decay_to=0.02, **P1)
        add('P1_base_d16', d_base=16, **P1)
        add('P1_base_h64x2', hidden=[64, 64], **P1)
        add('P1_base_muon', opt='muon', lr=0.1, **P1)
        add('P1_base_ep60k', epochs=60000, **P1)
    elif stage == 'combo':       # generator lr split x test-function bank (P1)
        MU = dict(opt='muon', lr=0.1)
        add('P1_muon_g4_40', bank='sin:120:1,gauss:80:4', **MU, **P1)
        add('P1_muon_herm4_40', bank='sin:120:1,herm1:80:4', **MU, **P1)
        add('P1_muon_mex4_40', bank='sin:120:1,mexhat:80:4', **MU, **P1)
        add('P1_muon_rbf2_40', bank='sin:120:1,rbf:80:2', **MU, **P1)
        add('P1_rbf2_40', bank='sin:120:1,rbf:80:2', **P1)
        add('P1_rbf1_40', bank='sin:120:1,rbf:80:1', **P1)
        add('P1_muon_g4_rbf2', bank='sin:100:1,gauss:50:4,rbf:50:2', **MU, **P1)
        add('P1_muon_lr0.03', opt='muon', lr=0.03, **P1)
        add('P1_muon_lr0.3', opt='muon', lr=0.3, **P1)
        add('P1_adamsplit_lr0.1', opt='adam_split', lr=0.1, **P1)
        add('P1_muon_h64x2', hidden=[64, 64], **MU, **P1)
        add('P1_muon_sin1+sin3', bank='sin:100:1,sin:100:3', **MU, **P1)
    elif stage == 'final':       # more seeds for the finalists + two polish variants
        MU = dict(opt='muon', lr=0.1)
        add('P1_base', seeds=(3,), **P1)
        add('P1_base_muon', seeds=(2, 3), **MU, **P1)
        add('P1_muon_herm4_40', seeds=(2, 3), bank='sin:120:1,herm1:80:4', **MU, **P1)
        add('P1_herm4_40', seeds=(2, 3), bank='sin:120:1,herm1:80:4', **P1)
        add('P1_mex4_40', seeds=(2, 3), bank='sin:120:1,mexhat:80:4', **P1)
        add('P1_adamsplit_lr0.1', seeds=(2, 3), opt='adam_split', lr=0.1, **P1)
        add('P1_muon_an4', anneal_from=4.0, **MU, **P1)
        add('P1_muon_M2000', M=2000, **MU, **P1)
        add('P1_muon_ep40k_lrdecay', epochs=40000, lr_decay_to=0.05, **MU, **P1)
        add('P3_muon', seeds=(0, 1, 2), **MU, **P3)
    elif stage == 'smooth':      # keep the wavelet bank's smooth 2-D density AND lose the interior leak
        W = dict(bank='sin:120:1,mexhat:80:4')
        add('P1_mex4_h64x2', hidden=[64, 64], **W, **P1)
        add('P1_mex4_d16', d_base=16, **W, **P1)
        add('P1_mex4_d16_h64x2', d_base=16, hidden=[64, 64], **W, **P1)
        add('P1_mex4_ep60k', epochs=60000, **W, **P1)
        add('P1_mex4_muon0.03', opt='muon', lr=0.03, **W, **P1)
        add('P1_mex4_adamsplit0.03', opt='adam_split', lr=0.03, **W, **P1)
        add('P1_mex+herm', bank='sin:120:1,mexhat:40:4,herm1:40:4', **P1)
    elif stage == 'rbfamp':      # fully localized wavelets with enough amplitude to be heard
        add('P1_mex4_rbf2x5', bank='sin:120:1,mexhat:40:4,rbf:40:2:5', **P1)
        add('P1_mex4_rbf2x10', bank='sin:120:1,mexhat:40:4,rbf:40:2:10', **P1)
        add('P1_rbf2x5_40', bank='sin:120:1,rbf:80:2:5', **P1)
        add('P1_mex4_rbf1.5x5_sp2', bank='sin:120:1,mexhat:40:4,rbf:40:1.5:5', spread=2.0, **P1)
    elif stage == 'rbfamp2':     # push the amplitude of the localized family
        add('P1_mex4_rbf2x20', seeds=(0, 1, 2), bank='sin:120:1,mexhat:40:4,rbf:40:2:20', **P1)
        add('P1_mex4_rbf2x40', seeds=(0, 1, 2), bank='sin:120:1,mexhat:40:4,rbf:40:2:40', **P1)
        add('P1_mex4_rbf2x10', seeds=(2, 3), bank='sin:120:1,mexhat:40:4,rbf:40:2:10', **P1)
    elif stage == 'focus':       # split learning rate + wavelets only
        MU = dict(opt='muon', lr=0.1)
        H = 'sin:120:1,herm1:80:4'
        add('P1_sw_advlr0.1', bank=H, lr_test=0.1, **MU, **P1)
        add('P1_sw_herm4+8', bank='sin:120:1,herm1:40:4,herm1:40:8', **MU, **P1)
        add('P1_sw_herm+mex', bank='sin:120:1,herm1:40:4,mexhat:40:4', **MU, **P1)
        add('P1_sw_lr0.05', bank=H, opt='muon', lr=0.05, **P1)
        add('P1_sw_herm_rbf', bank='sin:120:1,herm1:40:4,rbf:40:2:10', **MU, **P1)
    else:
        raise ValueError(stage)
    return runs


def launch(item):
    tag, cfg = item
    out = os.path.join(HERE, 'results', f'{tag}.json')
    if not os.path.exists(out):
        subprocess.run([PY, os.path.join(HERE, 'ring_lab.py'), '--tag', tag,
                        '--cfg', json.dumps(cfg)], cwd=HERE, capture_output=True)
    if os.path.exists(out):
        import ring_lab as RL
        print(RL.fmt(json.load(open(out))), flush=True)
    else:
        print(f'{tag}: FAILED', flush=True)


KEYS = ['r_std_ratio', 'r_iqr_ratio', 'peak_ratio', 'radial_hellinger', 'hellinger',
        'angular_tv', 'frac_inside']


def summary(prefixes):
    import glob
    recs = {}
    for f in glob.glob(os.path.join(HERE, 'results', '*_s*.json')):
        r = json.load(open(f))
        if 'tag' not in r:
            continue
        base = r['tag'].rsplit('_s', 1)[0]
        if prefixes and not any(base.startswith(p) for p in prefixes):
            continue
        recs.setdefault(base, []).append(r)
    print(f"{'config':26s} n  " + ' '.join(f'{k[:11]:>13s}' for k in KEYS))
    rows = []
    for base, rs in recs.items():
        ok = [r for r in rs if not r.get('diverged')]
        if not ok:
            print(f'{base:26s} all diverged'); continue
        rows.append((np.mean([r['radial_hellinger'] for r in ok]), base, ok, len(rs)))
    for _, base, ok, n in sorted(rows):
        cells = []
        for k in KEYS:
            v = [r[k] for r in ok]
            cells.append(f'{np.mean(v):6.3f}±{np.std(v):5.3f}')
        print(f'{base:26s} {len(ok)}/{n} ' + ' '.join(f'{c:>13s}' for c in cells))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--summary', nargs='*')
    args = ap.parse_args()
    if args.stage:
        runs = stage_configs(args.stage)
        with ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(launch, runs))
    if args.summary is not None:
        summary(args.summary)
