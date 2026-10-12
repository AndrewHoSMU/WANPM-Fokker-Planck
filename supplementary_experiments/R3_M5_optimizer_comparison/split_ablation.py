"""
Is per-layer learning-rate splitting -- not geometry-aware optimization --
what actually improves the Sec. 4.2 double-well result?

The Muon control study (muon_controls.py) showed that removing Muon's
Newton-Schulz orthogonalization while keeping its two-group learning-rate
split costs nothing: AdamW-split reached 0.102 +- 0.017 against Muon's
0.110 +- 0.025 over three seeds. That implicates the SPLIT, not the
orthogonalized update.

If that reading is right, the same trick should help the manuscript's own
optimizers. This script applies the identical split to plain Adam and plain
SGD:

    group 1 (hidden):  net.0.weight  (100x8)   -- learning rate swept
    group 2 (head):    net.0.bias, net.2.weight, net.2.bias
                                                -- held at that optimizer's
                                                   own best UNSPLIT lr

Fixing the head group at the optimizer's best unsplit setting means every
sweep starts exactly at its unsplit baseline (hidden lr == head lr is
algorithmically identical to no split at all), so any improvement as the
hidden lr rises is attributable to the split and nothing else.

    Adam: head lr 2e-3   (the manuscript's Sec. 4.2 value)
    SGD:  head lr 2e-2   (SGD's own best unsplit setting from the LR sweep)

Usage:
    python split_ablation.py --mode sweep   # phase 1, seed 0
    python split_ablation.py --mode seeds   # phase 2, best setting x 3 seeds
    python split_ablation.py --mode plot    # figures + summary
"""
import os, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import run_experiment as R
import muon_controls as MC

OUTDIR = MC.OUTDIR
SEEDS = MC.SEEDS

ADAM_HEAD_LR = 2e-3     # manuscript Sec. 4.2
SGD_HEAD_LR = 2e-2      # SGD's best unsplit lr (sweep.json / fine.json)


# Captured BEFORE the override below: falling back to MC.make_opt after
# rebinding it would call this function again and recurse forever.
_BASE_MAKE_OPT = MC.make_opt


def make_opt(variant, model, lr):
    """Extends muon_controls.make_opt with the split variants of Adam / SGD."""
    hidden, rest = MC.split_hidden_head(model)

    if variant == 'Adam-split':
        return torch.optim.Adam([
            dict(params=hidden, lr=lr),
            dict(params=rest, lr=ADAM_HEAD_LR),
        ])
    if variant == 'SGD-split':
        return torch.optim.SGD([
            dict(params=hidden, lr=lr),
            dict(params=rest, lr=SGD_HEAD_LR),
        ])
    if variant == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr)
    return _BASE_MAKE_OPT(variant, model, lr)


# muon_controls.run() resolves make_opt at module level, so overriding the
# module attribute makes the shared training loop use the variants above.
MC.make_opt = make_opt

SWEEPS = {
    # each grid STARTS at the corresponding head lr, i.e. at the unsplit baseline
    'Adam-split': [2e-3, 1e-2, 3e-2, 1e-1, 3e-1],
    'SGD-split':  [2e-2, 5e-2, 1e-1, 3e-1, 1e0],
}
REFERENCES = {
    'Adam': ADAM_HEAD_LR,     # unsplit baseline, manuscript setting
    'SGD': SGD_HEAD_LR,       # unsplit baseline
    'AdamW-split': 3e-2,      # from the Muon control study
    'Muon': 2e-1,             # from the Muon control study
}

SWEEP_JSON = 'split_sweep.json'
SEEDS_JSON = 'split_seeds.json'
SAMPLES_NPZ = 'split_samples.npz'


def mode_sweep(args):
    recs = []
    for variant, lr in REFERENCES.items():
        print(f"[ref]  {variant:12s} lr={lr:g}", flush=True)
        rec, _ = MC.run(variant, lr, seed=0)
        MC._show(rec); recs.append(rec); MC._save(recs, SWEEP_JSON)
    for variant, lrs in SWEEPS.items():
        for lr in lrs:
            print(f"[abl]  {variant:12s} hidden_lr={lr:g}", flush=True)
            rec, _ = MC.run(variant, lr, seed=0)
            MC._show(rec); recs.append(rec); MC._save(recs, SWEEP_JSON)
    return recs


def mode_seeds(args):
    sweep = json.load(open(os.path.join(OUTDIR, SWEEP_JSON)))
    best = {}
    for variant in list(REFERENCES) + list(SWEEPS):
        cand = [r for r in sweep if r['variant'] == variant and not r['diverged']
                and r['well_balance_err'] < 0.40]
        if not cand:
            cand = [r for r in sweep if r['variant'] == variant and not r['diverged']]
        if not cand:
            print(f"[seeds] {variant}: nothing usable, skipping")
            continue
        best[variant] = min(cand, key=lambda r: r['hellinger'])['lr']
        print(f"[seeds] {variant:12s} best lr = {best[variant]:g}")

    recs, samples = [], {}
    for variant, lr in best.items():
        for seed in SEEDS:
            print(f"[seeds] {variant:12s} lr={lr:g} seed={seed}", flush=True)
            rec, x = MC.run(variant, lr, seed=seed, keep_samples=(seed == 0))
            MC._show(rec); recs.append(rec); MC._save(recs, SEEDS_JSON)
            if seed == 0 and x is not None:
                samples[variant] = x.astype(np.float32)
    np.savez_compressed(os.path.join(OUTDIR, SAMPLES_NPZ), **samples)
    return recs


LABEL = {
    'Adam': 'Adam\n(unsplit, paper)',
    'Adam-split': 'Adam-split',
    'SGD': 'SGD\n(unsplit)',
    'SGD-split': 'SGD-split',
    'AdamW-split': 'AdamW-split',
    'Muon': 'Muon',
}
COLOR = {'Adam': '#1f77b4', 'Adam-split': '#08519c', 'SGD': '#969696',
         'SGD-split': '#525252', 'AdamW-split': '#e31a1c', 'Muon': '#6a3d9a'}
ORDER = ['Adam', 'Adam-split', 'SGD', 'SGD-split', 'AdamW-split', 'Muon']


def mode_plot(args):
    sweep = json.load(open(os.path.join(OUTDIR, SWEEP_JSON)))
    spath = os.path.join(OUTDIR, SEEDS_JSON)
    seeds = json.load(open(spath)) if os.path.exists(spath) else []
    npz = os.path.join(OUTDIR, SAMPLES_NPZ)
    samples = np.load(npz) if os.path.exists(npz) else None

    def ref(v):
        rs = [r for r in sweep if r['variant'] == v and not r['diverged']]
        return min(rs, key=lambda r: r['hellinger'])['hellinger'] if rs else np.nan

    adam_ref, sgd_ref, muon_ref = ref('Adam'), ref('SGD'), ref('Muon')

    # ---- Figure 1: split sweeps vs their own unsplit baselines ----
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for v in SWEEPS:
        rs = sorted([r for r in sweep if r['variant'] == v], key=lambda r: r['lr'])
        ax.plot([r['lr'] for r in rs],
                [r['hellinger'] if not r['diverged'] else np.nan for r in rs],
                'o-', color=COLOR[v], lw=2.0, ms=6.5, label=LABEL[v])
    ax.axhline(adam_ref, color=COLOR['Adam'], ls=':', lw=2,
               label=f'Adam unsplit baseline ({adam_ref:.3f})')
    ax.axhline(sgd_ref, color=COLOR['SGD'], ls=':', lw=2,
               label=f'SGD unsplit baseline ({sgd_ref:.3f})')
    ax.axhline(muon_ref, color=COLOR['Muon'], ls='--', lw=2,
               label=f'Muon ({muon_ref:.3f})')
    ax.set_xscale('log')
    ax.set_xlabel('hidden-group learning rate  '
                  '(head group fixed: Adam 2e-3, SGD 2e-2)')
    ax.set_ylabel('Hellinger distance to analytical solution')
    ax.set_title('Does a per-layer learning-rate split help the paper\'s own optimizers?\n'
                 'Each curve starts at its unsplit baseline (hidden lr = head lr)',
                 fontsize=11.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p1 = os.path.join(OUTDIR, 'split_ablation_sweep.png')
    plt.savefig(p1, dpi=150); plt.close()
    print(f"  wrote {p1}")

    if not seeds:
        print("  (no split_seeds.json yet -- run --mode seeds for figure 2)")
        return

    order = [v for v in ORDER if any(r['variant'] == v for r in seeds)]
    grid = np.linspace(R.GRID_LO, R.GRID_HI, 800)
    pt = R.true_pdf(grid)
    n = len(order)
    fig = plt.figure(figsize=(3.5 * n, 8.2))
    for i, v in enumerate(order):
        ax = fig.add_subplot(2, n, i + 1)
        ax.plot(grid, pt, 'k-', lw=2.1, zorder=3,
                label='analytical' if i == 0 else None)
        if samples is not None and v in samples.files:
            ax.hist(samples[v], bins=150, range=(R.GRID_LO, R.GRID_HI),
                    density=True, color=COLOR[v], alpha=0.62,
                    label='learned' if i == 0 else None)
        h0 = [r for r in seeds if r['variant'] == v and r['seed'] == 0
              and not r['diverged']]
        sub = f"Hellinger = {h0[0]['hellinger']:.3f}" if h0 else 'failed'
        lr0 = [r for r in seeds if r['variant'] == v]
        lrtxt = f"lr={lr0[0]['lr']:g}" if lr0 else ''
        ax.set_title(f"{LABEL[v]}  {lrtxt}\n{sub}", fontsize=9)
        ax.set_xlim(R.GRID_LO, R.GRID_HI); ax.set_ylim(0, 1.6)
        ax.set_xlabel('$x$'); ax.grid(alpha=0.25)
        if i == 0:
            ax.set_ylabel('density'); ax.legend(fontsize=7.5)

    ax = fig.add_subplot(2, 1, 2)
    means, sds, labs, cols = [], [], [], []
    for v in order:
        vals = np.array([r['hellinger'] for r in seeds
                         if r['variant'] == v and not r['diverged']
                         and r['well_balance_err'] < 0.40], dtype=float)
        nbad = len([r for r in seeds if r['variant'] == v
                    and (r['diverged'] or r['well_balance_err'] >= 0.40)])
        means.append(vals.mean() if len(vals) else np.nan)
        sds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        labs.append(LABEL[v].replace('\n', ' ') + (f'\n[{nbad}/3 failed]' if nbad else '\n[3/3 ok]'))
        cols.append(COLOR[v])
    ax.bar(range(len(order)), means, yerr=sds, capsize=5, color=cols, alpha=0.85)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labs, fontsize=8.5)
    ax.set_ylabel('Hellinger  (mean $\\pm$ sd, successful runs)')
    ax.set_title(f'Seed-averaged accuracy at each variant\'s best learning rate '
                 f'(seeds {SEEDS}; collapsed/diverged runs excluded from the mean '
                 f'and counted in the label)', fontsize=10.5)
    ax.grid(alpha=0.3, axis='y')
    for i, (m, s) in enumerate(zip(means, sds)):
        if np.isfinite(m):
            ax.text(i, m + s + 0.006, f'{m:.3f}', ha='center', fontsize=9.5)
    plt.tight_layout()
    p2 = os.path.join(OUTDIR, 'split_ablation_result.png')
    plt.savefig(p2, dpi=150); plt.close()
    print(f"  wrote {p2}")

    print()
    print(f"{'variant':16s} {'best lr':>9} {'hellinger (mean+-sd)':>24} {'ok':>5}")
    print('-' * 62)
    for v, m, s in zip(order, means, sds):
        rs = [r for r in seeds if r['variant'] == v]
        lr = rs[0]['lr'] if rs else float('nan')
        ok = len([r for r in rs if not r['diverged'] and r['well_balance_err'] < 0.40])
        print(f"{LABEL[v].replace(chr(10),' '):16s} {lr:9g} {m:>12.4f} +- {s:.4f} {ok:>3}/3")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['sweep', 'seeds', 'plot'])
    args = ap.parse_args()
    print(f"device: {R.device}")
    {'sweep': mode_sweep, 'seeds': mode_seeds, 'plot': mode_plot}[args.mode](args)
