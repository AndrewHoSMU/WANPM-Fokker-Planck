"""
Is the whole effect just Adam's beta2?

Trail of reasoning that led here:
  1. Muon beat Adam ~2x on the Sec. 4.2 double well (0.091 vs 0.210).
  2. Control A removed Muon's orthogonalization but kept its two-group
     learning-rate split, and lost nothing (0.102 vs 0.110 over 3 seeds).
     Conclusion at the time: the SPLIT is what matters.
  3. But applying the same split to plain Adam did NOT help at any hidden
     learning rate (best 0.210 = its own unsplit baseline), while Control A
     at the identical split and identical learning rates reached 0.099.

Control A was built as `torch.optim.AdamW(..., betas=(0.9, 0.95),
weight_decay=0.0)` -- betas copied from Muon's internal AdamW. With
weight_decay=0 AdamW is algorithmically identical to Adam, so Control A and
Adam-split differ in EXACTLY ONE hyperparameter:

    beta2 = 0.95   (Control A / Muon's internal AdamW)
    beta2 = 0.999  (PyTorch default, and the manuscript's Adam)

This script tests whether beta2 alone accounts for the improvement, with and
without the parameter-group split, so the two candidate causes are fully
crossed:

    unsplit  x  beta2 in {0.999, 0.99, 0.95, 0.9}   (lr swept)
    split    x  beta2 in {0.999, 0.95}              (hidden lr swept)

If unsplit Adam with beta2=0.95 reaches ~0.10 on its own, then neither Muon,
nor the orthogonalization, nor the learning-rate split is responsible -- the
entire effect is one hyperparameter in the manuscript's existing optimizer.

Usage:
    python beta2_ablation.py --mode sweep
    python beta2_ablation.py --mode seeds
    python beta2_ablation.py --mode plot
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
HEAD_LR = 2e-3

SWEEP_JSON = 'beta2_sweep.json'
SEEDS_JSON = 'beta2_seeds.json'
SAMPLES_NPZ = 'beta2_samples.npz'

_BASE_MAKE_OPT = MC.make_opt


def _parse(variant):
    """'adam-b0.95' -> (False, 0.95);  'adamsplit-b0.95' -> (True, 0.95)"""
    kind, b = variant.split('-b')
    return kind == 'adamsplit', float(b)


def make_opt(variant, model, lr):
    if variant.startswith('adam-b') or variant.startswith('adamsplit-b'):
        split, beta2 = _parse(variant)
        betas = (0.9, beta2)
        if not split:
            return torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
        hidden, rest = MC.split_hidden_head(model)
        return torch.optim.Adam([
            dict(params=hidden, lr=lr),
            dict(params=rest, lr=HEAD_LR),
        ], betas=betas)
    return _BASE_MAKE_OPT(variant, model, lr)


MC.make_opt = make_opt

# unsplit: sweep the single lr.  split: sweep the hidden lr, head fixed 2e-3.
UNSPLIT_LRS = [5e-4, 2e-3, 1e-2, 3e-2, 1e-1]
SPLIT_LRS = [2e-3, 1e-2, 3e-2, 1e-1]
BETA2_UNSPLIT = [0.999, 0.99, 0.95, 0.9]
BETA2_SPLIT = [0.999, 0.95]


def _variants():
    out = []
    for b in BETA2_UNSPLIT:
        out += [(f'adam-b{b}', lr) for lr in UNSPLIT_LRS]
    for b in BETA2_SPLIT:
        out += [(f'adamsplit-b{b}', lr) for lr in SPLIT_LRS]
    return out


def mode_sweep(args):
    recs = []
    for variant, lr in _variants():
        print(f"[b2]  {variant:18s} lr={lr:g}", flush=True)
        rec, _ = MC.run(variant, lr, seed=0)
        MC._show(rec); recs.append(rec); MC._save(recs, SWEEP_JSON)
    return recs


def mode_seeds(args):
    sweep = json.load(open(os.path.join(OUTDIR, SWEEP_JSON)))
    names = sorted({r['variant'] for r in sweep})
    best = {}
    for v in names:
        cand = [r for r in sweep if r['variant'] == v and not r['diverged']
                and r['well_balance_err'] < 0.40]
        if not cand:
            print(f"[seeds] {v}: no non-collapsed lr, skipping")
            continue
        best[v] = min(cand, key=lambda r: r['hellinger'])['lr']
        print(f"[seeds] {v:18s} best lr = {best[v]:g}")

    recs, samples = [], {}
    for v, lr in best.items():
        for seed in SEEDS:
            print(f"[seeds] {v:18s} lr={lr:g} seed={seed}", flush=True)
            rec, x = MC.run(v, lr, seed=seed, keep_samples=(seed == 0))
            MC._show(rec); recs.append(rec); MC._save(recs, SEEDS_JSON)
            if seed == 0 and x is not None:
                samples[v] = x.astype(np.float32)
    np.savez_compressed(os.path.join(OUTDIR, SAMPLES_NPZ), **samples)
    return recs


def mode_plot(args):
    sweep = json.load(open(os.path.join(OUTDIR, SWEEP_JSON)))
    spath = os.path.join(OUTDIR, SEEDS_JSON)
    seeds = json.load(open(spath)) if os.path.exists(spath) else []

    cmap = plt.get_cmap('viridis')
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), sharey=True)

    ax = axes[0]
    for i, b in enumerate(BETA2_UNSPLIT):
        rs = sorted([r for r in sweep if r['variant'] == f'adam-b{b}'],
                    key=lambda r: r['lr'])
        ax.plot([r['lr'] for r in rs],
                [r['hellinger'] if not r['diverged'] else np.nan for r in rs],
                'o-', lw=2, ms=6, color=cmap(i / max(1, len(BETA2_UNSPLIT) - 1)),
                label=rf'$\beta_2$ = {b}')
    ax.set_xscale('log'); ax.set_xlabel('learning rate')
    ax.set_ylabel('Hellinger distance to analytical solution')
    ax.set_title('Plain (unsplit) Adam, varying $\\beta_2$', fontsize=11.5)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    ax = axes[1]
    for i, b in enumerate(BETA2_SPLIT):
        rs = sorted([r for r in sweep if r['variant'] == f'adamsplit-b{b}'],
                    key=lambda r: r['lr'])
        ax.plot([r['lr'] for r in rs],
                [r['hellinger'] if not r['diverged'] else np.nan for r in rs],
                's--', lw=2, ms=7, color=cmap(i / max(1, len(BETA2_SPLIT) - 1)),
                label=rf'split, $\beta_2$ = {b}')
    ax.set_xscale('log')
    ax.set_xlabel('hidden-group learning rate (head fixed at 2e-3)')
    ax.set_title('Adam with the hidden/head split, varying $\\beta_2$', fontsize=11.5)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    for a in axes:
        a.axhline(0.210, color='#1f77b4', ls=':', lw=1.8)
        a.axhline(0.091, color='#6a3d9a', ls='--', lw=1.8)
    axes[0].text(UNSPLIT_LRS[0], 0.215, ' manuscript Adam (0.210)',
                 fontsize=8, color='#1f77b4', va='bottom')
    axes[0].text(UNSPLIT_LRS[0], 0.096, ' Muon (0.091)',
                 fontsize=8, color='#6a3d9a', va='bottom')

    fig.suptitle('Is the double-well improvement just Adam\'s $\\beta_2$?\n'
                 'main.tex Sec. 4.2, seed 0, everything else identical',
                 fontsize=12.5)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    p1 = os.path.join(OUTDIR, 'beta2_ablation.png')
    plt.savefig(p1, dpi=150); plt.close()
    print(f"  wrote {p1}")

    if seeds:
        print()
        print(f"{'variant':20s} {'best lr':>9} {'hellinger (mean+-sd)':>24} {'ok':>5}")
        print('-' * 64)
        agg = {}
        for v in sorted({r['variant'] for r in seeds}):
            rs = [r for r in seeds if r['variant'] == v]
            good = [r['hellinger'] for r in rs
                    if not r['diverged'] and r['well_balance_err'] < 0.40]
            m = float(np.mean(good)) if good else float('nan')
            s = float(np.std(good, ddof=1)) if len(good) > 1 else 0.0
            agg[v] = (rs[0]['lr'], m, s, len(good), len(rs))
            print(f"{v:20s} {rs[0]['lr']:9g} {m:>12.4f} +- {s:.4f} "
                  f"{len(good):>2}/{len(rs)}")
        with open(os.path.join(OUTDIR, 'beta2_summary.json'), 'w') as fh:
            json.dump(agg, fh, indent=2)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['sweep', 'seeds', 'plot'])
    args = ap.parse_args()
    print(f"device: {R.device}")
    {'sweep': mode_sweep, 'seeds': mode_seeds, 'plot': mode_plot}[args.mode](args)
