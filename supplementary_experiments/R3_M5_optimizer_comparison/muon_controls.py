"""
Control study: WHY does Muon win on the Sec. 4.2 double-well?

The Muon run differs from a plain-Adam run in TWO ways at once, so its result
cannot be attributed to the orthogonalized update without controls:

  (i)  the update rule on the hidden matrix is orthogonalized (Newton-Schulz)
  (ii) the network is split into two parameter groups with a 100:1 learning
       rate ratio -- Muon lr=0.2 on net.0.weight, internal AdamW lr=2e-3 on
       net.0.bias, net.2.weight (the output layer) and net.2.bias

CONTROL A -- "split, no orthogonalization".
    Identical two-group split, identical learning rates, and AdamW configured
    exactly as Muon's own internal AdamW (betas=(0.9,0.95), weight_decay=0),
    applied to the hidden matrix as well. The ONLY thing removed is the
    orthogonalization. Sweeping the hidden-group lr tells us how much of
    Muon's advantage is just the learning-rate split.
      - if its best ~= Muon's 0.091  -> the split explains the result
      - if its best ~= Adam's 0.20   -> the orthogonalization does the work

CONTROL B -- "orthogonalization, different head lr".
    Muon held at lr=0.2 on the hidden matrix while its internal AdamW lr is
    varied. Tests whether the result is sensitive to the untuned 2e-3 head
    learning rate inherited from the paper.

CONTROL C -- "fair SGD".
    The main comparison gave plain SGD momentum=0 while Shampoo got 0.9 and
    Muon 0.95, which understates SGD. This re-runs SGD with momentum=0.9.

Everything else matches main.tex Sec. 4.2 exactly (see muon_only.py).

Usage:
    python muon_controls.py --mode sweep    # phase 1: lr sweeps, seed 0
    python muon_controls.py --mode seeds    # phase 2: best setting x 3 seeds
    python muon_controls.py --mode plot     # figures + summary
"""
import os, json, time, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pytorch_optimizer as po
import run_experiment as R

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'results')
os.makedirs(OUTDIR, exist_ok=True)

HEAD_LR = R.LR_GEN_PAPER      # 2e-3, the lr Muon's internal AdamW inherits
MUON_BETAS = (0.9, 0.95)      # pytorch-optimizer Muon's adamw_betas default
SEEDS = [0, 1, 2]


# ---------------------------------------------------------------------------
# parameter split -- must match run_experiment._split_muon_groups exactly
# ---------------------------------------------------------------------------
def split_hidden_head(model):
    """(hidden 2-D weights, everything else incl. the output layer)."""
    named = list(model.named_parameters())
    linear_idx = sorted({int(n.split('.')[1]) for n, _ in named})
    head_idx = linear_idx[-1]
    hidden, rest = [], []
    for n, p in named:
        idx = int(n.split('.')[1])
        (hidden if (p.ndim >= 2 and idx != head_idx) else rest).append(p)
    return hidden, rest


def make_opt(variant, model, lr):
    """lr means: hidden-group lr for split variants, plain lr otherwise."""
    if variant == 'Muon':
        return po.Muon([
            dict(params=split_hidden_head(model)[0], lr=lr,
                 weight_decay=0.0, use_muon=True),
            dict(params=split_hidden_head(model)[1], lr=HEAD_LR,
                 weight_decay=0.0, use_muon=False),
        ], weight_decay=0.0)

    if variant == 'AdamW-split':
        # CONTROL A: same split, same lrs, same betas as Muon's internal
        # AdamW -- orthogonalization removed, nothing else changed.
        hidden, rest = split_hidden_head(model)
        return torch.optim.AdamW([
            dict(params=hidden, lr=lr),
            dict(params=rest, lr=HEAD_LR),
        ], betas=MUON_BETAS, weight_decay=0.0)

    if variant == 'Muon-headlr':
        # CONTROL B: Muon fixed at 0.2 on hidden, `lr` varies the HEAD lr.
        return po.Muon([
            dict(params=split_hidden_head(model)[0], lr=2e-1,
                 weight_decay=0.0, use_muon=True),
            dict(params=split_hidden_head(model)[1], lr=lr,
                 weight_decay=0.0, use_muon=False),
        ], weight_decay=0.0)

    if variant == 'SGD-momentum':
        # CONTROL C: SGD with momentum 0.9 (the main sweep used momentum 0).
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    if variant == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr)

    raise ValueError(variant)


SWEEPS = {
    # CONTROL A: does a 100:1 hidden/head lr split alone reproduce Muon?
    'AdamW-split':  [2e-3, 1e-2, 3e-2, 1e-1, 2e-1],
    # CONTROL B: is the result sensitive to the untuned head lr?
    'Muon-headlr':  [5e-4, 2e-3, 1e-2, 5e-2],
    # CONTROL C: SGD given the momentum the other methods got
    # 1e-3/5e-3/1e-2 collapse and 2e-2 diverges, so the usable band (if any)
    # is between them. Momentum 0.9 inflates the effective step by ~1/(1-m)=10x,
    # so plain SGD's working window of 2e-2..3e-2 maps to ~2e-3..3e-3 here.
    'SGD-momentum': [1e-3, 2e-3, 3e-3, 5e-3, 7e-3, 1e-2, 2e-2, 5e-2],
}
REFERENCES = {'Muon': 2e-1, 'Adam': 2e-3}   # re-run so every number is same-code


def run(variant, lr, seed=0, n_epochs=None, keep_samples=False):
    n_epochs = n_epochs or R.N_EPOCHS
    torch.manual_seed(seed)
    np.random.seed(seed)
    pf = R.SteadyPushforward(d_base=R.D_BASE, n_out=1, hidden=R.HIDDEN).to(R.device)
    tf = R.SteadyTestFunctions(n_in=1, K=R.K_TEST).to(R.device)
    gen_opt = make_opt(variant, pf, lr)
    test_opt = R.make_test_opt('SGD', tf.parameters(), R.LR_TEST_PAPER)

    cur = [R.SIGMA2]

    def sigma2_cur(x):
        return cur[0] * torch.ones(x.shape[0], 1, device=x.device)

    def loss_fn():
        return R.steady_loss(pf, tf, M=R.BATCH_M, drift_fn=R.drift_fn,
                             sigma2_fn=sigma2_cur,
                             base_sampler=R.gaussian_base_sampler, device=R.device)

    losses = np.empty(n_epochs)
    diverged = False
    t0 = time.time()
    for epoch in range(n_epochs):
        cur[0] = R.sigma2_at_epoch(epoch, n_epochs, anneal=True)
        if epoch % R.ADV_EVERY == 0 and epoch > 0:
            for _ in range(R.ADV_STEPS):
                l, _ = loss_fn()
                test_opt.zero_grad(); (-l).backward(); test_opt.step()
        l, _ = loss_fn()
        gen_opt.zero_grad(); l.backward(); gen_opt.step()
        losses[epoch] = l.item()
        if not np.isfinite(losses[epoch]):
            diverged = True
            losses[epoch:] = np.nan
            break
    dt = time.time() - t0

    with torch.no_grad():
        x = pf(torch.randn(200000, R.D_BASE, device=R.device)).cpu().numpy().ravel()

    rec = {'variant': variant, 'lr': lr, 'seed': seed, 'diverged': diverged,
           'train_time_s': dt}
    if diverged:
        rec.update({k: float('nan') for k in
                    ['hellinger', 'tv', 'well_balance_err', 'peak_overshoot',
                     'bridge_ratio', 'std_learned']})
    else:
        rec.update(_metrics(x))
    return (rec, x) if keep_samples else (rec, None)


def _metrics(x):
    edges = np.linspace(R.GRID_LO, R.GRID_HI, R.N_BINS + 1)
    c = 0.5 * (edges[:-1] + edges[1:])
    dx = edges[1] - edges[0]
    ph, _ = np.histogram(x, bins=edges, density=True)
    ph = ph * float(((x >= R.GRID_LO) & (x <= R.GRID_HI)).mean())
    pt = R.true_pdf(c)
    pt = pt / (pt.sum() * dx)
    bridge_true = float(pt[np.abs(c) < 0.5].sum() * dx)
    frac_right = float((x > 0).mean())
    return {
        'hellinger': float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(ph * pt)) * dx))),
        'tv': float(0.5 * np.sum(np.abs(ph - pt)) * dx),
        'well_balance_err': abs(frac_right - 0.5),
        'peak_overshoot': float(ph.max() / pt.max()),
        'bridge_ratio': float((np.abs(x) < 0.5).mean() / bridge_true),
        'std_learned': float(x.std()),
    }


def _save(recs, fname):
    with open(os.path.join(OUTDIR, fname), 'w') as fh:
        json.dump(recs, fh, indent=2)


def _show(rec):
    if rec['diverged']:
        print(f"    DIVERGED  ({rec['train_time_s']:.0f}s)", flush=True)
    else:
        print(f"    hellinger={rec['hellinger']:.4f}  peak={rec['peak_overshoot']:.2f}x  "
              f"bridge={rec['bridge_ratio']:.1f}x  bal={rec['well_balance_err']:.4f}  "
              f"({rec['train_time_s']:.0f}s)", flush=True)


def mode_sweep(args):
    recs = []
    for variant, lr in REFERENCES.items():
        print(f"[ref]  {variant:14s} lr={lr:g}", flush=True)
        rec, _ = run(variant, lr, seed=0)
        _show(rec); recs.append(rec); _save(recs, 'controls_sweep.json')
    for variant, lrs in SWEEPS.items():
        for lr in lrs:
            print(f"[ctrl] {variant:14s} lr={lr:g}", flush=True)
            rec, _ = run(variant, lr, seed=0)
            _show(rec); recs.append(rec); _save(recs, 'controls_sweep.json')
    return recs


def mode_seeds(args):
    sweep = json.load(open(os.path.join(OUTDIR, 'controls_sweep.json')))
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
        print(f"[seeds] {variant:14s} best lr = {best[variant]:g}")

    recs, samples = [], {}
    for variant, lr in best.items():
        for seed in SEEDS:
            print(f"[seeds] {variant:14s} lr={lr:g} seed={seed}", flush=True)
            rec, x = run(variant, lr, seed=seed, keep_samples=(seed == 0))
            _show(rec); recs.append(rec); _save(recs, 'controls_seeds.json')
            if seed == 0 and x is not None:
                samples[variant] = x.astype(np.float32)
    np.savez_compressed(os.path.join(OUTDIR, 'controls_samples.npz'), **samples)
    return recs


LABEL = {
    'Muon': 'Muon\n(orthogonalized + split)',
    'AdamW-split': 'CONTROL A: AdamW-split\n(split only, no orthog.)',
    'Muon-headlr': 'CONTROL B: Muon\n(head lr retuned)',
    'SGD-momentum': 'CONTROL C: SGD\n(momentum 0.9)',
    'Adam': 'Adam\n(paper baseline)',
}
COLOR = {'Muon': '#6a3d9a', 'AdamW-split': '#e31a1c', 'Muon-headlr': '#9970c4',
         'SGD-momentum': '#8c8c8c', 'Adam': '#1f77b4'}


def mode_plot(args):
    sweep = json.load(open(os.path.join(OUTDIR, 'controls_sweep.json')))
    spath = os.path.join(OUTDIR, 'controls_seeds.json')
    seeds = json.load(open(spath)) if os.path.exists(spath) else []
    npz_path = os.path.join(OUTDIR, 'controls_samples.npz')
    samples = np.load(npz_path) if os.path.exists(npz_path) else None

    muon_ref = min([r for r in sweep if r['variant'] == 'Muon'],
                   key=lambda r: r['hellinger'])['hellinger']
    adam_ref = min([r for r in sweep if r['variant'] == 'Adam'],
                   key=lambda r: r['hellinger'])['hellinger']

    # ---- Figure 1: control sweeps against the Muon / Adam reference lines ----
    fig, ax = plt.subplots(figsize=(9, 5.6))
    for variant in SWEEPS:
        rs = sorted([r for r in sweep if r['variant'] == variant],
                    key=lambda r: r['lr'])
        ax.plot([r['lr'] for r in rs],
                [r['hellinger'] if not r['diverged'] else np.nan for r in rs],
                'o-', color=COLOR[variant], lw=1.9, ms=6,
                label=LABEL[variant].replace('\n', ' '))
    ax.axhline(muon_ref, color=COLOR['Muon'], ls='--', lw=2,
               label=f'Muon reference ({muon_ref:.3f})')
    ax.axhline(adam_ref, color=COLOR['Adam'], ls=':', lw=2,
               label=f'Adam paper baseline ({adam_ref:.3f})')
    ax.set_xscale('log')
    ax.set_xlabel('learning rate  (hidden-group lr for split variants)')
    ax.set_ylabel('Hellinger distance to analytical solution')
    ax.set_title('Muon controls: is the advantage the orthogonalization,\n'
                 'or just the 100:1 hidden/head learning-rate split?', fontsize=11.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p1 = os.path.join(OUTDIR, 'muon_controls_sweep.png')
    plt.savefig(p1, dpi=150); plt.close()
    print(f"  wrote {p1}")

    if not seeds:
        print("  (no controls_seeds.json yet -- run --mode seeds for figure 2)")
        return

    # ---- Figure 2: densities + seed-averaged accuracy ----
    order = [v for v in ['Muon', 'AdamW-split', 'Muon-headlr', 'SGD-momentum', 'Adam']
             if any(r['variant'] == v for r in seeds)]
    grid = np.linspace(R.GRID_LO, R.GRID_HI, 800)
    pt = R.true_pdf(grid)
    n = len(order)
    fig = plt.figure(figsize=(4.1 * n, 8.4))
    for i, v in enumerate(order):
        ax = fig.add_subplot(2, n, i + 1)
        ax.plot(grid, pt, 'k-', lw=2.2, zorder=3,
                label='analytical' if i == 0 else None)
        if samples is not None and v in samples.files:
            ax.hist(samples[v], bins=150, range=(R.GRID_LO, R.GRID_HI),
                    density=True, color=COLOR[v], alpha=0.6,
                    label='learned' if i == 0 else None)
        rs = [r for r in seeds if r['variant'] == v and not r['diverged']]
        h0 = [r for r in rs if r['seed'] == 0]
        sub = f"Hellinger = {h0[0]['hellinger']:.3f}" if h0 else 'diverged'
        ax.set_title(f"{LABEL[v]}\n{sub}", fontsize=9.5)
        ax.set_xlim(R.GRID_LO, R.GRID_HI); ax.set_ylim(0, 1.55)
        ax.set_xlabel('$x$'); ax.grid(alpha=0.25)
        if i == 0:
            ax.set_ylabel('density'); ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 1, 2)
    means, sds, labs, cols = [], [], [], []
    for v in order:
        vals = np.array([r['hellinger'] for r in seeds
                         if r['variant'] == v and not r['diverged']], dtype=float)
        ndiv = len([r for r in seeds if r['variant'] == v and r['diverged']])
        means.append(vals.mean() if len(vals) else np.nan)
        sds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        labs.append(LABEL[v].replace('\n', ' ') +
                    (f'  [{ndiv} diverged]' if ndiv else ''))
        cols.append(COLOR[v])
    ax.bar(range(len(order)), means, yerr=sds, capsize=5, color=cols, alpha=0.85)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel('Hellinger  (mean $\\pm$ sd over seeds)')
    ax.set_title(f'Seed-averaged accuracy, each at its own best learning rate '
                 f'(seeds {SEEDS})', fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    for i, (m, s) in enumerate(zip(means, sds)):
        if np.isfinite(m):
            ax.text(i, m + s + 0.006, f'{m:.3f}', ha='center', fontsize=9.5)

    plt.tight_layout()
    p2 = os.path.join(OUTDIR, 'muon_controls_result.png')
    plt.savefig(p2, dpi=150); plt.close()
    print(f"  wrote {p2}")

    print()
    print(f"{'variant':34s} {'best lr':>9} {'hellinger (mean+-sd)':>24} {'n':>4}")
    print('-' * 76)
    for v, m, s in zip(order, means, sds):
        rs = [r for r in seeds if r['variant'] == v]
        lr = rs[0]['lr'] if rs else float('nan')
        ok = len([r for r in rs if not r['diverged']])
        print(f"{LABEL[v].replace(chr(10),' '):34s} {lr:9g} "
              f"{m:>12.4f} +- {s:.4f} {ok:>4}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['sweep', 'seeds', 'plot'])
    args = ap.parse_args()
    print(f"device: {R.device}")
    if args.mode == 'sweep':
        mode_sweep(args)
    elif args.mode == 'seeds':
        mode_seeds(args)
    else:
        mode_plot(args)
