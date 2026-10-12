"""
The Adam hyperparameter search for main.tex Sec. 4.2, in one figure.

Answers: what is the best Adam configuration for the double-well problem,
and how much better is it than the manuscript's current setting?

Searched (all at seed 0, everything else matching Sec. 4.2 exactly):
  - unsplit Adam:  beta2 in {0.999, 0.99, 0.95, 0.9}  x  lr in {5e-4 .. 1e-1}
  - split Adam:    beta2 in {0.999, 0.95}  x  hidden lr in {2e-3 .. 1e-1},
                   head group (biases + output layer) fixed at 2e-3
Best cells were then re-run at seeds 0,1,2.

Reads only cached results -- no training. Regenerate the inputs with
    python beta2_ablation.py --mode sweep && --mode seeds
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import run_experiment as R

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'results')

PAPER = dict(label='manuscript Sec. 4.2\nAdam, $\\beta_2$=0.999, lr=2e-3',
             short='manuscript Adam', color='#1f77b4')
BEST = dict(label='best found\nAdam split, $\\beta_2$=0.95,\nhidden lr=3e-2 / head 2e-3',
            short='best Adam', color='#c62828')

UNSPLIT_B2 = [0.999, 0.99, 0.95, 0.9]
UNSPLIT_LRS = [5e-4, 2e-3, 1e-2, 3e-2, 1e-1]
SPLIT_B2 = [0.999, 0.95]
SPLIT_LRS = [2e-3, 1e-2, 3e-2, 1e-1]

COLLAPSE = 0.40   # |P(x>0)-0.5| above this = one well only


def _grid(sweep, prefix, betas, lrs):
    """-> (values, status) with status in {'ok','collapsed','diverged','missing'}"""
    vals = np.full((len(betas), len(lrs)), np.nan)
    stat = np.full((len(betas), len(lrs)), 'missing', dtype=object)
    for i, b in enumerate(betas):
        for j, lr in enumerate(lrs):
            rs = [r for r in sweep
                  if r['variant'] == f'{prefix}{b}' and abs(r['lr'] - lr) < 1e-12]
            if not rs:
                continue
            r = rs[0]
            if r['diverged']:
                stat[i, j] = 'diverged'
            elif r['well_balance_err'] >= COLLAPSE:
                stat[i, j] = 'collapsed'
                vals[i, j] = r['hellinger']
            else:
                stat[i, j] = 'ok'
                vals[i, j] = r['hellinger']
    return vals, stat


def _heat(ax, vals, stat, betas, lrs, title, xlabel, vmin, vmax):
    masked = np.where(np.array([[s == 'ok' for s in row] for row in stat]),
                      vals, np.nan)
    im = ax.imshow(masked, cmap='viridis_r', vmin=vmin, vmax=vmax,
                   aspect='auto', origin='upper')
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([f'{l:g}' for l in lrs], fontsize=9)
    ax.set_yticks(range(len(betas)))
    ax.set_yticklabels([f'{b}' for b in betas], fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_ylabel(r'$\beta_2$', fontsize=10)
    ax.set_title(title, fontsize=10.5)

    best = (None, np.inf)
    for i in range(len(betas)):
        for j in range(len(lrs)):
            s = stat[i, j]
            if s == 'ok':
                v = vals[i, j]
                ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=8.5,
                        color='white' if v > (vmin + vmax) / 2 else 'black')
                if v < best[1]:
                    best = ((i, j), v)
            else:
                txt = {'diverged': 'div', 'collapsed': 'collapse',
                       'missing': '--'}[s]
                ax.text(j, i, txt, ha='center', va='center', fontsize=7.5,
                        color='#999999', style='italic')
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=True,
                                       fc='#f0f0f0', ec='none', zorder=0))
    if best[0] is not None:
        i, j = best[0]
        ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                               ec='#c62828', lw=3, zorder=5))
    return im, best


def main():
    sweep = json.load(open(os.path.join(OUTDIR, 'beta2_sweep.json')))
    seeds = json.load(open(os.path.join(OUTDIR, 'beta2_seeds.json')))
    simple = np.load(os.path.join(OUTDIR, 'simple_samples.npz'))
    b2s = np.load(os.path.join(OUTDIR, 'beta2_samples.npz'))

    u_vals, u_stat = _grid(sweep, 'adam-b', UNSPLIT_B2, UNSPLIT_LRS)
    s_vals, s_stat = _grid(sweep, 'adamsplit-b', SPLIT_B2, SPLIT_LRS)
    finite = np.concatenate([u_vals[np.isfinite(u_vals)],
                             s_vals[np.isfinite(s_vals)]])
    vmin, vmax = 0.08, min(0.45, float(np.nanmax(finite)))

    fig = plt.figure(figsize=(15.5, 9.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.12], hspace=0.46, wspace=0.42)

    # ---- (a) unsplit heatmap ----
    ax = fig.add_subplot(gs[0, 0])
    im, ubest = _heat(ax, u_vals, u_stat, UNSPLIT_B2, UNSPLIT_LRS,
                      '(a) Plain Adam\n(one learning rate for all parameters)',
                      'learning rate', vmin, vmax)

    # ---- (b) split heatmap ----
    ax = fig.add_subplot(gs[0, 1])
    im2, sbest = _heat(ax, s_vals, s_stat, SPLIT_B2, SPLIT_LRS,
                       '(b) Adam with hidden/head split\n(head group fixed at 2e-3)',
                       'hidden-group learning rate', vmin, vmax)
    cb = fig.colorbar(im2, ax=ax, fraction=0.036, pad=0.02)
    cb.set_label('Hellinger distance', fontsize=9)

    # ---- (c) seed-averaged bars ----
    ax = fig.add_subplot(gs[0, 2])
    names = ['adam-b0.999', 'adam-b0.99', 'adam-b0.9', 'adam-b0.95',
             'adamsplit-b0.999', 'adamsplit-b0.95']
    pretty = {'adam-b0.999': 'Adam 0.999', 'adam-b0.99': 'Adam 0.99',
              'adam-b0.9': 'Adam 0.9', 'adam-b0.95': 'Adam 0.95',
              'adamsplit-b0.999': 'split 0.999', 'adamsplit-b0.95': 'split 0.95'}
    ms, ss, ls, cs = [], [], [], []
    for n in names:
        rs = [r for r in seeds if r['variant'] == n]
        good = [r['hellinger'] for r in rs
                if not r['diverged'] and r['well_balance_err'] < COLLAPSE]
        ms.append(np.mean(good) if good else np.nan)
        ss.append(np.std(good, ddof=1) if len(good) > 1 else 0.0)
        ls.append(f"{pretty[n]}\n{len(good)}/{len(rs)} ok")
        cs.append(BEST['color'] if n == 'adamsplit-b0.95' else '#8fb8d8')
    ax.bar(range(len(names)), ms, yerr=ss, capsize=4, color=cs, alpha=0.9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(ls, fontsize=7.8, rotation=38, ha='right')
    ax.set_ylabel('Hellinger (mean $\\pm$ sd, 3 seeds)', fontsize=9.5)
    ax.set_title('(c) Best cell of each row, re-run at 3 seeds', fontsize=10.5)
    ax.grid(alpha=0.3, axis='y')
    for i, (m, s) in enumerate(zip(ms, ss)):
        if np.isfinite(m):
            ax.text(i, m + s + 0.005, f'{m:.3f}', ha='center', fontsize=8.5)

    # ---- (d,e) before / after densities ----
    grid = np.linspace(R.GRID_LO, R.GRID_HI, 800)
    pt = R.true_pdf(grid)
    x_paper = simple['Adam (paper, 2e-3)|x']
    x_best = b2s['adamsplit-b0.95']

    def _dens(ax, x, cfg, h, tag):
        ax.hist(x, bins=160, range=(R.GRID_LO, R.GRID_HI), density=True,
                color=cfg['color'], alpha=0.6, label='learned samples')
        ax.plot(grid, pt, 'k-', lw=2.3, zorder=3, label='analytical Boltzmann')
        ax.set_xlim(R.GRID_LO, R.GRID_HI)
        ax.set_ylim(0, 2.45)
        ax.set_xlabel('$x$'); ax.set_ylabel('density')
        ax.set_title(f'{tag} {cfg["label"]}\nHellinger = {h:.3f}',
                     fontsize=10, color=cfg['color'])
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.5, loc='upper left')

    def _h(x):
        edges = np.linspace(R.GRID_LO, R.GRID_HI, R.N_BINS + 1)
        c = 0.5 * (edges[:-1] + edges[1:]); dx = edges[1] - edges[0]
        ph, _ = np.histogram(x, bins=edges, density=True)
        ph = ph * float(((x >= R.GRID_LO) & (x <= R.GRID_HI)).mean())
        q = R.true_pdf(c); q = q / (q.sum() * dx)
        return float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(ph * q)) * dx)))

    h_paper, h_best = _h(x_paper), _h(x_best)
    ax = fig.add_subplot(gs[1, 0]); _dens(ax, x_paper, PAPER, h_paper, '(d)')
    ax = fig.add_subplot(gs[1, 1]); _dens(ax, x_best, BEST, h_best, '(e)')

    # ---- (f) overlay ----
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(grid, pt, 'k-', lw=2.6, label='analytical Boltzmann', zorder=3)
    for x, cfg, h in ((x_paper, PAPER, h_paper), (x_best, BEST, h_best)):
        d, e = np.histogram(x, bins=160, range=(R.GRID_LO, R.GRID_HI), density=True)
        ax.plot(0.5 * (e[:-1] + e[1:]), d, lw=1.9, color=cfg['color'],
                label=f"{cfg['short']}  ({h:.3f})")
    ax.set_xlim(-1.8, 1.8); ax.set_ylim(0, 2.45)
    ax.set_xlabel('$x$'); ax.set_ylabel('density')
    ax.set_title('(f) Overlay: the manuscript setting over-peaks\n'
                 'and bridges the barrier', fontsize=10.5)
    ax.legend(fontsize=8.5); ax.grid(alpha=0.25)

    fig.suptitle('Best Adam configuration for the Sec. 4.2 double well  '
                 f'({h_paper:.3f} $\\rightarrow$ {h_best:.3f}, a '
                 f'{h_paper / h_best:.1f}$\\times$ reduction)\n'
                 r'$V(x)=(x^2-1)^2$, $\sigma=0.6$, (8,100,1) tanh MLP, K=100, '
                 r'16k epochs, $\sigma$-annealed; red box = best cell',
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUTDIR, 'adam_best_config.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  wrote {out}")
    print(f"  manuscript Adam : {h_paper:.4f}")
    print(f"  best Adam       : {h_best:.4f}   ({h_paper / h_best:.2f}x better)")


if __name__ == '__main__':
    main()
