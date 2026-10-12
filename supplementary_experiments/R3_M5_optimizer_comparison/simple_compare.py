"""
Simple, figure-first version of the R3_M5 optimizer comparison.

One training run per optimizer, each at the best learning rate found by the
earlier LR sweep (sweep.json + fine.json), on main.tex's own Sec. 4.2
1D double-peak problem. The deliverable is the picture the paper actually
shows -- learned samples against the analytical Boltzmann density -- not a
table of scalars.

    python simple_compare.py          # train + plot   (~20 min)
    python simple_compare.py --plot   # re-plot from saved samples
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import run_experiment as R   # problem definition, optimizers, curriculum

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'results')
os.makedirs(OUTDIR, exist_ok=True)

# (label, optimizer name, lr) -- lrs are the best non-collapsed settings from
# the completed sweep. "Adam (paper)" is the manuscript's own configuration,
# kept as the reference row.
CONFIGS = [
    ('Adam (paper, 2e-3)', 'Adam',            2e-3),
    ('Adam (best, 1e-2)',  'Adam',            1e-2),
    ('SGD',                'SGD',             2e-2),
    ('AdamW',              'AdamW',           1e-3),
    ('Shampoo',            'Shampoo',         3e-3),
    ('ScalableShampoo',    'ScalableShampoo', 5e-3),
    ('SOAP',               'SOAP',            3e-2),
    ('Muon',               'Muon',            2e-1),
]

SEED = 0
N_SAMPLES = 100000


def train_and_sample(gen_name, lr, seed=SEED, n_epochs=R.N_EPOCHS):
    """Same loop as run_experiment.run(), but returns the samples so we can
    plot the learned density rather than only scoring it."""
    torch.manual_seed(seed); np.random.seed(seed)
    pf = R.SteadyPushforward(d_base=R.D_BASE, n_out=1, hidden=R.HIDDEN).to(R.device)
    tf = R.SteadyTestFunctions(n_in=1, K=R.K_TEST).to(R.device)
    gen_opt = R.make_gen_opt(gen_name, pf, lr)
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
        x = pf(torch.randn(N_SAMPLES, R.D_BASE, device=R.device)).cpu().numpy().ravel()
    return x, losses, dt, diverged


def hellinger(x):
    edges = np.linspace(R.GRID_LO, R.GRID_HI, R.N_BINS + 1)
    c = 0.5 * (edges[:-1] + edges[1:])
    dx = edges[1] - edges[0]
    ph, _ = np.histogram(x, bins=edges, density=True)
    ph = ph * float(((x >= R.GRID_LO) & (x <= R.GRID_HI)).mean())
    pt = R.true_pdf(c)
    pt = pt / (pt.sum() * dx)
    return float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(ph * pt)) * dx)))


TITLE_SUB = (r'$V(x)=(x^2-1)^2$, $\sigma=0.6$, (8,100,1) tanh MLP, '
             r'K=100 plane waves, 16k epochs, seed 0, $\sigma$-annealed')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plot', action='store_true', help='re-plot from saved npz')
    args = ap.parse_args()
    npz = os.path.join(OUTDIR, 'simple_samples.npz')

    if not args.plot:
        print(f"device: {R.device}")
        store = {}
        for label, name, lr in CONFIGS:
            print(f"  training {label:22s} lr={lr:<8g} ...", end='', flush=True)
            x, losses, dt, div = train_and_sample(name, lr)
            h = hellinger(x)
            print(f" hellinger={h:.4f}  std={x.std():.3f}  {dt:.0f}s"
                  f"{'  [DIVERGED]' if div else ''}", flush=True)
            store[label + '|x'] = x.astype(np.float32)
            store[label + '|loss'] = losses[::50].astype(np.float32)
            store[label + '|meta'] = np.array([lr, h, dt, float(div), x.std()])
        np.savez_compressed(npz, **store)
        print(f"  -> saved {npz}")

    d = np.load(npz)
    grid = np.linspace(R.GRID_LO, R.GRID_HI, 600)
    pt = R.true_pdf(grid)

    # ---- Figure 1: learned density vs analytical solution, one panel each ----
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5), sharex=True, sharey=True)
    for ax, (label, name, lr) in zip(axes.ravel(), CONFIGS):
        x = d[label + '|x']
        lr_, h, dt, div, sd = d[label + '|meta']
        ax.plot(grid, pt, 'k-', lw=2.2, label='analytical Boltzmann', zorder=3)
        if div:
            ax.text(0.5, 0.5, 'DIVERGED', ha='center', va='center',
                    transform=ax.transAxes, fontsize=15, color='crimson',
                    weight='bold')
        else:
            ax.hist(x, bins=140, range=(R.GRID_LO, R.GRID_HI), density=True,
                    color='steelblue', alpha=0.65, label='learned samples')
        good = (not div) and h < 0.30
        ax.set_title(f"{label}   (lr={lr_:g})\nHellinger = {h:.3f}"
                     f"{'' if good else '   <-- fails'}",
                     fontsize=10.5, color=('black' if good else 'crimson'))
        ax.set_xlim(R.GRID_LO, R.GRID_HI)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8.5, loc='upper left')
    for ax in axes[1]:
        ax.set_xlabel('x')
    for ax in axes[:, 0]:
        ax.set_ylabel('density')
    fig.suptitle('main.tex Sec. 4.2 double-well: learned distribution by optimizer\n'
                 + TITLE_SUB, fontsize=12.5)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    p1 = os.path.join(OUTDIR, 'density_by_optimizer.png')
    plt.savefig(p1, dpi=140)
    plt.close()
    print(f"  wrote {p1}")

    # ---- Figure 2: loss curves + accuracy bars ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    cmap = plt.get_cmap('tab10')
    for i, (label, name, lr) in enumerate(CONFIGS):
        y = d[label + '|loss']
        axes[0].semilogy(np.arange(len(y)) * 50, y, lw=1.4, label=label,
                         color=cmap(i % 10))
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('training loss')
    axes[0].set_title('Loss curves\n(minimax objective - NOT a fair accuracy ranking)')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    labels = [c[0] for c in CONFIGS]
    hs = [float(d[l + '|meta'][1]) for l in labels]
    divs = [bool(d[l + '|meta'][3]) for l in labels]
    cols = ['#c62828' if (dv or h >= 0.30) else '#2e7d32'
            for h, dv in zip(hs, divs)]
    axes[1].barh(range(len(labels)), [0 if dv else h for h, dv in zip(hs, divs)],
                 color=cols, alpha=0.85)
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_yticklabels(labels, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Hellinger distance to analytical solution  (lower = better)')
    axes[1].set_title('Accuracy (seed 0, each at its own best LR)')
    axes[1].grid(alpha=0.3, axis='x')
    for i, (h, dv) in enumerate(zip(hs, divs)):
        axes[1].text(0.005, i, '  diverged' if dv else f'  {h:.3f}',
                     va='center', fontsize=9)
    plt.tight_layout()
    p2 = os.path.join(OUTDIR, 'loss_and_accuracy.png')
    plt.savefig(p2, dpi=140)
    plt.close()
    print(f"  wrote {p2}")


if __name__ == '__main__':
    main()
