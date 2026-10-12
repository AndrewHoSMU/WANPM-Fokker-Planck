"""
Does a larger Monte-Carlo batch M reduce the transition-state bridge?

Motivation. With the best Adam configuration found so far (hidden/head split,
beta2=0.95), the residual error is no longer peak shape -- it is the bridge of
unphysical samples across the barrier, which accounts for 60% of the squared
Hellinger integrand (85% for Muon). The squared weak-form loss carries a bias

    E[Ehat^2] = (E[Lf])^2 + Var(Lf) / M

whose second term does not vanish at the true solution and is reduced by
raising M. If the bridge is driven by that term, it should shrink with M.

Per the author: the number of test functions K should be set equal to M, so
the requested configuration is (M=2000, K=2000). Changing M and K together
would confound the two, so the two single-factor cells are run as well:

    (M=1000, K=100)    baseline, the Sec. 4.2 setting
    (M=2000, K=100)    M alone
    (M=1000, K=1000)   K alone
    (M=2000, K=2000)   both -- the requested configuration

Generator optimizer is held at the best configuration throughout:
Adam, betas=(0.9, 0.95), hidden lr 3e-2 / head lr 2e-3.
Seed 0 only, to keep this short.
"""
import os, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import run_experiment as R
import muon_controls as MC

OUTDIR = MC.OUTDIR
HIDDEN_LR, HEAD_LR, BETAS = 3e-2, 2e-3, (0.9, 0.95)
SEED = 0

CONFIGS = [
    ('M=1000, K=100  (Sec. 4.2 baseline)', 1000, 100),
    ('M=2000, K=100  (M alone)', 2000, 100),
    ('M=1000, K=1000 (K alone)', 1000, 1000),
    ('M=2000, K=2000 (requested)', 2000, 2000),
]


def run(M, K, seed=SEED, n_epochs=None):
    n_epochs = n_epochs or R.N_EPOCHS
    torch.manual_seed(seed); np.random.seed(seed)
    pf = R.SteadyPushforward(d_base=R.D_BASE, n_out=1, hidden=R.HIDDEN).to(R.device)
    tf = R.SteadyTestFunctions(n_in=1, K=K).to(R.device)

    hidden, rest = MC.split_hidden_head(pf)
    gen_opt = torch.optim.Adam([dict(params=hidden, lr=HIDDEN_LR),
                                dict(params=rest, lr=HEAD_LR)], betas=BETAS)
    test_opt = torch.optim.SGD(tf.parameters(), lr=R.LR_TEST_PAPER)

    cur = [R.SIGMA2]

    def sigma2_cur(x):
        return cur[0] * torch.ones(x.shape[0], 1, device=x.device)

    def loss_fn():
        return R.steady_loss(pf, tf, M=M, drift_fn=R.drift_fn,
                             sigma2_fn=sigma2_cur,
                             base_sampler=R.gaussian_base_sampler, device=R.device)

    losses = np.empty(n_epochs); diverged = False
    if R.device.type == 'cuda':
        torch.cuda.synchronize()
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
            diverged = True; losses[epoch:] = np.nan; break
    if R.device.type == 'cuda':
        torch.cuda.synchronize()
    dt = time.time() - t0

    with torch.no_grad():
        x = pf(torch.randn(200000, R.D_BASE, device=R.device)).cpu().numpy().ravel()
    rec = dict(M=M, K=K, seed=seed, diverged=diverged, train_time_s=dt)
    rec.update(MC._metrics(x))
    return rec, x


def hellinger_by_region(x):
    edges = np.linspace(R.GRID_LO, R.GRID_HI, R.N_BINS + 1)
    c = 0.5 * (edges[:-1] + edges[1:]); dx = edges[1] - edges[0]
    ph, _ = np.histogram(x, bins=edges, density=True)
    ph = ph * float(((x >= R.GRID_LO) & (x <= R.GRID_HI)).mean())
    pt = R.true_pdf(c); pt = pt / (pt.sum() * dx)
    integ = (np.sqrt(ph) - np.sqrt(pt)) ** 2 * 0.5 * dx
    bar = np.abs(c) < 0.5
    return float(integ.sum()), float(integ[bar].sum() / integ.sum())


def main():
    print(f"device: {R.device}")
    print(f"generator: Adam betas={BETAS}, hidden lr={HIDDEN_LR:g}, head lr={HEAD_LR:g}\n")
    recs, samples = [], {}
    for label, M, K in CONFIGS:
        print(f"  {label:36s} ...", end='', flush=True)
        rec, x = run(M, K)
        tot, frac = hellinger_by_region(x)
        rec['label'] = label
        rec['barrier_frac'] = frac
        print(f" H={rec['hellinger']:.4f}  bridge={rec['bridge_ratio']:.2f}x  "
              f"peak={rec['peak_overshoot']:.2f}x  barrier-share={100*frac:.0f}%  "
              f"({rec['train_time_s']:.0f}s)", flush=True)
        recs.append(rec); samples[label] = x.astype(np.float32)
        with open(os.path.join(OUTDIR, 'batch_size.json'), 'w') as fh:
            json.dump(recs, fh, indent=2)
    np.savez_compressed(os.path.join(OUTDIR, 'batch_size_samples.npz'), **samples)

    # ---- figure ----
    grid = np.linspace(R.GRID_LO, R.GRID_HI, 800)
    pt = R.true_pdf(grid)
    fig, axes = plt.subplots(1, len(CONFIGS) + 1,
                             figsize=(4.0 * (len(CONFIGS) + 1), 4.6))
    for ax, rec in zip(axes[:-1], recs):
        x = samples[rec['label']]
        ax.hist(x, bins=150, range=(R.GRID_LO, R.GRID_HI), density=True,
                color='#c62828', alpha=0.6, label='learned')
        ax.plot(grid, pt, 'k-', lw=2.2, zorder=3, label='analytical')
        ax.set_ylim(0, 1.75); ax.set_xlim(-2.2, 2.2)
        ax.set_xlabel('$x$'); ax.grid(alpha=0.25)
        ax.set_title(f"{rec['label']}\nH={rec['hellinger']:.3f}, "
                     f"bridge={rec['bridge_ratio']:.1f}x", fontsize=9.5)
    axes[0].set_ylabel('density'); axes[0].legend(fontsize=8)

    ax = axes[-1]
    labs = [r['label'].split('(')[0].strip() for r in recs]
    ax.bar(range(len(recs)), [r['bridge_ratio'] for r in recs],
           color='#c62828', alpha=0.85)
    ax.axhline(1.0, color='k', ls='--', lw=1.6, label='analytical (1.0x)')
    ax.set_xticks(range(len(recs)))
    ax.set_xticklabels(labs, fontsize=8, rotation=30, ha='right')
    ax.set_ylabel('bridge mass / true bridge mass')
    ax.set_title('Excess mass across the barrier', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    fig.suptitle('Effect of Monte-Carlo batch size M (and test-function count K) '
                 'on the transition-state bridge\n'
                 'Sec. 4.2 double well, best Adam configuration, seed 0',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    out = os.path.join(OUTDIR, 'batch_size_test.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f"\n  wrote {out}")


if __name__ == '__main__':
    main()
