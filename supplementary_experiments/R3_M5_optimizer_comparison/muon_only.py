"""
Muon in isolation on main.tex's Sec. 4.2 1D double-peak problem.

Standalone single-optimizer run, mirroring the paper's own Fig. 4 layout:
(a) learned distribution vs. the analytical Boltzmann solution, (b) the
training loss curve.

Everything except the generator optimizer matches the manuscript's Sec. 4.2
training configuration:
    V(x) = (x^2-1)^2, sigma = 0.6, b(x) = -4x^3 + 4x
    pushforward (8, 100, 1) tanh MLP, 8-dim Gaussian base
    test functions f^(k)(x) = sin(w^(k) x + b^(k)), K = 100
    16,000 epochs, batch M = 1000
    adversary: SGD lr 1e-2, 5 ascent steps every 10 epochs
plus the sigma-annealing curriculum (without which this problem collapses to
a point mass for EVERY optimizer -- see STAGE1/STAGE2 summaries on curriculum
being load-bearing).

IMPORTANT -- what "Muon" means here. Muon is defined for 2-D hidden weight
matrices, so on this (8,100,1) network it governs exactly ONE of the four
parameter tensors:
    net.0.weight  (100, 8)  -> Muon,             lr = MUON_LR
    net.0.bias    (100,)    -> internal AdamW,   lr = ADAMW_LR
    net.2.weight  (1, 100)  -> internal AdamW,   lr = ADAMW_LR   (output layer)
    net.2.bias    (1,)      -> internal AdamW,   lr = ADAMW_LR
This is the standard Muon convention (hidden matrices on Muon; biases and the
head on AdamW), but it means the configuration differs from a plain-Adam run
in TWO ways at once: the orthogonalized update, and a 100:1 hidden-vs-head
learning-rate split. Attributing the result to the orthogonalization alone
requires the control run in muon_controls.py.

Usage:
    python muon_only.py            # train + plot  (~2 min on a laptop GPU)
    python muon_only.py --plot     # re-plot from saved samples
"""
import os, json, time, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import run_experiment as R

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'results')
os.makedirs(OUTDIR, exist_ok=True)

MUON_LR = 2e-1          # best non-collapsed setting from the LR sweep
ADAMW_LR = R.LR_GEN_PAPER   # 2e-3, Muon's internal AdamW for biases + head
SEED = 0
N_SAMPLES = 200000
NPZ = os.path.join(OUTDIR, 'muon_only.npz')


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    pf = R.SteadyPushforward(d_base=R.D_BASE, n_out=1, hidden=R.HIDDEN).to(R.device)
    tf = R.SteadyTestFunctions(n_in=1, K=R.K_TEST).to(R.device)

    gen_opt = R.make_gen_opt('Muon', pf, MUON_LR)
    test_opt = R.make_test_opt('SGD', tf.parameters(), R.LR_TEST_PAPER)

    named = {id(p): n for n, p in pf.named_parameters()}
    print("  parameter routing:")
    for g in gen_opt.param_groups:
        tag = 'Muon' if g.get('use_muon') else 'AdamW'
        for p in g['params']:
            print(f"    {named[id(p)]:14s} {str(tuple(p.shape)):10s} -> {tag:6s} lr={g['lr']:g}")

    cur = [R.SIGMA2]

    def sigma2_cur(x):
        return cur[0] * torch.ones(x.shape[0], 1, device=x.device)

    def loss_fn():
        return R.steady_loss(pf, tf, M=R.BATCH_M, drift_fn=R.drift_fn,
                             sigma2_fn=sigma2_cur,
                             base_sampler=R.gaussian_base_sampler, device=R.device)

    n_epochs = R.N_EPOCHS
    losses = np.empty(n_epochs)
    sigmas = np.empty(n_epochs)
    t0 = time.time()
    for epoch in range(n_epochs):
        cur[0] = R.sigma2_at_epoch(epoch, n_epochs, anneal=True)
        sigmas[epoch] = cur[0]
        if epoch % R.ADV_EVERY == 0 and epoch > 0:
            for _ in range(R.ADV_STEPS):
                l, _ = loss_fn()
                test_opt.zero_grad(); (-l).backward(); test_opt.step()
        l, _ = loss_fn()
        gen_opt.zero_grad(); l.backward(); gen_opt.step()
        losses[epoch] = l.item()
        if epoch % 4000 == 0 or epoch == n_epochs - 1:
            print(f"    epoch {epoch:6d}/{n_epochs}  sigma^2={cur[0]:.3f}  "
                  f"loss={losses[epoch]:.4e}", flush=True)
    dt = time.time() - t0
    print(f"  training time {dt:.1f}s")

    with torch.no_grad():
        x = pf(torch.randn(N_SAMPLES, R.D_BASE, device=R.device)).cpu().numpy().ravel()

    np.savez_compressed(NPZ, x=x.astype(np.float32),
                        loss=losses[::25].astype(np.float32),
                        sigma2=sigmas[::25].astype(np.float32),
                        meta=np.array([MUON_LR, ADAMW_LR, dt, n_epochs]))
    print(f"  -> saved {NPZ}")


def metrics(x):
    edges = np.linspace(R.GRID_LO, R.GRID_HI, R.N_BINS + 1)
    c = 0.5 * (edges[:-1] + edges[1:])
    dx = edges[1] - edges[0]
    ph, _ = np.histogram(x, bins=edges, density=True)
    ph = ph * float(((x >= R.GRID_LO) & (x <= R.GRID_HI)).mean())
    pt = R.true_pdf(c)
    pt = pt / (pt.sum() * dx)

    hell = float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(ph * pt)) * dx)))
    tv = float(0.5 * np.sum(np.abs(ph - pt)) * dx)
    frac_right = float((x > 0).mean())
    bridge = float((np.abs(x) < 0.5).mean())
    bridge_true = float(pt[np.abs(c) < 0.5].sum() * dx)

    # true moments by quadrature
    from scipy.integrate import quad
    Z, _ = quad(R.unnorm_density, -6, 6, limit=400)
    m2, _ = quad(lambda t: t * t * R.unnorm_density(t) / Z, -6, 6, limit=400)
    true_std = float(np.sqrt(m2))

    return {
        'hellinger': hell, 'tv': tv,
        'frac_right': frac_right, 'well_balance_err': abs(frac_right - 0.5),
        'peak_learned': float(ph.max()), 'peak_true': float(pt.max()),
        'peak_overshoot': float(ph.max() / pt.max()),
        'bridge_learned': bridge, 'bridge_true': bridge_true,
        'bridge_ratio': bridge / bridge_true,
        'std_learned': float(x.std()), 'std_true': true_std,
    }


def plot():
    d = np.load(NPZ)
    x = d['x']; loss = d['loss']; sig = d['sigma2']
    muon_lr, adamw_lr, dt, n_epochs = d['meta']
    m = metrics(x)

    with open(os.path.join(OUTDIR, 'muon_only.json'), 'w') as fh:
        json.dump({**m, 'muon_lr': float(muon_lr), 'adamw_lr': float(adamw_lr),
                   'train_time_s': float(dt), 'n_epochs': int(n_epochs),
                   'seed': SEED}, fh, indent=2)

    grid = np.linspace(R.GRID_LO, R.GRID_HI, 800)
    pt = R.true_pdf(grid)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6),
                             gridspec_kw={'width_ratios': [1.25, 1]})

    # ---- (a) learned distribution ----
    ax = axes[0]
    ax.hist(x, bins=160, range=(R.GRID_LO, R.GRID_HI), density=True,
            color='#7ba7d1', alpha=0.75, label='learned samples (Muon)',
            edgecolor='none')
    ax.plot(grid, pt, 'k-', lw=2.4, label='analytical Boltzmann', zorder=3)
    ax.set_xlabel('$x$'); ax.set_ylabel('density')
    ax.set_xlim(R.GRID_LO, R.GRID_HI)
    ax.set_title('(a) Learned distribution vs. analytical solution', fontsize=11)
    ax.legend(fontsize=9, loc='upper center')
    ax.grid(alpha=0.25)

    txt = (f"Hellinger      {m['hellinger']:.3f}\n"
           f"total variation {m['tv']:.3f}\n"
           f"peak ratio      {m['peak_overshoot']:.2f}x\n"
           f"mode balance    {m['well_balance_err']:.4f}\n"
           f"std   {m['std_learned']:.3f}  (true {m['std_true']:.3f})")
    ax.text(0.015, 0.975, txt, transform=ax.transAxes, va='top', ha='left',
            fontsize=8.6, family='monospace',
            bbox=dict(boxstyle='round,pad=0.45', fc='white', ec='0.6', alpha=0.92))

    # ---- (b) loss curve + annealing schedule ----
    ax = axes[1]
    ep = np.arange(len(loss)) * 25
    ax.semilogy(ep, loss, color='#6a3d9a', lw=1.0, alpha=0.85)
    k = 40
    if len(loss) > k:
        sm = np.convolve(loss, np.ones(k) / k, mode='valid')
        ax.semilogy(ep[k - 1:], sm, color='#2d0f4a', lw=2.0, label=f'moving avg ({k*25} ep)')
    ax.set_xlabel('epoch'); ax.set_ylabel('training loss')
    ax.set_title('(b) Training loss convergence', fontsize=11)
    ax.grid(alpha=0.25, which='both')

    ax2 = ax.twinx()
    ax2.plot(ep, sig, color='#d95f02', lw=1.6, ls='--', label=r'$\sigma^2$ (curriculum)')
    ax2.set_ylabel(r'$\sigma^2$', color='#d95f02')
    ax2.tick_params(axis='y', labelcolor='#d95f02')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc='upper right')

    fig.suptitle(
        'Muon on the Sec. 4.2 double-well  '
        f'(Muon lr={muon_lr:g} on the 100x8 hidden matrix; '
        f'AdamW lr={adamw_lr:g} on biases + output layer)\n'
        r'$V(x)=(x^2-1)^2$, $\sigma=0.6$, (8,100,1) tanh MLP, K=100 plane waves, '
        r'16k epochs, seed 0, $\sigma$-annealed',
        fontsize=11.5)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(OUTDIR, 'muon_only.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  wrote {out}")

    print()
    for k_, v in m.items():
        print(f"    {k_:18s} {v:.4f}" if isinstance(v, float) else f"    {k_:18s} {v}")


def plot_paper(dest_a, dest_b):
    """Manuscript version of the figure for main.tex Sec. 4.2.

    Written as TWO separate image files so main.tex can place them side by
    side as subfigures. Differs from plot() only in presentation: no titles
    and no statistics box (both move into the LaTeX caption), legends
    retained. The underlying data is identical.
    """
    d = np.load(NPZ)
    x = d['x']; loss = d['loss']; sig = d['sigma2']
    grid = np.linspace(R.GRID_LO, R.GRID_HI, 800)
    pt = R.true_pdf(grid)

    # main.tex has 	extwidth = 390pt = 5.40in, and each panel is included at
    # roughly half of that, so a panel drawn W inches wide is shrunk by about
    # 2.8/W. Fonts are sized up accordingly to land near 8-9pt when printed.
    plt.rcParams.update({
        'axes.labelsize': 17, 'xtick.labelsize': 15, 'ytick.labelsize': 15,
        'legend.fontsize': 14,
    })

    # ---------------- (a) learned distribution ----------------
    # Restricted to [-2, 2]: the density is numerically zero outside it, and
    # the tighter range plus extra headroom keeps the legend clear of the two
    # peaks at x = +-1.
    LO, HI = -2.0, 2.0
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.hist(x, bins=140, range=(LO, HI), density=True,
            color='#7ba7d1', alpha=0.75, label='learned samples',
            edgecolor='none')
    gm = (grid >= LO) & (grid <= HI)
    ax.plot(grid[gm], pt[gm], 'k-', lw=2.4, label='analytical Boltzmann', zorder=3)
    ax.set_xlabel('$x$'); ax.set_ylabel('density')
    ax.set_xlim(LO, HI)
    ax.set_ylim(0, 1.78)          # headroom so the legend clears the peaks
    ax.legend(loc='upper center', framealpha=0.95)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(dest_a, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  wrote {dest_a}")

    # ---------------- (b) loss curve + annealing schedule ----------------
    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    ep = np.arange(len(loss)) * 25
    ax.semilogy(ep, loss, color='#6a3d9a', lw=1.0, alpha=0.8,
                label=r'loss ${L}_{\mathrm{total}}$')
    k = 40
    if len(loss) > k:
        sm = np.convolve(loss, np.ones(k) / k, mode='valid')
        ax.semilogy(ep[k - 1:], sm, color='#2d0f4a', lw=2.0,
                    label='moving average')
    ax.set_xlabel('epoch'); ax.set_ylabel('training loss')
    ax.set_ylim(top=2e2)          # headroom for the legend
    ax.grid(alpha=0.25, which='both')

    ax2 = ax.twinx()
    ax2.plot(ep, sig, color='#d95f02', lw=1.6, ls='--', label=r'$\sigma^2$ schedule')
    ax2.set_ylabel(r'$\sigma^2$', color='#d95f02')
    ax2.tick_params(axis='y', labelcolor='#d95f02')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=12, loc='upper right', framealpha=0.95)

    plt.tight_layout()
    plt.savefig(dest_b, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  wrote {dest_b}")
    return dest_a, dest_b


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--plot', action='store_true', help='re-plot from saved npz')
    ap.add_argument('--paper', nargs=2, metavar=('DEST_A', 'DEST_B'),
                    help='write the manuscript version as two separate files '
                         '(no titles, no stats box): panel (a) to DEST_A and '
                         'panel (b) to DEST_B')
    args = ap.parse_args()
    print(f"device: {R.device}")
    if args.paper is not None:
        plot_paper(args.paper[0], args.paper[1])
    else:
        if not args.plot:
            train()
        plot()
