"""
A LESS SEPARATED asymmetric double well: does the method succeed there?

The R1.5 example (V = (x^2-1)^2 + eps x, wells at x = +-1, separation 2.0) is
an extreme case: relative barrier 7.3, M needed for the split to be
detectable ~6e5 (residual_theory.py / identifiability.py). The author's
objection is fair -- that says nothing about whether the method can do an
asymmetric double well AT ALL, only that it fails on this one very hard
instance.

THIS SCRIPT.  Same functional form, closer wells: V(x) = (x^2 - a^2)^2 + eps x
with a = 0.7 (wells at x = +-0.7, separation 1.4, 30% closer). Same eps = 0.1,
same target sigma^2 = 0.3, same curriculum, same architecture, same batch size
M = 1000 -- nothing scaled up. By quadrature this instance has:

    true split          31:69   (vs 78:22 for a=1.0 -- a much milder tilt)
    rho*(0)             0.166   (vs 1.5e-3 -- 113x larger transition density)
    relative barrier    2.08    (vs 7.34)
    M for SNR=1         ~120    (vs ~6e5 -- comfortably below M=1000)

So the identifiability argument itself predicts this SHOULD be solvable at
the manuscript's own batch size, with no estimator or batch-size fix needed.

Two configurations, 3 seeds each:
    'manuscript'  sin test functions, the manuscript's OWN biased loss, Adam
                  at the paper's own settings -- i.e. nothing changed from
                  main.tex except the potential itself.
    'mixed'       the best configuration found in this study: 75% plane wave
                  + 25% Gaussian-wavelet test functions, U-statistic loss.

Usage:  python closer_wells.py [--epochs 16000] [--seeds 3]
"""
import os, json, time, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad

import asym_muon as A
from wavelet_testfuncs import MixedTestFunctions

OUTDIR = A.OUTDIR
device = A.device

# ---------------------------------------------------------------------------
A_WELL = 0.7          # well half-separation: wells at x = +-A_WELL (overridable via set_a)
EPS = 0.1
SIGMA2 = 0.3
D_BASE, HIDDEN, K_TEST = A.D_BASE, A.HIDDEN, A.K_TEST
N_EPOCHS_DEFAULT = A.N_EPOCHS
BATCH_M = A.BATCH_M


def V(x):
    return (x ** 2 - A_WELL ** 2) ** 2 + EPS * x


def drift_fn(x):
    return -4 * x ** 3 + 4 * A_WELL ** 2 * x - EPS


def unnorm_density(x, a=SIGMA2):
    return np.exp(-2 * V(x) / a)


def true_stats(a=SIGMA2):
    Z, _ = quad(unnorm_density, -4, 4, args=(a,), limit=400)
    fr, _ = quad(lambda t: unnorm_density(t, a) / Z, 0, 4, limit=400)
    return Z, float(fr)


Z_TRUE, TRUE_FRAC_RIGHT = true_stats()
TRUE_FRAC_LEFT = 1.0 - TRUE_FRAC_RIGHT


def true_pdf(x):
    return unnorm_density(x) / Z_TRUE


def true_moments(a=SIGMA2):
    Z, _ = quad(unnorm_density, -5, 5, args=(a,), limit=400)
    m1, _ = quad(lambda t: t * unnorm_density(t, a) / Z, -5, 5, limit=400)
    m2, _ = quad(lambda t: (t - m1) ** 2 * unnorm_density(t, a) / Z, -5, 5, limit=400)
    return float(m1), float(np.sqrt(m2))


TRUE_MEAN, TRUE_STD = true_moments()

GRID_LO, GRID_HI, N_BINS, N_EVAL = -2.5, 2.5, 250, 200000


def evaluate(x):
    edges = np.linspace(GRID_LO, GRID_HI, N_BINS + 1)
    c = 0.5 * (edges[:-1] + edges[1:])
    dx = edges[1] - edges[0]
    ph, _ = np.histogram(x, bins=edges, density=True)
    frac_in = float(((x >= GRID_LO) & (x <= GRID_HI)).mean())
    ph = ph * frac_in
    pt = true_pdf(c)
    pt = pt / (pt.sum() * dx)

    frac_right = float((x > 0).mean())
    return {
        'hellinger': float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(ph * pt)) * dx))),
        'tv': float(0.5 * np.sum(np.abs(ph - pt)) * dx),
        'frac_right': frac_right,
        'frac_right_err': abs(frac_right - TRUE_FRAC_RIGHT),
        'direction_correct': bool((frac_right - 0.5) * (TRUE_FRAC_RIGHT - 0.5) > 0),
        'mean': float(x.mean()), 'std': float(x.std()),
        'true_mean': TRUE_MEAN, 'true_std': TRUE_STD,
        'asymmetry_captured': float(x.mean() / TRUE_MEAN) if TRUE_MEAN else float('nan'),
        'frac_in_window': frac_in,
    }


# ---------------------------------------------------------------------------
def make_tf(kind, seed=None):
    if kind == 'sin':
        return A.SteadyTestFunctions(n_in=1, K=K_TEST).to(device)
    if kind == 'mix-gauss':
        return MixedTestFunctions(1, K_TEST, activation='gauss', frac_wave=0.75,
                                  init_scale=4.0, seed=seed).to(device)
    raise ValueError(kind)


def run(config, seed, n_epochs, verbose=False):
    """config: 'manuscript' (sin + biased loss, Adam paper lr),
    'mixed' (75/25 mix-gauss + U-statistic, Adam), or
    'mixed_biased' (75/25 mix-gauss + the MANUSCRIPT's biased loss, same lr as
    'mixed' -- isolates whether the U-statistic estimator is doing the work,
    or whether the wavelet test functions alone (with the ORIGINAL loss) are
    enough. Only the loss formula differs from 'mixed'; test functions, lr,
    everything else held fixed."""
    torch.manual_seed(seed); np.random.seed(seed)
    pf = A.SteadyPushforward(d_base=D_BASE, n_out=1, hidden=HIDDEN).to(device)

    if config == 'manuscript':
        tf = make_tf('sin')
        loss_impl = A.steady_loss
        lr_gen = A.LR_GEN_PAPER            # 2e-3, the paper's own Adam lr
    elif config == 'mixed':
        tf = make_tf('mix-gauss', seed=seed)
        loss_impl = A.steady_loss_ustat
        lr_gen = 5e-3                      # best Adam lr found in this study
    elif config == 'mixed_biased':
        tf = make_tf('mix-gauss', seed=seed)
        loss_impl = A.steady_loss          # the manuscript's own loss, unchanged
        lr_gen = 5e-3                      # SAME lr as 'mixed', for a clean ablation
    else:
        raise ValueError(config)

    gen_opt = torch.optim.Adam(pf.parameters(), lr=lr_gen)
    test_opt = torch.optim.SGD(tf.parameters(), lr=A.LR_TEST_PAPER)
    cur = [SIGMA2]

    def s2fn(x):
        return cur[0] * torch.ones(x.shape[0], 1, device=x.device)

    def loss_fn():
        return loss_impl(pf, tf, M=BATCH_M, drift_fn=drift_fn, sigma2_fn=s2fn,
                         base_sampler=A.gaussian_base_sampler, device=device)

    losses = np.empty(n_epochs)
    t0 = time.time()
    diverged = False
    for ep in range(n_epochs):
        cur[0] = A.sigma2_at_epoch(ep, n_epochs, True, A.SIGMA2_START, SIGMA2)
        if ep % A.ADV_EVERY == 0 and ep > 0:
            for _ in range(A.ADV_STEPS):
                l, _ = loss_fn()
                test_opt.zero_grad(); (-l).backward(); test_opt.step()
        l, _ = loss_fn()
        gen_opt.zero_grad(); l.backward(); gen_opt.step()
        lv = l.item()
        losses[ep] = lv
        if not np.isfinite(lv):
            diverged = True
            break
        if verbose and (ep % 4000 == 0 or ep == n_epochs - 1):
            print(f"    epoch {ep:6d}/{n_epochs}  s2={cur[0]:.3f}  loss={lv:.4e}",
                  flush=True)
    dt = time.time() - t0

    rec = {'config': config, 'seed': seed, 'n_epochs': n_epochs,
           'train_time_s': dt, 'diverged': diverged}
    x = None
    if diverged:
        rec.update({k: float('nan') for k in
                    ('hellinger', 'tv', 'frac_right', 'frac_right_err',
                     'mean', 'std', 'asymmetry_captured', 'frac_in_window')})
        rec['direction_correct'] = False
    else:
        with torch.no_grad():
            x = pf(torch.randn(N_EVAL, D_BASE, device=device)).cpu().numpy().ravel()
        rec.update(evaluate(x))
    return rec, x


def set_a(a_new):
    """Reset the module-level well half-separation and everything derived
    from it (V, drift_fn, true stats/moments). Needed because A_WELL is
    referenced by closure in drift_fn/V rather than passed as an argument --
    this keeps run()'s signature unchanged from the a=0.7 version."""
    global A_WELL, Z_TRUE, TRUE_FRAC_RIGHT, TRUE_FRAC_LEFT, TRUE_MEAN, TRUE_STD
    A_WELL = a_new
    Z_TRUE, TRUE_FRAC_RIGHT = true_stats()
    TRUE_FRAC_LEFT = 1.0 - TRUE_FRAC_RIGHT
    TRUE_MEAN, TRUE_STD = true_moments()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=N_EPOCHS_DEFAULT)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--a-well', type=float, default=A_WELL)
    ap.add_argument('--tag', default=None,
                    help='output file tag (default: derived from --a-well)')
    args = ap.parse_args()
    if args.a_well != A_WELL:
        set_a(args.a_well)
    tag = args.tag or f"a{A_WELL:g}"


    print(f"device: {device}")
    print(f"CLOSER WELLS: V(x) = (x^2 - {A_WELL}^2)^2 + {EPS}x,  sigma^2={SIGMA2}")
    print(f"  well separation {2*A_WELL:.2f}  (vs 2.00 for the R1.5 example -- "
          f"{100*(1-A_WELL):.0f}% closer)")
    print(f"  true split: left={TRUE_FRAC_LEFT:.4f}  right={TRUE_FRAC_RIGHT:.4f}  "
          f"(R1.5's was 0.784 / 0.216)")
    print(f"  true mean={TRUE_MEAN:.4f}  std={TRUE_STD:.4f}")
    print()

    recs, samples = [], {}
    for config in ('manuscript', 'mixed'):
        for seed in range(args.seeds):
            print(f"[{config}] seed={seed} ...", flush=True)
            rec, x = run(config, seed, args.epochs)
            print(f"    frac_right={rec['frac_right']:.4f} (true {TRUE_FRAC_RIGHT:.4f})  "
                  f"H={rec['hellinger']:.4f}  captured="
                  f"{100*rec['asymmetry_captured']:.1f}%  ({rec['train_time_s']:.0f}s)"
                  f"{'' if rec['direction_correct'] else '  [WRONG WELL]'}"
                  f"{'  DIVERGED' if rec['diverged'] else ''}", flush=True)
            recs.append(rec)
            if x is not None and seed == 0:
                samples[f"{config}_s0"] = x.astype(np.float32)
            with open(os.path.join(OUTDIR, f'closer_wells_{tag}.json'), 'w') as fh:
                json.dump({'a_well': A_WELL, 'eps': EPS, 'sigma2': SIGMA2,
                          'true_frac_right': TRUE_FRAC_RIGHT,
                          'true_mean': TRUE_MEAN, 'records': recs}, fh, indent=2)
            np.savez_compressed(os.path.join(OUTDIR, f'closer_wells_{tag}_samples.npz'),
                                **samples)

    summary(recs, samples, tag)


def summary(recs, samples, tag):
    import collections
    g = collections.defaultdict(list)
    for r in recs:
        if not r['diverged']:
            g[r['config']].append(r)

    print()
    print("=" * 72)
    print(f"{'config':12s} {'frac_right':>18s} {'captured':>18s} {'H':>8s} {'ok'}")
    for k in ('manuscript', 'mixed'):
        v = g[k]
        if not v:
            continue
        fr = np.array([r['frac_right'] for r in v])
        cap = np.array([100 * r['asymmetry_captured'] for r in v])
        h = np.array([r['hellinger'] for r in v])
        ok = sum(1 for r in v if r['direction_correct'])
        print(f"{k:12s} {fr.mean():8.4f}+/-{fr.std():<8.4f} "
              f"{cap.mean():8.1f}%+/-{cap.std():<7.1f} {h.mean():8.4f} {ok}/{len(v)}")
    print(f"{'true':12s} {TRUE_FRAC_RIGHT:18.4f} {100:18.1f}%")

    # ---- figure ----
    grid = np.linspace(GRID_LO, GRID_HI, 800)
    pt = true_pdf(grid)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))
    for ax, k, col, lab in zip(axes, ('manuscript', 'mixed'),
                               ('#b23a48', '#2c6e49'),
                               ('sin, manuscript loss', '75/25 mix-gauss, U-stat')):
        key = f"{k}_s0"
        if key in samples:
            ax.hist(samples[key], bins=150, range=(GRID_LO, GRID_HI), density=True,
                    color=col, alpha=0.6, edgecolor='none', label='learned (seed 0)')
        ax.plot(grid, pt, 'k-', lw=2.3, label='analytical Boltzmann')
        ax.axvline(0, color='0.5', ls=':', lw=1)
        ax.set_xlim(GRID_LO, GRID_HI)
        ax.set_xlabel('$x$'); ax.set_ylabel('density')
        ax.set_title(lab, fontsize=11)
        ax.legend(fontsize=8.5); ax.grid(alpha=0.25)
    fig.suptitle(f'Closer double well: $V(x)=(x^2-{A_WELL}^2)^2+{EPS}x$, '
                f'$\\sigma^2={SIGMA2}$ — true split '
                f'{TRUE_FRAC_LEFT:.2f}:{TRUE_FRAC_RIGHT:.2f} '
                f'(vs 0.78:0.22 for the R1.5 example), $M=1000$',
                fontsize=11.5)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    p = os.path.join(OUTDIR, f'closer_wells_{tag}.png')
    plt.savefig(p, dpi=150); plt.close()
    print(f"\n  wrote {p}")


if __name__ == '__main__':
    main()
