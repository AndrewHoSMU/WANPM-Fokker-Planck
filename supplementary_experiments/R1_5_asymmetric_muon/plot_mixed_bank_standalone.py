"""Standalone figure for the mixed test-function bank experiment: learned
density (left) and training loss vs. epoch (right). No other configuration
is shown."""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import closer_wells as C

OUTDIR = C.OUTDIR


def main():
    d = json.load(open(os.path.join(OUTDIR, 'mixed_bank_standalone.json')))
    recs = d['records']
    samples = dict(np.load(os.path.join(OUTDIR, 'mixed_bank_standalone_samples.npz')))

    cap = np.array([100 * r['asymmetry_captured'] for r in recs])
    h = np.array([r['hellinger'] for r in recs])
    fr = np.array([r['frac_right'] for r in recs])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # ---- (a) learned density, seed 0 ----
    ax = axes[0]
    grid = np.linspace(C.GRID_LO, C.GRID_HI, 800)
    pt = C.true_pdf(grid)
    if 's0' in samples:
        ax.hist(samples['s0'], bins=150, range=(C.GRID_LO, C.GRID_HI), density=True,
                color='#2c6e49', alpha=0.6, edgecolor='none', label='learned samples')
    ax.plot(grid, pt, 'k-', lw=2.3, label='analytical steady-state density')
    ax.axvline(0, color='0.5', ls=':', lw=1)
    ax.set_xlim(C.GRID_LO, C.GRID_HI)
    ax.set_xlabel('$x$'); ax.set_ylabel('density')
    ax.set_title('(a) Learned density (seed 0)', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.25)
    txt = (f"P(x>0)   {fr.mean():.3f} +/- {fr.std():.3f}\n"
          f"  true   {d['true_frac_right']:.3f}\n"
          f"Hellinger {h.mean():.3f} +/- {h.std():.3f}\n"
          f"({len(recs)} seeds)")
    ax.text(0.985, 0.975, txt, transform=ax.transAxes, va='top', ha='right',
           fontsize=8.5, family='monospace',
           bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.6', alpha=0.92))

    # ---- (b) training loss vs epoch, all seeds ----
    ax = axes[1]
    for i, r in enumerate(recs):
        log_every = r['log_every']
        ep = np.arange(len(r['loss_log'])) * log_every
        ax.semilogy(ep, r['loss_log'], lw=1.3, alpha=0.85,
                   label=f"seed {r['seed']}")
    ax.set_xlabel('epoch'); ax.set_ylabel('training loss')
    ax.set_title('(b) Training loss', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')

    ax2 = ax.twinx()
    r0 = recs[0]
    ep0 = np.arange(len(r0['sigma_log'])) * r0['log_every']
    ax2.plot(ep0, r0['sigma_log'], color='#d95f02', lw=1.6, ls='--',
            label=r'$\sigma^2$ schedule')
    ax2.set_ylabel(r'$\sigma^2$', color='#d95f02')
    ax2.tick_params(axis='y', labelcolor='#d95f02')

    fig.suptitle('Mixed plane-wave / Gaussian-wavelet test-function bank\n'
                r'$(8,100,1)$ tanh pushforward, $K=100$ test functions, $M=1000$, '
                '16{,}000 epochs'.replace('{,}', ','), fontsize=11.5)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    p = os.path.join(OUTDIR, 'mixed_bank_result.png')
    plt.savefig(p, dpi=150); plt.close()
    print(f"wrote {p}")
    print(f"captured {cap.mean():.1f}% +/- {cap.std():.1f}   H={h.mean():.4f} +/- {h.std():.4f}")


if __name__ == '__main__':
    main()
