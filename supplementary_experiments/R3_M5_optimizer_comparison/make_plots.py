"""Figures + summary tables for the R3_M5 optimizer comparison."""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ORDER = ['SGD', 'Adam', 'AdamW', 'Shampoo', 'ScalableShampoo', 'SOAP', 'Muon']
COLORS = {
    'SGD': '#8c8c8c', 'Adam': '#1f77b4', 'AdamW': '#4fa3d1',
    'Shampoo': '#d62728', 'ScalableShampoo': '#ff9896',
    'SOAP': '#2ca02c', 'Muon': '#9467bd',
}


def _load(outdir, name):
    path = os.path.join(outdir, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _agg(records):
    """group by gen_opt -> mean/std of metrics across seeds"""
    out = {}
    for name in ORDER:
        rs = [r for r in records if r['gen_opt'] == name]
        if not rs:
            continue
        ok = [r for r in rs if not r['diverged'] and np.isfinite(r['hellinger'])]
        d = {'n': len(rs), 'n_diverged': len(rs) - len(ok),
             'lr_gen': rs[0]['lr_gen']}
        for k in ['hellinger', 'tv', 'well_balance_err', 'std', 'train_time_s',
                  'final_loss']:
            vals = np.array([r[k] for r in ok], dtype=float)
            d[k + '_mean'] = float(vals.mean()) if len(vals) else float('nan')
            d[k + '_std'] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out[name] = d
    return out


def _md_table(agg, title):
    lines = [f"### {title}", "",
             "| optimizer | best lr | Hellinger | TV | well-balance err | learned std | s/run | diverged |",
             "|---|---|---|---|---|---|---|---|"]
    for name in ORDER:
        if name not in agg:
            continue
        d = agg[name]
        lines.append(
            f"| {name} | {d['lr_gen']:g} | "
            f"{d['hellinger_mean']:.4f} ± {d['hellinger_std']:.4f} | "
            f"{d['tv_mean']:.4f} ± {d['tv_std']:.4f} | "
            f"{d['well_balance_err_mean']:.4f} ± {d['well_balance_err_std']:.4f} | "
            f"{d['std_mean']:.4f} | {d['train_time_s_mean']:.0f} | "
            f"{d['n_diverged']}/{d['n']} |")
    lines.append("")
    lines.append("True std = 0.9719; perfect well balance = 0.0000.")
    return "\n".join(lines)


def main(outdir):
    sweep = _load(outdir, 'sweep.json')
    seeds = _load(outdir, 'seeds.json')
    adv = _load(outdir, 'adversary.json')

    # ---------------- Figure 1: LR sensitivity ----------------
    if sweep:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        for name in ORDER:
            rs = sorted([r for r in sweep if r['gen_opt'] == name],
                        key=lambda r: r['lr_gen'])
            if not rs:
                continue
            lrs = [r['lr_gen'] for r in rs]
            hs = [r['hellinger'] if not r['diverged'] else np.nan for r in rs]
            ax.plot(lrs, hs, 'o-', label=name, color=COLORS.get(name), lw=1.8, ms=5)
        ax.set_xscale('log'); ax.set_xlabel('generator learning rate')
        ax.set_ylabel('Hellinger distance to analytical Boltzmann')
        ax.set_title('Learning-rate sensitivity (seed 0)')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[1]
        for name in ORDER:
            rs = sorted([r for r in sweep if r['gen_opt'] == name],
                        key=lambda r: r['lr_gen'])
            if not rs:
                continue
            lrs = [r['lr_gen'] for r in rs]
            bs = [r['well_balance_err'] if not r['diverged'] else np.nan for r in rs]
            ax.plot(lrs, bs, 'o-', label=name, color=COLORS.get(name), lw=1.8, ms=5)
        ax.axhline(0.5, color='k', ls=':', alpha=0.5)
        ax.text(ax.get_xlim()[0], 0.5, ' total collapse into one well',
                va='bottom', fontsize=8, color='k')
        ax.set_xscale('log'); ax.set_xlabel('generator learning rate')
        ax.set_ylabel('|P(x>0) - 0.5|')
        ax.set_title('Mode-balance error')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'lr_sensitivity.png'), dpi=150)
        plt.close()
        print("  wrote lr_sensitivity.png")

    # ---------------- Figure 2: seed comparison ----------------
    if seeds:
        agg = _agg(seeds)
        names = [n for n in ORDER if n in agg]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        ax = axes[0]
        means = [agg[n]['hellinger_mean'] for n in names]
        errs = [agg[n]['hellinger_std'] for n in names]
        ax.bar(range(len(names)), means, yerr=errs, capsize=4,
               color=[COLORS.get(n) for n in names], alpha=0.85)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Hellinger distance')
        ax.set_title(f'Accuracy vs. analytical solution\n(mean ± sd over seeds, each at its own best LR)')
        ax.grid(alpha=0.3, axis='y')

        ax = axes[1]
        for name in names:
            rs = [r for r in seeds if r['gen_opt'] == name and not r['diverged']]
            if not rs:
                continue
            curves = np.array([r['loss_log_every50'] for r in rs], dtype=float)
            med = np.nanmedian(curves, axis=0)
            ep = np.arange(len(med)) * 50
            ax.semilogy(ep, med, label=name, color=COLORS.get(name), lw=1.5)
        ax.set_xlabel('epoch'); ax.set_ylabel('training loss (median over seeds)')
        ax.set_title('Loss curves\n(NOT a fair accuracy ranking - minimax objective)')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = axes[2]
        t = [agg[n]['train_time_s_mean'] for n in names]
        ax.bar(range(len(names)), t, color=[COLORS.get(n) for n in names], alpha=0.85)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('wall-clock seconds per 16k-epoch run')
        ax.set_title('Cost')
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'optimizer_comparison.png'), dpi=150)
        plt.close()
        print("  wrote optimizer_comparison.png")

        with open(os.path.join(outdir, 'summary.md'), 'w') as fh:
            fh.write(_md_table(agg, 'Generator-side optimizer comparison '
                                    '(main.tex Sec. 4.2, seed-averaged)'))
            fh.write("\n\n")
            if adv:
                fh.write("### Adversary-side sweep (generator held at Adam lr 2e-3)\n\n")
                fh.write("| test-fn optimizer | lr | Hellinger | TV | well-balance err |\n")
                fh.write("|---|---|---|---|---|\n")
                for r in adv:
                    h = 'diverged' if r['diverged'] else f"{r['hellinger']:.4f}"
                    tv = 'diverged' if r['diverged'] else f"{r['tv']:.4f}"
                    be = 'diverged' if r['diverged'] else f"{r['well_balance_err']:.4f}"
                    fh.write(f"| {r['test_opt']} | {r['lr_test']:g} | {h} | {tv} | {be} |\n")
        print("  wrote summary.md")
        print()
        print(_md_table(agg, 'Generator-side optimizer comparison'))


if __name__ == '__main__':
    main(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'))
