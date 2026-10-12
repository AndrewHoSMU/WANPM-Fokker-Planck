"""Two-panel loss-vs-epoch figure: WANPM (left) and t-NF (right), case E (sigma=1).

WANPM's loss is saved directly in its .npz (every 20 epochs). t-NF's per-epoch L_r/L_ic
is not saved to json/npz -- only per-round (pi+, sw2) summaries are -- so this parses the
per-epoch lines out of its training log. Cumulative epoch count is used on the x-axis for
t-NF, with dashed lines marking adaptive-round boundaries (each round resamples the
residual/IC points from the model's own current density).

    python make_loss_figure.py --wanpm-tag E_adam_saddle --tnf-tag E_tnf_L4spl
"""
import os, re, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results')
LOGDIR = os.path.join(HERE, 'logs')

ap = argparse.ArgumentParser()
ap.add_argument('--wanpm-tag', default='E_adam_saddle')
ap.add_argument('--tnf-tag', default='E_tnf_L4spl')
ap.add_argument('--out', default=None)
args = ap.parse_args()


def parse_tnf_log(path):
    """Returns cumulative_epoch, L_r, L_ic, round_boundaries (cumulative epoch at each
    round's start)."""
    pat = re.compile(r'round (\d+) ep\s+(\d+)/(\d+)\s+L_r=(\S+)\s+L_ic=(\S+)\s+\[(\d+)s\]')
    rows = []
    for line in open(path, encoding='utf-8', errors='replace'):
        m = pat.search(line)
        if m:
            rnd, ep, n_ep, l_r, l_ic, _ = m.groups()
            rows.append((int(rnd), int(ep), int(n_ep), float(l_r), float(l_ic)))
    round_lens = {}
    for rnd, ep, n_ep, *_ in rows:
        round_lens[rnd] = n_ep
    offsets, acc = {}, 0
    for rnd in sorted(round_lens):
        offsets[rnd] = acc
        acc += round_lens[rnd]
    cum_ep = np.array([offsets[rnd] + ep for rnd, ep, *_ in rows])
    l_r = np.array([r[3] for r in rows])
    l_ic = np.array([r[4] for r in rows])
    boundaries = [offsets[r] for r in sorted(offsets) if r > 0]
    return cum_ep, l_r, l_ic, boundaries


# ---- WANPM ----
d = np.load(os.path.join(OUT, f'{args.wanpm_tag}.npz'))
w_loss = d['loss']
w_epoch = np.arange(len(w_loss)) * 20            # saved every 20 epochs

# ---- t-NF ----
t_epoch, t_lr, t_lic, boundaries = parse_tnf_log(os.path.join(LOGDIR, f'{args.tnf_tag}.log'))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

ax[0].semilogy(w_epoch, w_loss, color='#2b7bba', lw=1.2)
ax[0].set_xlabel('epoch'); ax[0].set_ylabel(r'$L_{\rm total}$ (WANPM)')
ax[0].set_title('WANPM training loss'); ax[0].grid(alpha=0.3, which='both')

ax[1].semilogy(t_epoch, t_lr, color='#d95f02', lw=1.0, label=r'$L_r$ (residual)')
ax[1].semilogy(t_epoch, t_lic, color='#7570b3', lw=1.0, label=r'$L_{\rm ic}$ (initial cond.)')
for b in boundaries:
    ax[1].axvline(b, color='gray', ls='--', lw=0.8, alpha=0.7)
ax[1].set_xlabel('cumulative epoch (dashed = adaptive-round boundary)')
ax[1].set_ylabel('t-NF loss terms')
ax[1].set_title('t-NF training loss'); ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3, which='both')

fig.suptitle(r'Training loss vs. epoch, $\sigma=1$ double-well bifurcation', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.94])
out = args.out or os.path.join(OUT, 'E_loss_comparison.png')
plt.savefig(out, dpi=140)
print('wrote', out)
print(f'WANPM: {len(w_loss)} points over {w_epoch[-1]} epochs')
print(f't-NF: {len(t_epoch)} points over {t_epoch[-1]} cumulative epochs, '
      f'round boundaries at {boundaries}')
