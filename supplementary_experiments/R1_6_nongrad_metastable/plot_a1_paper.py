"""Paper version of the 4-panel Kramers figure (a=1.0, M=2000, seed 0).

Same data and layout as plot_a1_standalone.py, with the notation of the
manuscript: the temperature is written beta^{-1} (T is the final time elsewhere
in the paper), and the internal run description is not printed on the figure."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_kramers_closer')
A_WELL, GAMMA, T = 1.0, 1.0, 0.25
TAG = 'multimix_m2000_a1.0'

# --- temperature-annealing curriculum (matches T_at_epoch in the experiment script) ---
N_EPOCHS = 35000
T_START = 1.0
WARM, ANNEAL = 3000, 20000


def T_at_epoch(e):
    if e < WARM:
        return T_START
    if e < WARM + ANNEAL:
        p = 0.5 * (1 - np.cos(np.pi * (e - WARM) / ANNEAL))
        return T_START + (T - T_START) * p
    return T


def Uprime(x):
    return 4 * x ** 3 - 4 * A_WELL ** 2 * x


def drift_kramers(x, v):
    return v, -GAMMA * v - Uprime(x)


U = lambda x: (x ** 2 - A_WELL ** 2) ** 2
Z = quad(lambda x: np.exp(-U(x) / T), -4, 4, limit=400)[0]
x_grid = np.linspace(-2.5, 2.5, 400)
true_x_pdf = np.exp(-U(x_grid) / T) / Z
true_v_std = np.sqrt(T)

xv = np.load(os.path.join(D, f'samples_{TAG}.npz'))['multimix_s0']
x_samples, v_samples = xv[:, 0], xv[:, 1]
losses = np.load(os.path.join(D, f'loss_curves_{TAG}.npz'))['multimix_s0']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.scatter(x_samples[::20], v_samples[::20], s=2, alpha=0.2, color='steelblue')
xg = np.linspace(-2.2, 2.2, 18); vg = np.linspace(-2.2, 2.2, 18)
Xg, Vg = np.meshgrid(xg, vg)
Ug, Vfield = drift_kramers(Xg, Vg)
ax.streamplot(Xg, Vg, Ug, Vfield, color='gray', density=1.0, linewidth=0.7)
ax.plot([-A_WELL, A_WELL], [0, 0], 'kx', markersize=12, mew=3, label='true minima (v=0)')
ax.set_xlabel('x'); ax.set_ylabel('v')
ax.set_title('Phase-space samples and drift field')
ax.legend(fontsize=8)

ax = axes[0, 1]
ax.hist(x_samples, bins=100, density=True, alpha=0.6, color='steelblue', label='learned $x$')
ax.plot(x_grid, true_x_pdf, 'k-', lw=2, label=r'true $\rho_x(x)\propto e^{-\beta U(x)}$')
ax.set_xlabel('x'); ax.set_ylabel('density')
ax.set_title('x-marginal (double-well Boltzmann)')
ax.legend(fontsize=9)

ax = axes[1, 0]
ax.hist(v_samples, bins=100, density=True, alpha=0.6, color='darkorange', label='learned $v$')
v_grid = np.linspace(-3 * true_v_std, 3 * true_v_std, 300)
ax.plot(v_grid, norm.pdf(v_grid, 0, true_v_std), 'k-', lw=2,
        label=r'true $\rho_v(v)=N(0,\beta^{-1})$')
ax.set_xlabel('v'); ax.set_ylabel('density')
ax.set_title('v-marginal (Maxwell-Boltzmann)')
ax.legend(fontsize=9)

ax = axes[1, 1]
ep = np.arange(len(losses))
ax.semilogy(ep, losses, lw=0.5, color='steelblue', alpha=0.6, label='training loss')
# running median for readability
w = 200
if len(losses) > w:
    med = np.array([np.median(losses[max(0, i - w):i + 1]) for i in range(0, len(losses), 50)])
    ax.semilogy(np.arange(0, len(losses), 50), med, lw=2, color='navy', label=f'running median (w={w})')
ax.set_xlabel('epoch'); ax.set_ylabel('loss (log scale)')
ax.set_title('Training loss vs. epoch')
ax2 = ax.twinx()
ax2.plot(ep[::100], [T_at_epoch(e) for e in ep[::100]], color='firebrick', lw=1.6, ls='--',
         label=r'$\beta^{-1}$ (annealing schedule)')
ax2.set_ylabel(r'$\beta^{-1}$', color='firebrick'); ax2.tick_params(axis='y', colors='firebrick')
lines1, lab1 = ax.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc='upper right')

plt.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, 'a1_m2000_paper.png')
plt.savefig(out, dpi=150)
print(f"Saved plot to {out}")
print(f"final-1000-epoch median loss: {np.median(losses[-1000:]):.3e}")
print(f"x: learned frac(x>0)={float((x_samples>0).mean()):.4f}  v_std={float(v_samples.std()):.4f}")
