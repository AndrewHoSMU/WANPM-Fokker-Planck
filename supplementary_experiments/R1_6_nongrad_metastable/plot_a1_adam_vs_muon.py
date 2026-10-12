"""a=1.0: Adam vs Muon generator optimizer (same K=300 sin-majority bank, M=2000)."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm

D = os.path.join(os.path.dirname(__file__), 'results_kramers_closer')
A, T = 1.0, 0.25
U = lambda x: (x ** 2 - A ** 2) ** 2
Z = quad(lambda x: np.exp(-U(x) / T), -4, 4, limit=400)[0]
xg = np.linspace(-2.5, 2.5, 600)
true_x = np.exp(-U(xg) / T) / Z
vs_true = np.sqrt(T)
vg = np.linspace(-3 * vs_true, 3 * vs_true, 300)

muon = np.load(os.path.join(D, 'samples_multimix_m2000_a1.0.npz'))['multimix_s0']
adam = np.load(os.path.join(D, 'samples_multimix_m2000_a1.0_adam.npz'))['multimix_s0']

fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))

ax = axes[0]
ax.hist(adam[:, 0], bins=150, range=(-2.5, 2.5), density=True, color='#b23a48',
        alpha=0.55, label='Adam (lr 1e-2)  H=0.129,  P(x>0)=0.555', histtype='stepfilled')
ax.hist(muon[:, 0], bins=150, range=(-2.5, 2.5), density=True, color='#2c6e49',
        alpha=0.55, label='Muon (lr 0.2)  H=0.109,  P(x>0)=0.503', histtype='stepfilled')
ax.plot(xg, true_x, 'k-', lw=2, label='true')
ax.axvline(0, color='0.5', ls=':', lw=0.8)
ax.set_xlim(-2.3, 2.3); ax.set_xlabel('x'); ax.set_ylabel('density')
ax.set_title('x-marginal, a=1.0 (original instance)')
ax.legend(fontsize=8.5)

ax = axes[1]
ax.hist(adam[:, 1], bins=150, range=(-1.6, 1.6), density=True, color='#b23a48',
        alpha=0.55, label=f'Adam (v_std={adam[:,1].std():.3f})', histtype='stepfilled')
ax.hist(muon[:, 1], bins=150, range=(-1.6, 1.6), density=True, color='#2c6e49',
        alpha=0.55, label=f'Muon (v_std={muon[:,1].std():.3f})', histtype='stepfilled')
ax.plot(vg, norm.pdf(vg, 0, vs_true), 'k-', lw=2, label='true (std=0.500)')
ax.set_xlim(-1.6, 1.6); ax.set_xlabel('v'); ax.set_ylabel('density')
ax.set_title('v-marginal, a=1.0')
ax.legend(fontsize=8.5)

fig.suptitle('a=1.0 original Kramers instance: Adam vs Muon generator  '
             '(K=300 sin-majority bank, M=2000, seed 0)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.93])
p = os.path.join(D, 'a1_adam_vs_muon.png')
plt.savefig(p, dpi=150)
print('wrote', p)
