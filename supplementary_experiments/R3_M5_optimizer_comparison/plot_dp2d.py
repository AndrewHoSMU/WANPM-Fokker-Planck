"""Sec. 4.3 2D double peak: Adam vs Muon."""
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_lowdim')
recs = json.load(open(os.path.join(OUTDIR, 'dp2d.json')))
S = np.load(os.path.join(OUTDIR, 'dp2d_samples.npz'))
A = 0.16
LIM = 2.2

def V(X, Y):
    return ((X-1)**2 + (Y-1)**2) * ((X+1)**2 + (Y+1)**2)

edges = np.linspace(-LIM, LIM, 141)
c = 0.5*(edges[:-1]+edges[1:])
X, Y = np.meshgrid(c, c, indexing='ij')
PT = np.exp(-2*V(X, Y)/A); PT /= PT.sum()

order = ['Adam lr=0.01', 'Muon lr=0.03', 'Muon lr=0.1', 'Muon lr=0.2', 'Muon lr=0.5']
by = {f"{r['opt']} lr={r['lr']:g}": r for r in recs}

fig, axes = plt.subplots(2, len(order)+1, figsize=(3.3*(len(order)+1), 7.0))
for i, tag in enumerate(order):
    x = S[tag]; r = by[tag]
    H, _, _ = np.histogram2d(x[:,0], x[:,1], bins=[edges, edges])
    axes[0,i].imshow(H.T, origin='lower', cmap='magma',
                     extent=[-LIM,LIM,-LIM,LIM], aspect='equal')
    best = ' <-- best' if abs(r['hellinger'] - min(z['hellinger'] for z in recs)) < 1e-9 else ''
    axes[0,i].set_title(f"{tag}\nH={r['hellinger']:.4f}  bal={r['well_balance_err']:.3f}{best}",
                        fontsize=9, color=('#c62828' if best else 'black'))
    axes[0,i].plot([-1,1],[-1,1],'g+',ms=11,mew=2)
    axes[0,i].set_xlabel('$x_1$')
    # diagonal (minimum-energy path) profile
    s = (x[:,0]+x[:,1])/np.sqrt(2)
    axes[1,i].hist(s, bins=130, range=(-2.2,2.2), density=True,
                   color='#7ba7d1' if 'Adam' in tag else '#9a7fc4', alpha=0.75)
    sg = np.linspace(-2.2,2.2,500)
    pv = np.exp(-2*V(sg/np.sqrt(2), sg/np.sqrt(2))/A)
    axes[1,i].plot(sg, pv/np.trapezoid(pv, sg), 'k-', lw=2)
    axes[1,i].set_xlabel('$s$ along $x_1=x_2$'); axes[1,i].set_ylim(0, 2.6)
axes[0,0].set_ylabel('$x_2$'); axes[1,0].set_ylabel('density along MEP')

axes[0,-1].imshow(PT.T, origin='lower', cmap='magma',
                  extent=[-LIM,LIM,-LIM,LIM], aspect='equal')
axes[0,-1].set_title('analytical Boltzmann', fontsize=9)
axes[0,-1].plot([-1,1],[-1,1],'g+',ms=11,mew=2); axes[0,-1].set_xlabel('$x_1$')
ax = axes[1,-1]
ok = [r for r in recs if not r.get('diverged')]
ax.scatter([r['lr'] for r in ok if r['opt']=='Muon'],
           [r['hellinger'] for r in ok if r['opt']=='Muon'],
           c='#6a3d9a', s=70, label='Muon')
a = [r for r in ok if r['opt']=='Adam'][0]
ax.axhline(a['hellinger'], color='#1f77b4', ls=':', lw=2,
           label=f"Adam lr=1e-2 ({a['hellinger']:.4f})")
ax.set_xscale('log'); ax.set_xlabel('Muon learning rate')
ax.set_ylabel('Hellinger'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_title('LR sensitivity', fontsize=9.5)

fig.suptitle('Sec. 4.3  2D double peak (16,32,32,2), $\sigma$=0.4, K=200, 100k epochs, '
             '$\sigma$-annealed\nTop: 2D density (green + = true minima).  '
             'Bottom: profile along the minimum-energy path $x_1=x_2$', fontsize=12)
plt.tight_layout(rect=[0,0,1,0.90])
out = os.path.join(OUTDIR, 'dp2d_muon_comparison.png')
plt.savefig(out, dpi=140); plt.close()
print('wrote', out)
