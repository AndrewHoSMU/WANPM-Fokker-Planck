"""
Reviewer 2, comment 2: "For all the experiments, do you observe convergence
for the test functions. Can you visualize these nicely, such as a plot in
frequency space, and is there any clear connection between the true solution
and the test function space."

main.tex's own Sec. 4.1 example deliberately FREEZES the test-function
parameters (w, b) at random initialization, so it can't answer this
question. Here we run genuine adversarial training (w, b trainable via SGD
ascent) on the 2D ring example (main.tex Sec. 4.5: r0=2, omega=2, sigma=1,
rotational + radial drift) and log the test-function frequencies w^(k) -- now
2-vectors (direction + magnitude), giving a real frequency *plane* rather
than a single scalar -- over training.

Two earlier attempts on 1D targets (unimodal OU, 1D double well) showed
frequencies going essentially flat after ~40-100 epochs regardless of
adversary learning rate (tried 1e-1 and 7e-1): the generator solves an easy
1D target so fast that the adversary runs out of anything to exploit almost
immediately (confirmed not a stuck-optimizer bug via direct gradient-norm
instrumentation). The 2D ring is a genuinely harder, non-gradient target --
a ring is a 1D curve embedded in 2D, not a simple blob -- giving the
generator much more room to still be locally wrong for longer into training,
which is exactly what we see below: sustained, still-growing movement at
epoch 20,000, not a fast freeze.

Per earlier discussion in this session: we do NOT compare against any "true"
test-function configuration -- there isn't a well-defined one in this
minimax setup (at the Nash equilibrium the adversary's objective is
degenerate, not pointing at some canonical w). We only report what's
actually well-defined: trajectories of w^(k) in the 2D frequency plane, and
how much they move.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch
import matplotlib.pyplot as plt

from common_wanpm import SteadyPushforward, SteadyTestFunctions, steady_loss, gaussian_base_sampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)
OUTDIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUTDIR, exist_ok=True)

R0, OMEGA, SIGMA = 2.0, 2.0, 1.0
SIGMA2 = SIGMA ** 2
LR_TEST = 7e-1  # adversary learning rate, 7x the original 1e-1 used elsewhere in main.tex


def drift_ring(x):
    x1, x2 = x[:, 0:1], x[:, 1:2]
    r2 = x1 ** 2 + x2 ** 2
    b1_rad = -2 * x1 * (r2 - R0 ** 2)
    b2_rad = -2 * x2 * (r2 - R0 ** 2)
    b1_tan = -OMEGA * x2
    b2_tan = OMEGA * x1
    return torch.cat([b1_rad + b1_tan, b2_rad + b2_tan], dim=1)


def sigma2_fn(x):
    return SIGMA2 * torch.ones(x.shape[0], 1, device=x.device)


torch.manual_seed(0)
D_BASE, K, N_EPOCHS = 4, 200, 20000
pf = SteadyPushforward(d_base=D_BASE, n_out=2, hidden=(64,)).to(device)
tf = SteadyTestFunctions(n_in=2, K=K).to(device)
gen_opt = torch.optim.Adam(pf.parameters(), lr=1e-2)
test_opt = torch.optim.SGD(tf.parameters(), lr=LR_TEST)

w_history = []  # (n_logged, 2, K)
loss_history = []
log_every = 50

t0 = time.time()
for epoch in range(N_EPOCHS):
    if epoch % 10 == 0 and epoch > 0:
        loss, _ = steady_loss(pf, tf, M=1000, drift_fn=drift_ring, sigma2_fn=sigma2_fn,
                               base_sampler=gaussian_base_sampler, device=device)
        test_opt.zero_grad(); (-loss).backward(); test_opt.step()
    loss, _ = steady_loss(pf, tf, M=1000, drift_fn=drift_ring, sigma2_fn=sigma2_fn,
                           base_sampler=gaussian_base_sampler, device=device)
    gen_opt.zero_grad(); loss.backward(); gen_opt.step()

    if epoch % log_every == 0:
        w_history.append(tf.w.detach().cpu().numpy().copy())  # (2, K)
        loss_history.append(loss.item())
    if epoch % 4000 == 0 or epoch == N_EPOCHS - 1:
        print(f"  epoch {epoch:6d}/{N_EPOCHS}  loss={loss.item():.6e}")

train_time = time.time() - t0
print(f"training time: {train_time:.1f}s")

w_history = np.array(w_history)  # (T, 2, K)
w0 = w_history[0]
w_final = w_history[-1]

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
epochs_axis = np.arange(w_history.shape[0]) * log_every
mag_history = np.linalg.norm(w_history, axis=1)  # (T, K)
n_show = 15
idx_show = np.linspace(0, K - 1, n_show).astype(int)
for k in idx_show:
    ax.plot(epochs_axis, mag_history[:, k], alpha=0.7, lw=1)
ax.set_xlabel('epoch'); ax.set_ylabel('$|w^{(k)}|$ (frequency magnitude)')
ax.set_title(f'2D ring: test-function frequency magnitude\n(adversary LR={LR_TEST:.0e})')

ax = axes[1]
n_arrows = 60
idx_arrows = np.linspace(0, K - 1, n_arrows).astype(int)
ax.scatter(w0[0, idx_arrows], w0[1, idx_arrows], c='gray', s=15, alpha=0.6, label='initial $w^{(k)}$')
ax.scatter(w_final[0, idx_arrows], w_final[1, idx_arrows], c='crimson', s=15, alpha=0.8, label='final $w^{(k)}$')
for k in idx_arrows:
    ax.annotate('', xy=(w_final[0, k], w_final[1, k]), xytext=(w0[0, k], w0[1, k]),
                arrowprops=dict(arrowstyle='->', color='steelblue', alpha=0.5, lw=1))
ax.set_xlabel('$w_1^{(k)}$'); ax.set_ylabel('$w_2^{(k)}$')
ax.set_title('Movement in the 2D frequency plane\n(initial -> final)')
ax.legend(fontsize=9)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'frequency_analysis.png'), dpi=150)
print(f"\nSaved plot to {OUTDIR}/frequency_analysis.png")

per_k_drift = np.linalg.norm(w_final - w0, axis=0)  # (K,)
results = {
    'lr_test': LR_TEST, 'K': K, 'd_base': D_BASE, 'n_epochs': N_EPOCHS,
    'train_time_s': train_time,
    'final_loss': float(loss_history[-1]),
    'loss_at_epoch0': float(loss_history[0]),
    'mean_per_testfunc_drift_norm': float(per_k_drift.mean()),
    'max_per_testfunc_drift_norm': float(per_k_drift.max()),
    'mean_initial_w_norm': float(np.linalg.norm(w0, axis=0).mean()),
    'mean_final_w_norm': float(np.linalg.norm(w_final, axis=0).mean()),
}
print(json.dumps(results, indent=2))
with open(os.path.join(OUTDIR, 'results.json'), 'w') as fh:
    json.dump(results, fh, indent=2)
