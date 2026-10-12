"""
Reviewer 1, comment 7: "The currently used sqrt(t) time parameterization
should also be supported by an ablation study."

Per author comment (redo): use a near-delta initial condition instead of the
original X0_STD=0.25. Rationale -- for the OU process, the exact short-time
conditional variance is v(t) = (sigma^2/2theta)(1-e^{-2theta t}) ~ sigma^2 t
as t->0, a purely diffusive law independent of the initial spread. With a
broad-ish initial condition, that early diffusive growth is partly masked by
the initial variance already present at t=0. With a near-delta IC, essentially
ALL of the variance at small t must come from the network's own t^p * Ftilde
term, so matching v(t)~t forces std(Ftilde output) ~ t^{1-p}/... concretely:
Var[t^p Ftilde] ~ t^{2p} Var[Ftilde], and matching ~t requires
Var[Ftilde] ~ t^{1-2p}. For p=0.5 this is O(1) (Ftilde's output variance
stays roughly constant near t=0 -- an easy, well-behaved target). For p=1
this requires Var[Ftilde] ~ t^{-1}, DIVERGING as t->0 -- an increasingly
pathological target for the network to represent. p=0.25 requires
Var[Ftilde] ~ t^{0.5}, vanishing (also fine, just different). So a near-delta
IC should make the difference between powers much more stark, especially at
early times -- which is why we also add finer/earlier evaluation times below
(down to t=0.01) instead of starting at t=0.05.

Same 1D time-dependent OU problem otherwise (theta=1, mu=0, sigma=1, T=1),
under the pushforward ansatz

    F_vartheta(t, x0, r) = x0 + t^p * Ftilde_vartheta(t, x0, r)

for several exponents p, holding every other setting (architecture, base
dimension, batch sizes, learning rates, epoch budget, random seed) fixed.

Metrics: (1) final training loss, (2) mean/variance error vs. the analytical
solution at several times, integrated into a single RMSE-over-time score per
power, (3) loss-vs-epoch curves to compare convergence speed.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch
import matplotlib.pyplot as plt

from common_wanpm import (TimeDepPushforward, TimeDepTestFunctions, time_dep_loss,
                           gaussian_base_sampler, train_adversarial)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

OUTDIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUTDIR, exist_ok=True)

THETA, MU, SIGMA, T = 1.0, 0.0, 1.0, 1.0
X0_MEAN, X0_STD = 3.0, 0.02  # near-delta IC (was 0.25)


def drift_fn(t, x):
    return -THETA * (x - MU)


def sigma2_fn(t, x):
    return (SIGMA ** 2) * torch.ones(x.shape[0], 1, device=x.device)


def rho0_sampler(n, device):
    return X0_MEAN + X0_STD * torch.randn(n, 1, device=device)


def analytical(t_val):
    m = X0_MEAN * np.exp(-THETA * t_val)
    v = (X0_STD ** 2) * np.exp(-2 * THETA * t_val) + (SIGMA ** 2) / (2 * THETA) * (1 - np.exp(-2 * THETA * t_val))
    return m, v


POWERS = [0.25, 0.5, 1.0, 2.0]
N_EPOCHS = 20000
EVAL_TIMES = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

all_results = {}
loss_curves = {}

for p in POWERS:
    print(f"\n{'=' * 60}\nPower p = {p}\n{'=' * 60}")
    torch.manual_seed(0)  # identical init/data stream across powers
    pf = TimeDepPushforward(n_dim=1, d_base=8, hidden=(32, 32), power=p).to(device)
    tf = TimeDepTestFunctions(n_in=1, K=1000).to(device)

    def loss_fn():
        return time_dep_loss(pf, tf, M=2000, M0=500, MT=500, T=T, rho0_sampler=rho0_sampler,
                              drift_fn=drift_fn, sigma2_fn=sigma2_fn,
                              base_sampler=gaussian_base_sampler, device=device)

    t0 = time.time()
    log = train_adversarial(loss_fn, pf.parameters(), tf.parameters(),
                             n_epochs=N_EPOCHS, lr_gen=5e-2, lr_test=1e-1, log_every=5000)
    train_time = time.time() - t0
    loss_curves[p] = log.tolist()

    errs_mean, errs_var, learned_var = [], [], []
    with torch.no_grad():
        for t_val in EVAL_TIMES:
            n = 20000
            t_b = t_val * torch.ones(n, 1, device=device)
            x0_b = rho0_sampler(n, device)
            r_b = torch.randn(n, pf.d_base, device=device)
            x = pf(t_b, x0_b, r_b).cpu().numpy()
            m_true, v_true = analytical(t_val)
            errs_mean.append(float(abs(x.mean() - m_true)))
            errs_var.append(float(abs(x.var() - v_true)))
            learned_var.append(float(x.var()))

    rmse_mean = float(np.sqrt(np.mean(np.array(errs_mean) ** 2)))
    rmse_var = float(np.sqrt(np.mean(np.array(errs_var) ** 2)))
    all_results[p] = {
        'train_time_s': train_time,
        'final_loss': float(log[-1]),
        'final_loss_smoothed': float(np.mean(log[-200:])),
        'mean_abs_err_by_t': dict(zip(EVAL_TIMES, errs_mean)),
        'var_abs_err_by_t': dict(zip(EVAL_TIMES, errs_var)),
        'learned_var_by_t': dict(zip(EVAL_TIMES, learned_var)),
        'rmse_mean_over_t': rmse_mean,
        'rmse_var_over_t': rmse_var,
    }
    print(f"p={p}: final_loss(smoothed)={all_results[p]['final_loss_smoothed']:.4e}  "
          f"RMSE(mean)={rmse_mean:.4f}  RMSE(var)={rmse_var:.4f}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(22, 5))

ax = axes[0]
for p in POWERS:
    smoothed = np.convolve(loss_curves[p], np.ones(100) / 100, mode='valid')
    ax.semilogy(smoothed, label=f'p={p}' + (' (paper: sqrt(t))' if p == 0.5 else ''))
ax.set_xlabel('epoch'); ax.set_ylabel('loss (100-epoch moving avg)')
ax.set_title('Convergence speed by time-scaling power'); ax.legend(fontsize=8)

ax = axes[1]
ax.bar([str(p) for p in POWERS], [all_results[p]['rmse_mean_over_t'] for p in POWERS],
       color=['darkorange' if p != 0.5 else 'steelblue' for p in POWERS])
ax.set_xlabel('power p'); ax.set_ylabel(f'RMSE of mean, over t in [{EVAL_TIMES[0]},1]')
ax.set_title('Accuracy: mean trajectory error')

ax = axes[2]
ax.bar([str(p) for p in POWERS], [all_results[p]['rmse_var_over_t'] for p in POWERS],
       color=['darkorange' if p != 0.5 else 'steelblue' for p in POWERS])
ax.set_xlabel('power p'); ax.set_ylabel(f'RMSE of variance, over t in [{EVAL_TIMES[0]},1]')
ax.set_title('Accuracy: variance trajectory error')

ax = axes[3]
t_arr = np.array(EVAL_TIMES)
v_true_arr = np.array([analytical(t)[1] for t in EVAL_TIMES])
ax.loglog(t_arr, v_true_arr, 'k--', lw=2.5, label='true $v(t)\\approx\\sigma^2 t$ (short-t)')
for p in POWERS:
    v_learned = np.array([all_results[p]['learned_var_by_t'][t] for t in EVAL_TIMES])
    ax.loglog(t_arr, v_learned, 'o-', markersize=4,
              label=f'p={p}' + (' (paper)' if p == 0.5 else ''))
ax.set_xlabel('t (log scale)'); ax.set_ylabel('Var[X(t)] (log scale)')
ax.set_title('Short-time variance growth\n(near-delta IC exposes power-law mismatch)')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'ablation_plot.png'), dpi=150)
print(f"\nSaved plot to {OUTDIR}/ablation_plot.png")

with open(os.path.join(OUTDIR, 'results.json'), 'w') as fh:
    json.dump(all_results, fh, indent=2)
print(f"Saved results to {OUTDIR}/results.json")
