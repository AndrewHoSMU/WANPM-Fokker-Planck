"""
Follow-up to R2, comment 7: same idea (network discovers a singular density
from the weak-form loss alone), but now the singular measure's support is
NOT aligned with either coordinate axis -- it is the 45-degree diagonal
line x2=x1, so the collapsing (zero-variance) direction is the orthogonal
diagonal x2=-x1, at 135 degrees. This checks that the earlier result wasn't
somehow relying on the collapse direction being axis-aligned.

Construction: same isotropic mean-reverting drift as before,
    dX = -theta X dt + B dW,   X=(X1,X2), theta1=theta2=theta (isotropic),
but now the (rank-1) noise vector B points along the 45-degree direction
u=(1,1)/sqrt(2) instead of along a coordinate axis:
    B = sigma * u.
Because the drift is ISOTROPIC (a scalar multiple of the identity), the
stationary Lyapunov equation -theta*Sigma - theta*Sigma + B B^T = 0 gives
    Sigma = B B^T / (2 theta) = (sigma^2 / (2 theta)) * u u^T,
a rank-1 covariance whose support is exactly the line spanned by u (the
x2=x1 diagonal) and whose variance along the orthogonal direction
v=(-1,1)/sqrt(2) (the x2=-x1 diagonal) is EXACTLY zero -- a measure
singular w.r.t. 2D Lebesgue measure, concentrated on a line at 45 degrees
rather than on a coordinate axis.

The diffusion matrix a(x) = B B^T = sigma^2 u u^T is constant but NOT
diagonal in the (x1,x2) basis (it has equal off-diagonal entries), so this
needs a test-function generator that handles a general (possibly
non-diagonal) constant diffusion matrix -- implemented locally below
(FullConstDiffTestFunctions), analogous to but more general than
R2_7_learned_singular_density's DiagTestFunctions.

Base dimension kept at d=8 (> ambient dim 2, > 1 surviving stochastic
dimension) for the same "no dimensional hint" reason as the axis-aligned
version. Hyperparameters reuse the refined (noise-floor-fixed) config from
that experiment's follow-up: N_EPOCHS=5000, M=8000, K=1000, gen lr=5e-3,
adversary lr=2e-2.

The run reported in Sec. 4.3 of the revised manuscript (results/diag45_singular_density_plot.png,
results/results.json) uses the base distribution Uniform([-1,1]^8), set below via BASE_SAMPLER.
A standard-Gaussian base (gaussian_base_sampler) gives essentially the same result
(std along u 0.702, std along v 0.024).
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'rebuttal_stage1'))
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import norm

from common_wanpm import SteadyPushforward, gaussian_base_sampler, uniform_base_sampler, mlp

BASE_SAMPLER = uniform_base_sampler(-1.0, 1.0)   # base distribution of the manuscript's run

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)
OUTDIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUTDIR, exist_ok=True)

THETA, SIGMA = 1.0, 1.0
SIGMA2 = SIGMA ** 2
TRUE_SUPPORT_STD = np.sqrt(SIGMA2 / (2 * THETA))  # variance along the surviving diagonal direction

ANGLE = np.pi / 4  # 45 degrees
U = np.array([np.cos(ANGLE), np.sin(ANGLE)])   # surviving (noisy) direction: the x2=x1 diagonal
V = np.array([-np.sin(ANGLE), np.cos(ANGLE)])  # singular (zero-variance) direction: the x2=-x1 diagonal
U_t = torch.tensor(U, dtype=torch.float32, device=device)
V_t = torch.tensor(V, dtype=torch.float32, device=device)

# Constant (non-diagonal) diffusion matrix a(x) = sigma^2 * u u^T
A_DIFF = SIGMA2 * torch.outer(U_t, U_t)


class FullConstDiffTestFunctions(nn.Module):
    """f^(k)(x) = sin(w^(k).x + b^(k)), constant (possibly non-diagonal) diffusion a(x)=A."""

    def __init__(self, n_in, K, w_scale=1.0):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_in, K) * w_scale)
        self.b = nn.Parameter(torch.rand(K) * 2 * np.pi)

    def phase(self, x):
        return x @ self.w + self.b

    def Lf(self, x, drift_fn, A):
        """A: (n_in, n_in) constant diffusion matrix, need not be diagonal."""
        phase = self.phase(x)
        cos_p = torch.cos(phase)
        sin_p = torch.sin(phase)
        b_dot_w = drift_fn(x) @ self.w                       # (batch, K)
        Aw = A @ self.w                                       # (n_in, K)
        wAw = (self.w * Aw).sum(dim=0, keepdim=True)          # (1, K) = w_k^T A w_k
        return -b_dot_w * cos_p + 0.5 * wAw * sin_p


def diag45_loss(pushforward, testfuncs, M, drift_fn, A, base_sampler, device):
    r = base_sampler(M, pushforward.d_base, device)
    x = pushforward(r)
    Lf = testfuncs.Lf(x, drift_fn, A)
    Ehat = Lf.mean(dim=0)
    return (Ehat ** 2).mean(), x


def drift_fn(x):
    return -THETA * x  # isotropic decay -- rotation-invariant, so the 45deg tilt lives entirely in A_DIFF


torch.manual_seed(0)
D_BASE = 8
M, K = 8000, 1000
pf = SteadyPushforward(d_base=D_BASE, n_out=2, hidden=(32, 32)).to(device)
tf = FullConstDiffTestFunctions(n_in=2, K=K).to(device)
gen_opt = torch.optim.Adam(pf.parameters(), lr=5e-3)
test_opt = torch.optim.SGD(tf.parameters(), lr=2e-2)

N_EPOCHS = 5000
loss_log = []
su_std_log, sv_std_log = [], []  # std along surviving (u) / singular (v) directions
t0 = time.time()
for epoch in range(N_EPOCHS):
    if epoch % 10 == 0 and epoch > 0:
        loss, _ = diag45_loss(pf, tf, M=M, drift_fn=drift_fn, A=A_DIFF,
                               base_sampler=BASE_SAMPLER, device=device)
        test_opt.zero_grad(); (-loss).backward(); test_opt.step()
    loss, x = diag45_loss(pf, tf, M=M, drift_fn=drift_fn, A=A_DIFF,
                           base_sampler=BASE_SAMPLER, device=device)
    gen_opt.zero_grad(); loss.backward(); gen_opt.step()
    loss_log.append(loss.item())
    if epoch % 50 == 0:
        with torch.no_grad():
            su = (x @ U_t).std().item()
            sv = (x @ V_t).std().item()
            su_std_log.append(su); sv_std_log.append(sv)
    if epoch % 1000 == 0 or epoch == N_EPOCHS - 1:
        with torch.no_grad():
            print(f"  epoch {epoch:6d}/{N_EPOCHS}  loss={loss.item():.6e}  "
                  f"std(u)={(x @ U_t).std().item():.4f}  std(v)={(x @ V_t).std().item():.4f}")

train_time = time.time() - t0
print(f"training time: {train_time:.1f}s")

with torch.no_grad():
    r = BASE_SAMPLER(100000, D_BASE, device)
    x_samples = pf(r)
    s_u = (x_samples @ U_t).cpu().numpy()
    s_v = (x_samples @ V_t).cpu().numpy()
    x_samples = x_samples.cpu().numpy()

learned_u_std = float(s_u.std())
learned_v_std = float(s_v.std())
print(f"\nFinal: learned std(u)={learned_u_std:.4f} (true {TRUE_SUPPORT_STD:.4f}), "
      f"learned std(v)={learned_v_std:.4f} (true 0.0 -- singular direction, at 135deg)")
print(f"Ratio std(v)/std(u) = {learned_v_std / learned_u_std:.4f}  "
      f"(0 = perfectly singular, 1 = no collapse discovered at all)")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(22, 5))

ax = axes[0]
xr = max(3 * TRUE_SUPPORT_STD, np.abs(x_samples).max() * 1.05)
h = ax.hist2d(x_samples[:, 0], x_samples[:, 1], bins=150,
              range=[[-xr, xr], [-xr, xr]], density=True, cmap='viridis')
fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04)
diag_pts = np.array([-xr, xr])
ax.plot(diag_pts, diag_pts, color='w', linestyle='--', alpha=0.7, lw=1,
        label='true support: $x_2=x_1$ (45°)')
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_xlim(-xr, xr); ax.set_ylim(-xr, xr)
ax.set_aspect('equal')
ax.set_title('Learned density\n(true measure lies on the $x_2=x_1$ diagonal)')
ax.legend(fontsize=9)

ax = axes[1]
epochs_axis = np.arange(len(su_std_log)) * 50
ax.plot(epochs_axis, su_std_log, label='std(u) along 45° (should -> %.3f)' % TRUE_SUPPORT_STD, color='steelblue')
ax.plot(epochs_axis, sv_std_log, label='std(v) along 135° (should -> 0, singular)', color='crimson')
ax.axhline(TRUE_SUPPORT_STD, color='steelblue', linestyle=':', alpha=0.6)
ax.axhline(0, color='crimson', linestyle=':', alpha=0.6)
ax.set_xlabel('epoch'); ax.set_ylabel('marginal std')
ax.set_title('Discovering the 45° singular direction during training')
ax.legend(fontsize=9)

ax = axes[2]
ax.hist(s_u, bins=100, density=True, alpha=0.6, color='steelblue', label='learned, along 45°')
u_grid = np.linspace(-3 * TRUE_SUPPORT_STD, 3 * TRUE_SUPPORT_STD, 300)
ax.plot(u_grid, norm.pdf(u_grid, 0, TRUE_SUPPORT_STD), 'k-', lw=2, label='true marginal')
ax.set_xlabel('projection onto $u=(1,1)/\\sqrt{2}$'); ax.set_ylabel('density')
ax.set_title('Marginal along the 45° (surviving) direction')
ax.legend(fontsize=9)
marginal_xlim = ax.get_xlim()

ax = axes[3]
ax.hist(s_v, bins=100, density=True, alpha=0.6, color='crimson', label='learned, along 135°')
ax.axvline(0, color='k', linestyle='-', lw=2, label='true marginal: $\\delta_0$')
ax.set_xlim(marginal_xlim)
ax.set_xlabel('projection onto $v=(-1,1)/\\sqrt{2}$'); ax.set_ylabel('density')
ax.set_title('Marginal along the 135° (singular) direction')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'diag45_singular_density_plot.png'), dpi=150)
print(f"\nSaved plot to {OUTDIR}/diag45_singular_density_plot.png")

results = {
    'theta': THETA, 'sigma2': SIGMA2, 'd_base': D_BASE, 'M': M, 'K': K,
    'angle_deg': 45.0,
    'train_time_s': train_time,
    'true_u_std': TRUE_SUPPORT_STD, 'true_v_std': 0.0,
    'learned_u_std': learned_u_std, 'learned_v_std': learned_v_std,
    'collapse_ratio_v_over_u': learned_v_std / learned_u_std,
    'su_std_trajectory': su_std_log, 'sv_std_trajectory': sv_std_log,
}
with open(os.path.join(OUTDIR, 'results.json'), 'w') as fh:
    json.dump(results, fh, indent=2)
print(f"Saved results to {OUTDIR}/results.json")
