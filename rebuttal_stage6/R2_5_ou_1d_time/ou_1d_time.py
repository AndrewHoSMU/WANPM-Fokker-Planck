"""
Time-Dependent 1D Ornstein-Uhlenbeck Process - Gaussian Initial Condition
(Sec. 4.6 of the manuscript; replaces D:/PythonProjects/WANPM_FP/OUFP_T1S1.py)

Settings follow Sec. 3 of the manuscript:
  - pushforward map  F(t, x_0, r) = x_0 + sqrt(t) * F~(t, r),  x_0 ~ rho_0 sampled directly
  - test functions   f(t, x) = sin(w x + kappa t + b)
  - three-term loss  R = E_T - E_0 - E_interior
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=20000)
parser.add_argument('--out', type=str, default='results')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ============================================================================
# Problem Parameters
# ============================================================================
# dX_t = -θ(X_t - μ)dt + σ dW_t

SIGMA = 1.0  # Diffusion strength
THETA = 1.0  # Mean reversion rate
MU = 0.0  # Long-term mean
T = 1.0  # Final time

# Initial condition: Gaussian N(mu_0, sigma_0^2)
MU_0 = 3.0
SIGMA_0 = 0.5

# Training parameters
K = 100  # Number of test functions
M = 1000  # Batch size (interior)
M_0 = 500  # Batch size (t=0)
M_T = 500  # Batch size (t=T)
D_BASE = 2  # Base distribution dimension
N_EPOCHS = args.epochs
LR_GEN = 2e-3  # Generator learning rate (Adam)
LR_TEST = 1e-2  # Test function learning rate (SGD)
N_TEST_UPDATES = 5
TEST_UPDATE_FREQ = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

os.makedirs(args.out, exist_ok=True)


# ============================================================================
# Analytical Solution
# ============================================================================

def analytical_mean(t):
    """Mean: m(t) = mu_0·e^(-θt) + μ(1 - e^(-θt))"""
    return MU_0 * np.exp(-THETA * t) + MU * (1 - np.exp(-THETA * t))


def analytical_variance(t):
    """Variance: v(t) = sigma_0^2·e^(-2θt) + (σ²/2θ)(1 - e^(-2θt))"""
    return SIGMA_0 ** 2 * np.exp(-2 * THETA * t) + (SIGMA ** 2 / (2 * THETA)) * (1 - np.exp(-2 * THETA * t))


# ============================================================================
# Neural Networks
# ============================================================================

class PushforwardNetwork(nn.Module):
    """F_θ(t, x_0, r) = x_0 + √t * F̃_θ(t, r)"""

    def __init__(self, d_base, hidden_dims=[16, 16]):
        super().__init__()
        layers = []
        input_dim = 1 + d_base  # [t, r₁, ..., r_d]

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, t, x_0, r):
        delta = self.network(torch.cat([t, r], dim=1))
        return x_0 + torch.sqrt(t) * delta


class TestFunctions(nn.Module):
    """f^(k)(t, x) = sin(w^(k) x + κ^(k) t + b^(k))"""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_in, n_out))
        self.kappa = nn.Parameter(torch.randn(n_out))
        self.b = nn.Parameter(torch.rand(n_out) * 2 * np.pi)

    def forward(self, t, x):
        return torch.sin(x @ self.w + self.kappa * t + self.b)

    def interior_integrand(self, t, x):
        """∂f/∂t - Lf = ∂f/∂t + (σ²/2)∂²f/∂x² - θ(x-μ)∂f/∂x"""
        phase = x @ self.w + self.kappa * t + self.b
        sin_p, cos_p = torch.sin(phase), torch.cos(phase)
        df_dt = self.kappa * cos_p
        diffusion = -(SIGMA ** 2 / 2) * (self.w ** 2).sum(dim=0, keepdim=True) * sin_p
        drift = -THETA * ((x - MU) @ self.w) * cos_p
        return df_dt + diffusion + drift


def sample_x0(n):
    """x_0 ~ rho_0 = N(mu_0, sigma_0^2)"""
    return MU_0 + SIGMA_0 * torch.randn(n, 1, device=device)


def sample_batch(pushforward_net):
    """Interior and terminal samples of the pushforward map"""
    t = T * torch.rand(M, 1, device=device)
    x = pushforward_net(t, sample_x0(M), torch.randn(M, D_BASE, device=device))
    t_T = torch.full((M_T, 1), T, device=device)
    x_T = pushforward_net(t_T, sample_x0(M_T), torch.randn(M_T, D_BASE, device=device))
    return t, x, t_T, x_T


# ============================================================================
# Loss Function
# ============================================================================

def compute_loss(t, x, t_T, x_T, test_funcs):
    """Three-term weak-form loss: R = E_T - E_0 - E_interior"""
    E_interior = T * test_funcs.interior_integrand(t, x).mean(dim=0)
    E_0 = test_funcs(torch.zeros(M_0, 1, device=device), sample_x0(M_0)).mean(dim=0)
    E_T = test_funcs(t_T, x_T).mean(dim=0)
    residual = E_T - E_0 - E_interior
    return (residual ** 2).mean()


# ============================================================================
# Validation Functions
# ============================================================================

def sample_at(pushforward_net, t_val, n_samples):
    with torch.no_grad():
        t_batch = torch.full((n_samples, 1), t_val, device=device)
        r_batch = torch.randn(n_samples, D_BASE, device=device)
        return pushforward_net(t_batch, sample_x0(n_samples), r_batch).cpu().numpy().ravel()


def validate_statistics(pushforward_net, times, n_samples=5000):
    """Validate mean and variance"""
    print("\n" + "=" * 70)
    print("Validation: Mean and Variance")
    print("=" * 70)
    print(f"{'Time':>8} {'Mean (learned)':>15} {'Mean (true)':>15} {'Var (learned)':>15} {'Var (true)':>15}")
    print("-" * 70)
    for t_val in times:
        x_samples = sample_at(pushforward_net, t_val, n_samples)
        print(f"{t_val:8.3f} {x_samples.mean():15.6f} {analytical_mean(t_val):15.6f} "
              f"{x_samples.var():15.6f} {analytical_variance(t_val):15.6f}")


def plot_distributions(pushforward_net, times, n_samples=10000):
    """Plot distributions at different times"""
    n_times = len(times)
    fig, axes = plt.subplots(1, n_times, figsize=(5 * n_times, 4))
    stats = []

    for idx, t_val in enumerate(times):
        x_samples = sample_at(pushforward_net, t_val, n_samples)

        axes[idx].hist(x_samples, bins=50, density=True, alpha=0.7,
                       label='Learned', color='skyblue', edgecolor='black')

        m_true = analytical_mean(t_val)
        v_true = analytical_variance(t_val)
        x_range = np.linspace(x_samples.min(), x_samples.max(), 1000)
        axes[idx].plot(x_range, norm.pdf(x_range, loc=m_true, scale=np.sqrt(v_true)), 'r-', lw=2,
                       label=f'True N({m_true:.2f}, {v_true:.2f})')

        axes[idx].set_xlabel('x')
        axes[idx].set_ylabel('Probability Density')
        axes[idx].set_title(f't = {t_val:.3f}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

        z = (x_samples - x_samples.mean()) / x_samples.std()
        stats.append({'t': t_val, 'mean': float(x_samples.mean()), 'mean_true': float(m_true),
                      'var': float(x_samples.var()), 'var_true': float(v_true),
                      'skew': float((z ** 3).mean()), 'excess_kurtosis': float((z ** 4).mean() - 3)})

    plt.tight_layout()
    plt.savefig(os.path.join(args.out, '1D_time_OU_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    return stats


def plot_mean_variance_evolution(pushforward_net, n_time_points=100, n_samples=5000):
    """Plot evolution of mean and variance"""
    times = np.linspace(1e-3, T, n_time_points)
    samples = [sample_at(pushforward_net, t_val, n_samples) for t_val in times]
    means_learned = np.array([s.mean() for s in samples])
    vars_learned = np.array([s.var() for s in samples])
    means_true = analytical_mean(times)
    vars_true = analytical_variance(times)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(times, means_learned, 'b-', lw=2, label='Learned')
    ax1.plot(times, means_true, 'r--', lw=2, label='True')
    ax1.axhline(y=MU, color='g', linestyle=':', label=f'Equilibrium ({MU})')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Mean')
    ax1.set_title('Mean Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(times, vars_learned, 'b-', lw=2, label='Learned')
    ax2.plot(times, vars_true, 'r--', lw=2, label='True')
    equilibrium_var = SIGMA ** 2 / (2 * THETA)
    ax2.axhline(y=equilibrium_var, color='g', linestyle=':',
                label=f'Equilibrium ({equilibrium_var:.2f})')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Variance')
    ax2.set_title('Variance Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out, '1D_time_OU_mean_var.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return {'times': times, 'means_learned': means_learned, 'vars_learned': vars_learned,
            'means_true': means_true, 'vars_true': vars_true}


# ============================================================================
# Main Training Loop
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Time-Dependent Ornstein-Uhlenbeck Process, Gaussian Initial Condition")
    print("=" * 70)
    print(f"Parameters: σ={SIGMA}, θ={THETA}, μ={MU}, T={T}")
    print(f"Initial condition: N({MU_0}, {SIGMA_0 ** 2})")
    print(f"Training: K={K}, M={M}, M_0={M_0}, M_T={M_T}, d={D_BASE}, {N_EPOCHS} epochs, seed={args.seed}")
    print("=" * 70)

    pushforward_net = PushforwardNetwork(D_BASE).to(device)
    test_funcs = TestFunctions(1, K).to(device)

    gen_optimizer = optim.Adam(pushforward_net.parameters(), lr=LR_GEN)
    test_optimizer = optim.SGD(test_funcs.parameters(), lr=LR_TEST)

    loss_log = []
    t_start = time.time()

    for epoch in range(N_EPOCHS):
        t, x, t_T, x_T = sample_batch(pushforward_net)

        # Adversarial test function update
        if epoch % TEST_UPDATE_FREQ == 0 and epoch > 0:
            for _ in range(N_TEST_UPDATES):
                test_loss = compute_loss(t, x.detach(), t_T, x_T.detach(), test_funcs)
                test_optimizer.zero_grad()
                (-test_loss).backward()
                test_optimizer.step()

        # Generator update
        gen_loss = compute_loss(t, x, t_T, x_T, test_funcs)
        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()

        loss_log.append(gen_loss.item())

        if epoch % 1000 == 0 or epoch == N_EPOCHS - 1:
            x_test = sample_at(pushforward_net, 0.5, 1000)
            print(f"Epoch {epoch:5d}, Loss: {gen_loss.item():.6f}, "
                  f"t=0.5: mean={x_test.mean():.4f}, std={x_test.std():.4f}", flush=True)

    train_time = time.time() - t_start
    print(f"\nTraining completed in {train_time:.1f} s")

    validate_statistics(pushforward_net, times=[0.001, 0.1, 0.2, 0.5, 0.8, 1.0], n_samples=100000)

    # Loss convergence
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_log)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Convergence')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, '1D_time_OU_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()

    dist_stats = plot_distributions(pushforward_net, times=[0.01, 0.2, 0.5, 1.0])
    evo = plot_mean_variance_evolution(pushforward_net)

    # Error metrics on a time grid, with enough samples that sampling noise is negligible
    grid = np.linspace(1e-3, T, 100)
    big = [sample_at(pushforward_net, t_val, 200000) for t_val in grid]
    mean_err = np.array([s.mean() for s in big]) - analytical_mean(grid)
    var_err = np.array([s.var() for s in big]) - analytical_variance(grid)
    loss_arr = np.array(loss_log)
    summary = {
        'seed': args.seed, 'epochs': N_EPOCHS, 'train_time_s': train_time,
        'rmse_mean': float(np.sqrt((mean_err ** 2).mean())), 'max_abs_err_mean': float(np.abs(mean_err).max()),
        'rmse_var': float(np.sqrt((var_err ** 2).mean())), 'max_abs_err_var': float(np.abs(var_err).max()),
        'loss_first': float(loss_arr[0]),
        'first_epoch_below_1e-4': int(np.argmax(loss_arr < 1e-4)) if (loss_arr < 1e-4).any() else None,
        'loss_median_epochs_1000_2000': float(np.median(loss_arr[1000:2000])),
        'loss_median_last_1000': float(np.median(loss_arr[-1000:])),
        'distribution_panels': dist_stats,
    }
    print(json.dumps(summary, indent=2))

    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    np.savez(os.path.join(args.out, 'curves.npz'), loss=loss_arr, grid=grid, mean_err=mean_err, var_err=var_err, **evo)
    torch.save({'generator': pushforward_net.state_dict(), 'test_funcs': test_funcs.state_dict()},
               os.path.join(args.out, 'model.pt'))
