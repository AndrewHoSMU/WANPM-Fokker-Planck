"""
Shared building blocks for the rebuttal-stage1 experiments (JCOMP-D-26-00768).

Implements the steady-state and time-dependent weak-adversarial neural
pushforward method exactly as specified in main.tex (Sections 2-3):
  - Steady state: eq:FPE, eq:weak_form, eq:loss_function
  - Time dependent: eq:time_FPE, eq:time_weak_form, eq:pushforward_map,
    eq:empirical_loss

Kept deliberately general (isotropic-diffusion drift/diffusion callables,
configurable base dimension, configurable time-scaling power) so every
stage-1 experiment can reuse the same tested implementation instead of
re-deriving the loss.
"""

import torch
import torch.nn as nn
import numpy as np


def mlp(in_dim, out_dim, hidden=(32, 32), activation=nn.Tanh):
    layers = []
    d = in_dim
    for h in hidden:
        layers.append(nn.Linear(d, h))
        layers.append(activation())
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Steady-state FPE:  L^† rho = 0,   Lf = -b(x)·∇f - (1/2) a(x) : ∇²f
# ---------------------------------------------------------------------------

class SteadyPushforward(nn.Module):
    """F_vartheta: R^d -> R^n"""

    def __init__(self, d_base, n_out, hidden=(32, 32)):
        super().__init__()
        self.d_base = d_base
        self.net = mlp(d_base, n_out, hidden)

    def forward(self, r):
        return self.net(r)


class SteadyTestFunctions(nn.Module):
    """f^(k)(x) = sin(w^(k) . x + b^(k)), isotropic diffusion a(x) = sigma^2 I."""

    def __init__(self, n_in, K, w_scale=1.0):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_in, K) * w_scale)
        self.b = nn.Parameter(torch.rand(K) * 2 * np.pi)

    def phase(self, x):
        return x @ self.w + self.b

    def Lf(self, x, drift_fn, sigma2_fn):
        """drift_fn(x) -> (batch, n); sigma2_fn(x) -> (batch, 1) isotropic a(x)=sigma2*I."""
        phase = self.phase(x)
        cos_p = torch.cos(phase)
        sin_p = torch.sin(phase)
        b_dot_w = drift_fn(x) @ self.w                      # (batch, K)
        w_sq = (self.w ** 2).sum(dim=0, keepdim=True)        # (1, K)
        sigma2 = sigma2_fn(x)                                # (batch, 1)
        return -b_dot_w * cos_p + 0.5 * sigma2 * w_sq * sin_p


def steady_loss(pushforward, testfuncs, M, drift_fn, sigma2_fn, base_sampler, device):
    r = base_sampler(M, pushforward.d_base, device)
    x = pushforward(r)
    Lf = testfuncs.Lf(x, drift_fn, sigma2_fn)
    Ehat = Lf.mean(dim=0)
    return (Ehat ** 2).mean(), x


def gaussian_base_sampler(M, d, device):
    return torch.randn(M, d, device=device)


def uniform_base_sampler(low=0.0, high=1.0):
    def _sampler(M, d, device):
        return low + (high - low) * torch.rand(M, d, device=device)
    return _sampler


# ---------------------------------------------------------------------------
# Time-dependent FPE, eq:time_FPE / eq:pushforward_map / eq:empirical_loss
# ---------------------------------------------------------------------------

class TimeDepPushforward(nn.Module):
    """F_vartheta(t, x0, r) = x0 + t^power * Ftilde(t, x0, r).

    power=0.5 reproduces the paper's sqrt(t) Brownian-scaling ansatz
    (eq:pushforward_map); the `power` knob is what the R1-7 ablation sweeps.
    """

    def __init__(self, n_dim, d_base, hidden=(32, 32), power=0.5):
        super().__init__()
        self.n_dim = n_dim
        self.d_base = d_base
        self.power = power
        self.net = mlp(1 + n_dim + d_base, n_dim, hidden)

    def forward(self, t, x0, r):
        inp = torch.cat([t, x0, r], dim=1)
        delta = self.net(inp)
        scale = t.clamp(min=0.0) ** self.power if self.power != 0 else torch.ones_like(t)
        return x0 + scale * delta


class TimeDepTestFunctions(nn.Module):
    """f^(k)(t,x) = sin(w^(k).x + kappa^(k) t + b^(k))."""

    def __init__(self, n_in, K, w_scale=1.0, kappa_scale=1.0):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_in, K) * w_scale)
        self.kappa = nn.Parameter(torch.randn(K) * kappa_scale)
        self.b = nn.Parameter(torch.rand(K) * 2 * np.pi)

    def phase(self, x, t):
        return x @ self.w + self.kappa * t + self.b

    def f(self, x, t):
        return torch.sin(self.phase(x, t))

    def dfdt(self, x, t):
        return self.kappa * torch.cos(self.phase(x, t))

    def Lf(self, x, t, drift_fn, sigma2_fn):
        phase = self.phase(x, t)
        cos_p = torch.cos(phase)
        sin_p = torch.sin(phase)
        b_dot_w = drift_fn(t, x) @ self.w
        w_sq = (self.w ** 2).sum(dim=0, keepdim=True)
        sigma2 = sigma2_fn(t, x)
        return -b_dot_w * cos_p + 0.5 * sigma2 * w_sq * sin_p


def time_dep_loss(pushforward, testfuncs, M, M0, MT, T, rho0_sampler, drift_fn,
                   sigma2_fn, base_sampler, device, eps=1e-3):
    n_dim, d_base = pushforward.n_dim, pushforward.d_base

    # Interior term.
    #
    # NOTE ON SIGN: main.tex's eq:time_weak_form displays "df/dt + Lf" using
    # the Lf defined right below it (Lf = -(1/2)a f'' - b f', the same
    # negated-generator convention as the steady-state section). Deriving
    # d/dt E_{rho_t}[f] from eq:time_FPE via integration by parts shows this
    # combination is only a true identity (residual = 0 for the exact
    # solution) if the interior term is "df/dt - Lf", not "df/dt + Lf" -- and
    # this is confirmed empirically: plugging the *exact* analytical
    # Ornstein-Uhlenbeck transition density into "df/dt + Lf" gives an O(1)
    # nonzero loss (~2.3, independent of batch size M), while "df/dt - Lf"
    # gives ~3e-5 (pure Monte Carlo noise). It also matches what the
    # reference training scripts (e.g. Time_Dependent/double_well_1d.py)
    # actually implement: their `laplacian_term + drift_term` is +(1/2)a f''
    # + b f', i.e. -Lf in the paper's own notation. So we use "df/dt - Lf"
    # here to match the validated, working implementation; see the stage-1
    # results summary for the corresponding main.tex correction.
    t_int = eps + (T - eps) * torch.rand(M, 1, device=device)
    x0_int = rho0_sampler(M, device)
    r_int = base_sampler(M, d_base, device)
    x_int = pushforward(t_int, x0_int, r_int)
    integrand = testfuncs.dfdt(x_int, t_int) - testfuncs.Lf(x_int, t_int, drift_fn, sigma2_fn)
    E_interior = (T - eps) * integrand.mean(dim=0)

    # Initial condition term
    x0_0 = rho0_sampler(M0, device)
    t_0 = torch.zeros(M0, 1, device=device)
    E_0 = testfuncs.f(x0_0, t_0).mean(dim=0)

    # Terminal term
    x0_T = rho0_sampler(MT, device)
    r_T = base_sampler(MT, d_base, device)
    t_T = T * torch.ones(MT, 1, device=device)
    x_T = pushforward(t_T, x0_T, r_T)
    E_T = testfuncs.f(x_T, t_T).mean(dim=0)

    R = E_T - E_0 - E_interior
    return (R ** 2).mean(), x_int


# ---------------------------------------------------------------------------
# Generic adversarial training driver
# ---------------------------------------------------------------------------

def train_adversarial(loss_fn, gen_params, test_params, n_epochs, lr_gen=1e-3,
                       lr_test=1e-2, test_every=10, log_every=1000, verbose=True):
    gen_opt = torch.optim.Adam(gen_params, lr=lr_gen)
    test_opt = torch.optim.SGD(test_params, lr=lr_test)
    loss_log = []
    for epoch in range(n_epochs):
        if epoch % test_every == 0 and epoch > 0:
            loss, _ = loss_fn()
            test_opt.zero_grad()
            (-loss).backward()
            test_opt.step()

        loss, extra = loss_fn()
        gen_opt.zero_grad()
        loss.backward()
        gen_opt.step()
        loss_log.append(loss.item())

        if verbose and (epoch % log_every == 0 or epoch == n_epochs - 1):
            print(f"  epoch {epoch:6d}/{n_epochs}  loss={loss.item():.6e}")
    return np.array(loss_log)
