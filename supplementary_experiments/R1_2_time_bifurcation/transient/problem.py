"""
Shared problem definition for the transient "one peak -> two peaks" study
(R1 comment 2: comparison with Feng-Zeng-Zhou's temporal normalizing flow).

SDE (manuscript convention, generator (sigma^2/2) Laplacian):
    dX = -grad V(X) dt + sigma dW,     d = 2
    V(x) = [(x1-1)^2 + (x2-1)^2] * [(x1+1)^2 + (x2+1)^2]     (manuscript Sec. 4.2)

Wells (V=0) at (1,1) and (-1,-1); saddle at the origin with V=4. Hessian of V:
+-8 at the saddle (unstable direction u = (x1+x2)/sqrt2, stable v = (x1-x2)/sqrt2),
16 at the wells. Every time scale is therefore O(0.1) and T=1 is far past the
transient. The line u=0 is invariant under the drift (its stable manifold), so
the population split is set by the initial mass on each side of u=0.

Initial condition: one Gaussian N(mu0, s0^2 I) sitting on the saddle.

Cases
    A  sigma=0.8  centred      mild noise   (relative barrier 2*4/sigma^2 = 12.5)
    B  sigma=0.4  centred      manuscript setting (relative barrier 50), 50/50 split
    C  sigma=0.4  shifted by delta=0.25 along u  -> asymmetric split ~ Phi(delta/s0)
    D  sigma=0.5  centred      sigma^2 = 0.25 (relative barrier 32)
    E  sigma=1.0  centred      sigma^2 = 1    (relative barrier 8)
"""
import numpy as np
import torch

T_FINAL = 1.0
SNAP_TIMES = (0.05, 0.1, 0.2, 0.5, 1.0)
S0 = 0.5
_D = 0.25 / np.sqrt(2.0)          # delta = 0.25 along u = (x1+x2)/sqrt2

CASES = {
    'A': dict(sigma=0.8, mu0=(0.0, 0.0), s0=S0),
    'B': dict(sigma=0.4, mu0=(0.0, 0.0), s0=S0),
    'C': dict(sigma=0.4, mu0=(_D, _D), s0=S0),
    'D': dict(sigma=0.5, mu0=(0.0, 0.0), s0=S0),
    'E': dict(sigma=1.0, mu0=(0.0, 0.0), s0=S0),
}


def V_torch(x):
    x1, x2 = x[:, 0:1], x[:, 1:2]
    r1 = (x1 - 1) ** 2 + (x2 - 1) ** 2
    r2 = (x1 + 1) ** 2 + (x2 + 1) ** 2
    return r1 * r2


def drift_torch(x):
    """-grad V, shape (batch, 2)."""
    x1, x2 = x[:, 0:1], x[:, 1:2]
    r1 = (x1 - 1) ** 2 + (x2 - 1) ** 2
    r2 = (x1 + 1) ** 2 + (x2 + 1) ** 2
    g1 = 2 * (x1 - 1) * r2 + 2 * (x1 + 1) * r1
    g2 = 2 * (x2 - 1) * r2 + 2 * (x2 + 1) * r1
    return -torch.cat([g1, g2], dim=1)


def lapV_torch(x):
    """Laplacian of V: 4(A+B) + 8*sum_i (x_i^2 - 1), shape (batch,)."""
    A = (x[:, 0] - 1) ** 2 + (x[:, 1] - 1) ** 2
    B = (x[:, 0] + 1) ** 2 + (x[:, 1] + 1) ** 2
    return 4 * (A + B) + 8 * ((x ** 2 - 1).sum(1))


def rho0_pdf_torch(case, x):
    c = CASES[case]
    mu = torch.tensor(c['mu0'], device=x.device, dtype=x.dtype)
    s0 = c['s0']
    return torch.exp(-((x - mu) ** 2).sum(1) / (2 * s0 ** 2)) / (2 * np.pi * s0 ** 2)


def V_np(x1, x2):
    return ((x1 - 1) ** 2 + (x2 - 1) ** 2) * ((x1 + 1) ** 2 + (x2 + 1) ** 2)


def rho0_sampler(case, device):
    c = CASES[case]
    mu = torch.tensor(c['mu0'], device=device, dtype=torch.float32)

    def sample(n, device=device):
        return mu + c['s0'] * torch.randn(n, 2, device=device)
    return sample


def u_coord(x):
    """(x1+x2)/sqrt2 -- the direction along which the peak splits."""
    return (x[..., 0] + x[..., 1]) / np.sqrt(2.0)


def v_coord(x):
    return (x[..., 0] - x[..., 1]) / np.sqrt(2.0)
