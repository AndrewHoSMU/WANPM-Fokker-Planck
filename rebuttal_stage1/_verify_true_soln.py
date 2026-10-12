import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
from common_wanpm import TimeDepTestFunctions, time_dep_loss, gaussian_base_sampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1)

THETA, MU, SIGMA, T = 1.0, 0.0, 1.0, 1.0
X0_MEAN, X0_STD = 3.0, 0.25

def drift_fn(t, x):
    return -THETA * (x - MU)

def sigma2_fn(t, x):
    return (SIGMA ** 2) * torch.ones(x.shape[0], 1, device=x.device)

def rho0_sampler(n, device):
    return X0_MEAN + X0_STD * torch.randn(n, 1, device=device)


class TrueOUPushforward:
    """Exact OU transition: X_t | X_0 = x0 ~ N(x0 e^{-theta t}, v(t))."""
    n_dim = 1
    d_base = 1

    def __call__(self, t, x0, r):
        v = (SIGMA ** 2) / (2 * THETA) * (1 - torch.exp(-2 * THETA * t))
        return x0 * torch.exp(-THETA * t) + torch.sqrt(v.clamp(min=1e-12)) * r

    def parameters(self):
        return []


pf_true = TrueOUPushforward()
tf = TimeDepTestFunctions(n_in=1, K=2000).to(device)

losses = []
for trial in range(10):
    loss, _ = time_dep_loss(pf_true, tf, M=20000, M0=5000, MT=5000, T=T,
                             rho0_sampler=rho0_sampler, drift_fn=drift_fn,
                             sigma2_fn=sigma2_fn, base_sampler=gaussian_base_sampler,
                             device=device)
    losses.append(loss.item())
    # re-randomize test functions each trial to check across many random f's
    tf = TimeDepTestFunctions(n_in=1, K=2000).to(device)

print("loss of TRUE analytical solution across 10 random test-function draws:")
print(losses)
print("mean:", sum(losses) / len(losses))

# For comparison: loss of a clearly WRONG constant-mean-8 solution
class WrongPushforward:
    n_dim = 1
    d_base = 1

    def __call__(self, t, x0, r):
        return 8.0 + 1.0 * r

    def parameters(self):
        return []


pf_wrong = WrongPushforward()
tf2 = TimeDepTestFunctions(n_in=1, K=2000).to(device)
loss_wrong, _ = time_dep_loss(pf_wrong, tf2, M=20000, M0=5000, MT=5000, T=T,
                               rho0_sampler=rho0_sampler, drift_fn=drift_fn,
                               sigma2_fn=sigma2_fn, base_sampler=gaussian_base_sampler,
                               device=device)
print("loss of a deliberately WRONG solution (mean stuck at 8):", loss_wrong.item())
