import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
from common_wanpm import TimeDepTestFunctions

torch.manual_seed(0)
tf = TimeDepTestFunctions(n_in=1, K=5)

THETA, MU, SIGMA = 1.0, 0.0, 1.0

def drift_fn(t, x):
    return -THETA * (x - MU)

def sigma2_fn(t, x):
    return (SIGMA ** 2) * torch.ones(x.shape[0], 1)

x = torch.tensor([[0.7], [-1.3], [2.1]], requires_grad=True)
t = torch.tensor([[0.3], [0.6], [0.9]], requires_grad=True)

f = tf.f(x, t)  # (batch, K)

# autograd df/dt
dfdt_auto = torch.autograd.grad(f.sum(), t, create_graph=True)[0].expand(-1, f.shape[1]) \
    if False else None
# need per-output grad; do it per test function via jacobian trick instead
dfdt_list = []
d2fdx2_list = []
dfdx_list = []
for k in range(f.shape[1]):
    fk = f[:, k]
    g_t = torch.autograd.grad(fk.sum(), t, create_graph=True, retain_graph=True)[0]
    g_x = torch.autograd.grad(fk.sum(), x, create_graph=True, retain_graph=True)[0]
    g_xx = torch.autograd.grad(g_x.sum(), x, create_graph=True, retain_graph=True)[0]
    dfdt_list.append(g_t)
    dfdx_list.append(g_x)
    d2fdx2_list.append(g_xx)

dfdt_auto = torch.cat(dfdt_list, dim=1)
dfdx_auto = torch.cat(dfdx_list, dim=1)
d2fdx2_auto = torch.cat(d2fdx2_list, dim=1)

# Lf_auto = -(1/2) a d2f/dx2 - b df/dx   (isotropic a = sigma2)
a = sigma2_fn(t, x)
b = drift_fn(t, x)
Lf_auto = -0.5 * a * d2fdx2_auto - b * dfdx_auto

dfdt_closed = tf.dfdt(x, t)
Lf_closed = tf.Lf(x, t, drift_fn, sigma2_fn)

print("dfdt max abs diff:", (dfdt_auto - dfdt_closed).abs().max().item())
print("Lf   max abs diff:", (Lf_auto - Lf_closed).abs().max().item())
