"""
Direct check: is the test-function ascent step actually doing anything?
Log gradient norms and per-epoch w values at fine resolution (case A, OU)
to distinguish "adversary correctly turned on but has near-zero gradient
almost immediately" from "adversary step isn't actually being applied."
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
import numpy as np
from common_wanpm import SteadyPushforward, SteadyTestFunctions, steady_loss, gaussian_base_sampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

def drift_A(x):
    return -(x - 2.0)

def sigma2_A(x):
    return 2.0 * torch.ones(x.shape[0], 1, device=x.device)

pf = SteadyPushforward(d_base=4, n_out=1, hidden=(16, 16)).to(device)
tf = SteadyTestFunctions(n_in=1, K=150).to(device)
gen_opt = torch.optim.Adam(pf.parameters(), lr=2e-2)
test_opt = torch.optim.SGD(tf.parameters(), lr=1e-1)

w0 = tf.w.detach().clone()
print("initial w[:5]:", w0.flatten()[:5].cpu().numpy())

for epoch in range(600):
    test_grad_norm = None
    if epoch % 10 == 0 and epoch > 0:
        loss, _ = steady_loss(pf, tf, M=1000, drift_fn=drift_A, sigma2_fn=sigma2_A,
                               base_sampler=gaussian_base_sampler, device=device)
        test_opt.zero_grad()
        (-loss).backward()
        test_grad_norm = tf.w.grad.norm().item()
        w_before = tf.w.detach().clone()
        test_opt.step()
        step_delta = (tf.w.detach() - w_before).norm().item()
    else:
        step_delta = None

    loss, _ = steady_loss(pf, tf, M=1000, drift_fn=drift_A, sigma2_fn=sigma2_A,
                           base_sampler=gaussian_base_sampler, device=device)
    gen_opt.zero_grad()
    loss.backward()
    gen_opt.step()

    if epoch % 50 == 0 or (test_grad_norm is not None and epoch < 100):
        cur_w_drift = (tf.w.detach() - w0).norm().item()
        print(f"epoch {epoch:4d}  loss={loss.item():.4e}  "
              f"test_grad_norm={test_grad_norm}  test_step_delta_norm={step_delta}  "
              f"cumulative_||w-w0||={cur_w_drift:.5f}")

final_drift = (tf.w.detach() - w0).norm().item()
print(f"\nFinal ||w_final - w0|| = {final_drift:.5f}  (||w0||={w0.norm().item():.5f})")
print("final w[:5]:", tf.w.detach().flatten()[:5].cpu().numpy())
