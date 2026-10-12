"""
Reviewer 2, comment 1: Sec. 4.1 (1D OU, exact N(2,1)) re-run WITH adversarial
test functions. Single run, seed 0. Setup identical to first_example.py except
that (w, b) are trained by SGD ascent, using the recipe of the other examples
(DoublePeak1D.py): SGD lr 1e-2, 5 ascent steps every 10th epoch, 1 otherwise.
"""
import os
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
MU, SIGMA, THETA, K, M, N_EPOCHS = 2.0, 1.41421, 1.0, 100, 1000, 10000

torch.manual_seed(0)
w = nn.Parameter(1.2 * torch.randn(1, K))            # w ~ N(0, 1.2^2)  (std 1.2)
b = nn.Parameter(2 * np.pi * torch.rand(K))          # b ~ U(0, 2pi)
net = nn.Sequential(nn.Linear(1, 10), nn.Tanh(), nn.Linear(10, 10), nn.Tanh(), nn.Linear(10, 1))
gen_opt = optim.Adam(net.parameters(), lr=1e-2)
sch = optim.lr_scheduler.LinearLR(gen_opt, start_factor=1.0, end_factor=0.001, total_iters=N_EPOCHS)
test_opt = optim.SGD([w, b], lr=1e-2)
w0 = w.detach().clone()


def loss_fn(x):
    drift_w = (-THETA * (x - MU)) @ w
    a = (SIGMA * w).square().sum(0, keepdim=True)
    ph = x @ w + b
    return (-drift_w * torch.cos(ph) + a * torch.sin(ph) / 2).mean(0).square().mean()


loss_log = np.empty(N_EPOCHS)
for ep in range(N_EPOCHS):
    x = net(torch.rand(M, 1))
    for _ in range(5 if ep % 10 == 0 else 1):        # adversary: maximize loss over (w, b)
        tl = loss_fn(x.detach())
        test_opt.zero_grad(); (-tl).backward(); test_opt.step()
    l = loss_fn(x)                                   # generator: minimize loss over theta
    gen_opt.zero_grad(); l.backward(); gen_opt.step(); sch.step()
    loss_log[ep] = l.item()
    if ep % 1000 == 0 or ep == N_EPOCHS - 1:
        print(f"epoch {ep:5d} loss {l.item():.3e}", flush=True)

with torch.no_grad():
    xs = net(torch.rand(50000, 1)).numpy().ravel()
print(f"\nlearned mean={xs.mean():.4f} (true 2)   std={xs.std():.4f} (true 1)")
print(f"relative change of test-function frequencies ||w-w0||/||w0|| = {(w.detach()-w0).norm()/w0.norm():.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(xs, bins=50, density=True, color="skyblue", edgecolor="k", label="Learned (adversarial)")
xg = np.linspace(xs.min(), xs.max(), 500)
ax[0].plot(xg, norm.pdf(xg, MU, 1), "r-", lw=2, label="N(2,1)"); ax[0].legend(); ax[0].set_xlabel("x")
ax[1].semilogy(loss_log); ax[1].set_xlabel("epoch"); ax[1].set_title("Generator loss")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "sec41_adversarial_single.png"), dpi=140)
