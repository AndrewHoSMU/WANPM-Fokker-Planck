"""
R2-1 final figure: Sec. 4.1 (1D OU, N(2,1)) trained WITH adversarial test functions.
Same settings as run_seeds.py, adversarial mode, seed 1:
  w ~ N(0, 1.5^2), b ~ U(0, 2pi), K=100, M=1000, net 1-10-10-1 tanh, base U(0,1),
  Adam lr 1e-2 (LinearLR -> 1e-3x, 10k epochs), test fns: SGD lr 1e-2,
  5 ascent steps every 10th epoch, 1 otherwise.
Figures follow the style of imgs/1dst_N21_dist.png and imgs/1dst_N21_LvE.png.
"""
import os, sys
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
MU, SIGMA, THETA, K, M, SEED = 2.0, 1.41421, 1.0, 100, 1000, 1
N_EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
SCHED = sys.argv[2] if len(sys.argv) > 2 else "linear"      # "linear" (first_example.py) or "step"
TAG = ("" if N_EPOCHS == 10000 else f"_{N_EPOCHS//1000}k") + ("_steplr" if SCHED == "step" else "")

torch.manual_seed(SEED)
w = nn.Parameter(1.5 * torch.randn(1, K))
b = nn.Parameter(2 * np.pi * torch.rand(K))
net = nn.Sequential(nn.Linear(1, 10), nn.Tanh(), nn.Linear(10, 10), nn.Tanh(), nn.Linear(10, 1))
gen_opt = optim.Adam(net.parameters(), lr=1e-2)
if SCHED == "step":   # halve the generator LR every 2000 epochs
    sch = optim.lr_scheduler.StepLR(gen_opt, step_size=2000, gamma=0.5)
else:
    sch = optim.lr_scheduler.LinearLR(gen_opt, start_factor=1.0, end_factor=0.001, total_iters=N_EPOCHS)
test_opt = optim.SGD([w, b], lr=1e-2)
w0 = w.detach().clone()


def loss_fn(x):
    drift_w = (-THETA * (x - MU)) @ w
    a = (SIGMA * w).square().sum(0, keepdim=True)
    ph = x @ w + b
    return (-drift_w * torch.cos(ph) + a * torch.sin(ph) / 2).mean(0).square().mean()


loss_log = np.empty(N_EPOCHS)
track = []
for ep in range(N_EPOCHS):
    x = net(torch.rand(M, 1))
    for _ in range(5 if ep % 10 == 0 else 1):        # adversary: maximize loss over (w, b)
        tl = loss_fn(x.detach())
        test_opt.zero_grad(); (-tl).backward(); test_opt.step()
    l = loss_fn(x)                                   # generator: minimize loss over theta
    gen_opt.zero_grad(); l.backward(); gen_opt.step(); sch.step()
    loss_log[ep] = l.item()
    if ep % 1000 == 0 or ep == N_EPOCHS - 1:
        with torch.no_grad():
            xt = net(torch.linspace(0, 1, 10000).reshape(-1, 1))
        track.append((ep, xt.mean().item(), xt.std().item(), loss_log[max(0, ep - 500):ep + 1].mean(), (w.detach().norm() / w0.norm()).item(), gen_opt.param_groups[0]['lr']))
        print(f"epoch {ep:6d} loss(avg500) {track[-1][3]:.3e}  mean {track[-1][1]:.3f}  std {track[-1][2]:.3f}  |w|/|w0| {track[-1][4]:.2f}  gen_lr {track[-1][5]:.1e}", flush=True)

with torch.no_grad():
    x = net(torch.linspace(0, 1, 10000).reshape(-1, 1)).numpy()   # as in first_example.py
print(f"\nlearned mean={x.mean():.6f} (true 2)  std={x.std():.6f} (true 1)  final loss={loss_log[-1]:.3e}")

# --- figure 1: density (style of imgs/1dst_N21_dist.png)
plt.figure()
plt.hist(x, bins=20, density=True, label="Learned distribution", color="skyblue", edgecolor="black")
xn = np.linspace(x.min(), x.max(), 1000)
plt.plot(xn, norm.pdf(xn, loc=2, scale=1), "r-", lw=2, label="Normal(2,1)")
plt.legend()
plt.savefig(os.path.join(OUT, f"1dst_N21_adv_dist{TAG}.png"), dpi=100)
plt.close()

# --- figure 2: loss (style of imgs/1dst_N21_LvE.png)
plt.figure()
plt.semilogy(loss_log)
plt.title("Loss vs. Epoch")
plt.savefig(os.path.join(OUT, f"1dst_N21_adv_LvE{TAG}.png"), dpi=100)
plt.close()

# --- side by side, for the manuscript figure (a)/(b)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].hist(x, bins=20, density=True, label="Learned distribution", color="skyblue", edgecolor="black")
ax[0].plot(xn, norm.pdf(xn, loc=2, scale=1), "r-", lw=2, label="Normal(2,1)"); ax[0].legend()
ax[1].semilogy(loss_log); ax[1].set_title("Loss vs. Epoch")
plt.tight_layout(); plt.savefig(os.path.join(OUT, f"1dst_N21_adv_pair{TAG}.png"), dpi=120)
