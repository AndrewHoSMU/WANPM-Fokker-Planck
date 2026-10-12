"""
Reviewer 2, comment 1: "For the first test case, in 4.1, why do you not use the
adversarial training in this case when all others do so?"

Sec. 4.1 (1D OU, steady state, exact solution N(2,1)) froze the test-function
parameters (w, b) at their random initialisation.  This script re-runs exactly
that experiment (first_example.py: net 1-10-10-1 tanh, base U(0,1), K=100 plane
waves sin(w x + b), M=1000, Adam lr 1e-2 with LinearLR to 1e-3x over 10k epochs)
in three modes:

  frozen        -- the manuscript's original Sec. 4.1 (no adversary)
  adversarial   -- the recipe used in every other example of the paper
                   (DoublePeak1D.py): SGD ascent on (w, b), lr 1e-2,
                   5 ascent steps every 10 generator epochs, 1 step otherwise
  adv_strong    -- same, lr 1e-1 (sensitivity check on the adversary strength)

Everything else (generator, data, seeds) is identical, so the comparison
isolates the effect of the adversary.  Repeated over several seeds.
"""
import os, json, time, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm, kstest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(4)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

MU, SIG_NOISE, THETA = 2.0, 1.41421, 1.0   # OU: N(2, 1)
K, M, N_EPOCHS = 100, 1000, 10000
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def loss_fn(x, w, b):
    """mean_k ( (1/M) sum_m Lf_k(x_m) )^2, identical to compute_loss in first_example.py."""
    drift_w = (-THETA * (x - MU)) @ w                # (M,K)
    a = (SIG_NOISE * w).square().sum(0, keepdim=True)  # (1,K)
    ph = x @ w + b
    return (-drift_w * torch.cos(ph) + a * torch.sin(ph) / 2).mean(0).square().mean()


def run(mode, seed):
    torch.manual_seed(seed)
    w = nn.Parameter(torch.randn(1, K))
    b = nn.Parameter(torch.randn(K))
    net = nn.Sequential(nn.Linear(1, 10), nn.Tanh(), nn.Linear(10, 10), nn.Tanh(), nn.Linear(10, 1))
    gen_opt = optim.Adam(net.parameters(), lr=1e-2)
    sch = optim.lr_scheduler.LinearLR(gen_opt, start_factor=1.0, end_factor=0.001, total_iters=N_EPOCHS)
    lr_test = {"frozen": 0.0, "adversarial": 1e-2, "adv_strong": 1e-1}[mode]
    test_opt = optim.SGD([w, b], lr=lr_test) if lr_test > 0 else None
    w0 = w.detach().clone()

    losses = np.empty(N_EPOCHS)
    mean_std = []
    for ep in range(N_EPOCHS):
        r = torch.rand(M, 1)
        x = net(r)
        if test_opt is not None:
            for _ in range(5 if ep % 10 == 0 else 1):
                tl = loss_fn(x.detach(), w, b)
                test_opt.zero_grad()
                (-tl).backward()
                test_opt.step()
        l = loss_fn(x, w, b)          # generator loss uses the (possibly updated) test functions
        gen_opt.zero_grad()
        l.backward()
        gen_opt.step()
        sch.step()
        losses[ep] = l.item()
        if ep % 100 == 0:
            with torch.no_grad():
                xs = net(torch.rand(5000, 1))
            mean_std.append((ep, xs.mean().item(), xs.std().item()))

    with torch.no_grad():
        xs = net(torch.rand(50000, 1)).numpy().ravel()
        # loss evaluated on a *fixed, common* random test bank so modes are comparable
        g = torch.Generator().manual_seed(12345)
        w_eval = torch.randn(1, 2000, generator=g)
        b_eval = torch.randn(2000, generator=g)
        eval_loss = loss_fn(net(torch.rand(50000, 1)), w_eval, b_eval).item()
    srt = np.sort(xs)
    q = (np.arange(len(srt)) + 0.5) / len(srt)
    w1 = np.mean(np.abs(srt - norm.ppf(q, MU, 1.0)))
    ks = kstest(xs, norm(MU, 1.0).cdf).statistic
    return dict(mode=mode, seed=seed, mean=float(xs.mean()), std=float(xs.std()),
                W1=float(w1), KS=float(ks), eval_loss_fixed_bank=eval_loss,
                final_train_loss=float(losses[-100:].mean()),
                w_shift=float((w.detach() - w0).norm() / w0.norm()),
                losses=losses, mean_std=mean_std, samples=xs[:20000])


if __name__ == "__main__":
    modes = ["frozen", "adversarial", "adv_strong"]
    res = {m: [] for m in modes}
    t0 = time.time()
    for s in range(N_SEEDS):
        for m in modes:
            r = run(m, s)
            res[m].append(r)
            print(f"seed {s} {m:12s} mean={r['mean']:.4f} std={r['std']:.4f} W1={r['W1']:.4f} "
                  f"KS={r['KS']:.4f} evalL={r['eval_loss_fixed_bank']:.2e} wshift={r['w_shift']:.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- summary
    summary = {}
    print("\n=== summary over", N_SEEDS, "seeds (mean +- sd; median) ===")
    for m in modes:
        row = {}
        for key in ["mean", "std", "W1", "KS", "eval_loss_fixed_bank", "w_shift"]:
            v = np.array([r[key] for r in res[m]])
            row[key] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                            median=float(np.median(v)), min=float(v.min()), max=float(v.max()))
        row["abs_mean_err"] = float(np.mean([abs(r["mean"] - MU) for r in res[m]]))
        row["abs_std_err"] = float(np.mean([abs(r["std"] - 1.0) for r in res[m]]))
        summary[m] = row
        print(f"{m:12s} mean={row['mean']['mean']:.4f}±{row['mean']['sd']:.4f}  "
              f"std={row['std']['mean']:.4f}±{row['std']['sd']:.4f}  "
              f"|std-1|={row['abs_std_err']:.4f}  W1={row['W1']['mean']:.4f}±{row['W1']['sd']:.4f} "
              f"(med {row['W1']['median']:.4f})  KS={row['KS']['mean']:.4f}±{row['KS']['sd']:.4f}  "
              f"wshift={row['w_shift']['mean']:.3f}")
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(dict(config=dict(K=K, M=M, N_EPOCHS=N_EPOCHS, N_SEEDS=N_SEEDS), summary=summary,
                       per_seed={m: [{k: v for k, v in r.items() if k not in ("losses", "mean_std", "samples")}
                                     for r in res[m]] for m in modes}), f, indent=1)

    # ---- figure
    col = {"frozen": "tab:gray", "adversarial": "tab:blue", "adv_strong": "tab:red"}
    lab = {"frozen": "frozen test fns (original Sec. 4.1)", "adversarial": "adversarial, SGD lr 1e-2 (paper recipe)",
           "adv_strong": "adversarial, SGD lr 1e-1"}
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.3))
    xg = np.linspace(-2, 6, 400)
    ax[0].plot(xg, norm.pdf(xg, MU, 1), "k-", lw=2, label="exact N(2,1)")
    for m in modes:
        ax[0].hist(res[m][0]["samples"], bins=60, density=True, histtype="step", lw=1.6, color=col[m], label=lab[m])
    ax[0].set_title("Learned density (seed 0)"); ax[0].legend(fontsize=7); ax[0].set_xlabel("x")
    for m in modes:
        L = np.array([r["losses"] for r in res[m]])
        ax[1].semilogy(np.median(L, 0), color=col[m], label=lab[m], lw=0.8)
    ax[1].set_title("Generator loss (median over seeds)"); ax[1].set_xlabel("epoch"); ax[1].legend(fontsize=7)
    for m in modes:
        ax[2].plot(np.arange(len(res[m][0]["mean_std"])) * 100,
                   np.median([[e[2] for e in r["mean_std"]] for r in res[m]], 0), color=col[m], lw=1.2)
    ax[2].axhline(1.0, color="k", ls="--"); ax[2].set_title("Sample std vs epoch (median; exact = 1)")
    ax[2].set_xlabel("epoch")
    pos = np.arange(len(modes))
    for i, key in enumerate(["W1"]):
        for j, m in enumerate(modes):
            v = [r[key] for r in res[m]]
            ax[3].scatter(np.full(len(v), j) + np.random.RandomState(0).uniform(-.12, .12, len(v)), v,
                          color=col[m], s=25)
            ax[3].hlines(np.mean(v), j - .25, j + .25, color="k")
    ax[3].set_xticks(pos); ax[3].set_xticklabels(["frozen", "adv 1e-2", "adv 1e-1"])
    ax[3].set_yscale("log"); ax[3].set_ylabel("Wasserstein-1 to N(2,1)"); ax[3].set_title(f"W1 per seed ({N_SEEDS} seeds)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "sec41_adversarial.png"), dpi=140)
    print("saved figure")
