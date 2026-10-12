"""
R2-1, seed study: Sec. 4.1 with w ~ N(0, 1.5^2), b ~ U(0, 2pi), frozen vs adversarial
(SGD lr 1e-2, 5 ascent steps every 10th epoch, 1 otherwise), seeds 0..4.
Everything else as in first_example.py.
"""
import os, json, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
MU, SIGMA, THETA, K, M, N_EPOCHS = 2.0, 1.41421, 1.0, 100, 1000, 10000
W_STD = 1.5
SEEDS = range(5)


def run(args):
    mode, seed = args
    import torch, torch.nn as nn, torch.optim as optim
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    w = nn.Parameter(W_STD * torch.randn(1, K))
    b = nn.Parameter(2 * np.pi * torch.rand(K))
    net = nn.Sequential(nn.Linear(1, 10), nn.Tanh(), nn.Linear(10, 10), nn.Tanh(), nn.Linear(10, 1))
    gen_opt = optim.Adam(net.parameters(), lr=1e-2)
    sch = optim.lr_scheduler.LinearLR(gen_opt, start_factor=1.0, end_factor=0.001, total_iters=N_EPOCHS)
    test_opt = optim.SGD([w, b], lr=1e-2) if mode == "adversarial" else None
    w0 = w.detach().clone()

    def loss_fn(x):
        drift_w = (-THETA * (x - MU)) @ w
        a = (SIGMA * w).square().sum(0, keepdim=True)
        ph = x @ w + b
        return (-drift_w * torch.cos(ph) + a * torch.sin(ph) / 2).mean(0).square().mean()

    losses = np.empty(N_EPOCHS)
    for ep in range(N_EPOCHS):
        x = net(torch.rand(M, 1))
        if test_opt is not None:
            for _ in range(5 if ep % 10 == 0 else 1):
                tl = loss_fn(x.detach())
                test_opt.zero_grad(); (-tl).backward(); test_opt.step()
        l = loss_fn(x)
        gen_opt.zero_grad(); l.backward(); gen_opt.step(); sch.step()
        losses[ep] = l.item()
    with torch.no_grad():
        xs = net(torch.rand(50000, 1)).numpy().ravel()
    from scipy.stats import norm, kstest
    srt = np.sort(xs); q = (np.arange(len(srt)) + 0.5) / len(srt)
    return dict(mode=mode, seed=seed, mean=float(xs.mean()), std=float(xs.std()),
                W1=float(np.mean(np.abs(srt - norm.ppf(q, MU, 1.0)))),
                KS=float(kstest(xs, norm(MU, 1.0).cdf).statistic),
                w_shift=float((w.detach() - w0).norm() / w0.norm()),
                final_loss=float(losses[-500:].mean())), xs[:20000], losses


if __name__ == "__main__":
    jobs = [(m, s) for s in SEEDS for m in ("frozen", "adversarial")]
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(run, jobs))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from scipy.stats import norm
    rows = [r[0] for r in results]
    for r in rows:
        print(f"{r['mode']:12s} seed {r['seed']}  mean={r['mean']:.3f} std={r['std']:.3f} "
              f"W1={r['W1']:.3f} KS={r['KS']:.3f} final_loss={r['final_loss']:.2e} wshift={r['w_shift']:.3f}")
    print()
    for m in ("frozen", "adversarial"):
        rr = [r for r in rows if r["mode"] == m]
        f = lambda k: (np.mean([r[k] for r in rr]), np.std([r[k] for r in rr], ddof=1))
        print(f"{m:12s} mean {f('mean')[0]:.3f}±{f('mean')[1]:.3f} | std {f('std')[0]:.3f}±{f('std')[1]:.3f} | "
              f"W1 {f('W1')[0]:.3f}±{f('W1')[1]:.3f} | KS {f('KS')[0]:.3f}±{f('KS')[1]:.3f}")
    json.dump(rows, open(os.path.join(OUT, "seeds_std1.5.json"), "w"), indent=1)

    fig, ax = plt.subplots(2, 5, figsize=(20, 6.5), sharex=True, sharey=True)
    xg = np.linspace(-2, 6, 400)
    for (r, xs, _), (mode, seed) in zip(results, jobs):
        a = ax[0 if mode == "frozen" else 1, seed]
        a.hist(xs, bins=50, density=True, color="lightgray" if mode == "frozen" else "skyblue", edgecolor="k", lw=0.3)
        a.plot(xg, norm.pdf(xg, MU, 1), "r-")
        a.set_title(f"{mode}, seed {seed}\nmean {r['mean']:.2f}, std {r['std']:.2f}, W1 {r['W1']:.3f}", fontsize=9)
        a.set_xlim(-2, 6)
    plt.suptitle("Sec. 4.1, w~N(0,1.5^2), b~U(0,2pi): frozen (top) vs adversarial (bottom); red = N(2,1)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "sec41_seeds_std1.5.png"), dpi=120)
