"""
Stage 0 for the t-NF baseline: reproduce the paper's own toy example (Feng-Zeng-Zhou, Sec 4.1,
eq 4.1-4.2) with the paper's hyper-parameters, to check that tnf.py is a faithful implementation
before it is used as a baseline.

    p_t - (1/2) Lap p = 0 on R^2, t in [0,1],   p(x,0) = N((4,4), I),
    exact:  p(x,t) = N((4,4), (1+t) I)

Paper settings: L=6 affine-coupling + actnorm layers, spline off, N_e=20, alpha=2, N_adaptive=5,
batch 1000, N_x=1000 spatial points per time slice, N_t=20 time slices, Adam 1e-3, initial training
points uniform in [-3,3]^2 (which does not even contain the mean (4,4)).
Paper result at t=1 (their Fig. 4): relative L2 ~ 2e-3 and relative KL ~ 5e-5 after 5 rounds.
Metrics as in the paper: relative L2 on a grid over [0,8]^2, and
relative KL = E_{p*}[log p* - log p_theta] / H(p*), by Monte Carlo (here 10^5 samples).
"""
import sys, os, json, time, math, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import tnf

device = tnf.device
T, S2 = 1.0, 1.0
MU0 = torch.tensor([4.0, 4.0], device=device)


def ic_pdf(x):
    return torch.exp(-0.5 * ((x - MU0) ** 2).sum(1)) / (2 * math.pi)


def exact_logp(x, t):
    v = 1.0 + t
    return -((x - MU0) ** 2).sum(1) / (2 * v) - math.log(2 * math.pi * v)


def residual_heat(model, x, t):
    return tnf.residual(model.log_prob, lambda x_: torch.zeros_like(x_),
                        lambda x_: torch.zeros(x_.shape[0], device=x_.device, dtype=x_.dtype), x, t, S2)


@torch.no_grad()
def evaluate(model, t_eval, n_grid=160, n_mc=100_000):
    g = torch.linspace(0, 8, n_grid, device=device)
    X = torch.stack(torch.meshgrid(g, g, indexing='ij'), -1).reshape(-1, 2)
    out = {}
    for t in t_eval:
        tt = t * torch.ones(X.shape[0], 1, device=device)
        p_th, p_ex = model.log_prob(X, tt).exp(), exact_logp(X, t).exp()
        l2 = float((p_th - p_ex).norm() / p_ex.norm())
        xs = MU0 + math.sqrt(1 + t) * torch.randn(n_mc, 2, device=device)
        tm = t * torch.ones(n_mc, 1, device=device)
        lp_ex, lp_th = exact_logp(xs, t), model.log_prob(xs, tm)
        kl = float((lp_ex - lp_th).mean() / (-lp_ex.mean()))
        out[t] = (l2, kl)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lam-ic', type=float, default=10.0)
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='heat_sanity')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    model = tnf.TNF(2, L=6, width=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_x, n_t, batch, box = 1000, 20, 1000, 3.0
    N_r = n_x * n_t
    x, t, xi = tnf.draw_points(model, None, N_r, T, box, True, n_x)
    model.data_init(x, t)
    print(f'[{args.tag}] heat-equation sanity check, lam_ic={args.lam_ic}, N_r={N_r}, '
          f'params={sum(p.numel() for p in model.parameters())}', flush=True)
    t0, n_ep, hist = time.time(), 20.0, []
    for rnd in range(args.rounds):
        if rnd > 0:
            x, t, xi = tnf.draw_points(model, None, N_r, T, box, False, n_x)
        for ep in range(int(round(n_ep))):
            perm = torch.randperm(N_r, device=device)
            for bi in range(0, N_r - batch + 1, batch):
                idx = perm[bi:bi + batch]
                r, _ = residual_heat(model, x[idx], t[idx])
                p0 = model.log_prob(xi, torch.zeros(xi.shape[0], 1, device=device)).exp()
                loss = (r ** 2).mean() + args.lam_ic * ((p0 - ic_pdf(xi)) ** 2).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step()
        ev = evaluate(model, (0.0, 0.5, 1.0))
        hist.append({str(k): v for k, v in ev.items()})
        print(f'  == end of round k={rnd + 1} ({int(round(n_ep))} epochs) [{time.time() - t0:.0f}s]   '
              + '   '.join(f't={k}: relL2={v[0]:.2e} relKL={v[1]:.2e}' for k, v in ev.items()), flush=True)
        n_ep *= 2.0
    with open(os.path.join(tnf.OUTDIR, f'{args.tag}.json'), 'w') as fh:
        json.dump(dict(args=vars(args), history=hist, paper_at_t1=dict(rel_l2_k5=2e-3, rel_kl_k5=5e-5)), fh, indent=1)
    print('paper (k=5, t=1): relL2 ~ 2e-3, relKL ~ 5e-5')


if __name__ == '__main__':
    main()
