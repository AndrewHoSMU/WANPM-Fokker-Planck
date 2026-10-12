"""
Sec. 4.8: time-dependent ring, with the SAME problem and training settings as
the steady ring of Sec. 4.6 (run P1_mex4_rbf2x10), plus the time dimension.

Problem   : V = (1/4)(r^2 - r0^2)^2,  b = -grad V + omega (-x2, x1),  generator
            (sigma^2/2) Laplacian,  r0 = 2, omega = 2, sigma = 1  (as Sec. 4.6);
            rho_0 = N((0, 1.2), 0.4^2 I),  T = 0.5                (as Sec. 4.8).
Generator : F(t, x0, r) = x0 + sqrt(t) Ftilde(t, x0, r),  Ftilde a tanh MLP with
            inputs (t, x0, r) in R^{1+2+4};  r ~ N(0, I_4).
Tests     : K = 200 = 120 plane waves            sin(w.x + kappa t + b)
                     +  40 Mexican-hat ridges     phi(w.x + kappa t + b)
                     +  40 Gaussian bumps         10 exp(-s^2|x-c|^2/2 - s_t^2 (t-tau)^2/2)
            i.e. the Sec. 4.6 bank with t treated as one more coordinate.
Loss      : eq:empirical_loss,  R = E_T[f] - E_0[f] - (T) E[ d_t f - L f ],  mean_k R_k^2.
Training  : Adam 1e-2 (generator), SGD 1e-2 (tests), 5 ascent steps every 10
            epochs, batch 1000 (interior, initial and terminal), 20,000 epochs.

    python ring_time.py --tag ringt_h64 --hidden 64
"""
import os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import ring_lab as RL

T_FINAL, EPS = 0.5, 1e-3
MU0, SIG0 = (0.0, 1.2), 0.4
KAPPA, A_DIFF = 1.0, 1.0
D_BASE = 4
TIMES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def sample_ic(n, device='cpu', gen=None):
    mu = torch.tensor(MU0, device=device)
    return mu + SIG0 * torch.randn(n, 2, device=device, generator=gen)


class Gen(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = RL.mlp(1 + 2 + D_BASE, hidden, 2)

    def forward(self, t, x0, r):
        return x0 + torch.sqrt(t.clamp(min=0.0)) * self.net(torch.cat([t, x0, r], 1))


class TimeBank(nn.Module):
    """Sec. 4.6 bank with time as an extra coordinate. Returns f, d_t f, L f."""

    def __init__(self, seed=0, n_sin=120, n_mex=40, n_bump=40, amp=10.0, spread=3.0):
        super().__init__()
        g = torch.Generator().manual_seed(10_000 + seed)
        P = nn.Parameter
        # plane waves (as common_wanpm.TimeDepTestFunctions)
        self.w_s = P(torch.randn(2, n_sin, generator=g))
        self.k_s = P(torch.randn(n_sin, generator=g))
        self.b_s = P(torch.rand(n_sin, generator=g) * 2 * np.pi)
        # Mexican-hat ridge functions
        mag = 4.0 * (0.5 + torch.rand(n_mex, generator=g))
        ang = torch.rand(n_mex, generator=g) * 2 * np.pi
        off = (torch.rand(n_mex, generator=g) * 2 - 1) * spread
        self.w_m = P(torch.stack([mag * torch.cos(ang), mag * torch.sin(ang)], 0))
        self.k_m = P(torch.randn(n_mex, generator=g))
        self.b_m = P(-mag * off)
        # space-time Gaussian bumps
        rad = spread * torch.sqrt(torch.rand(n_bump, generator=g))
        ang = torch.rand(n_bump, generator=g) * 2 * np.pi
        self.c = P(torch.stack([rad * torch.cos(ang), rad * torch.sin(ang)], 0))
        self.s = P((2.0 * (0.5 + torch.rand(n_bump, generator=g))).unsqueeze(0))
        self.tau = P(torch.rand(n_bump, generator=g) * T_FINAL)
        self.st = P(4.0 * (0.5 + torch.rand(n_bump, generator=g)))
        self.amp = amp
        self.K = n_sin + n_mex + n_bump

    def forward(self, x, t, need_derivs=True):
        bx = RL.ring_drift(x, KAPPA) if need_derivs else None
        out_f, out_t, out_L = [], [], []
        for w, k, b, phi in ((self.w_s, self.k_s, self.b_s, RL.PHI['sin']),
                             (self.w_m, self.k_m, self.b_m, RL.PHI['mexhat'])):
            z = x @ w + k * t + b
            f, d1, d2 = phi(z)
            out_f.append(f)
            if need_derivs:
                out_t.append(k * d1)
                out_L.append(-(bx @ w) * d1 - 0.5 * A_DIFF * (w ** 2).sum(0, keepdim=True) * d2)
        d0 = x[:, 0:1] - self.c[0:1]; d1_ = x[:, 1:2] - self.c[1:2]
        q = d0 ** 2 + d1_ ** 2; s2 = self.s ** 2; dt = t - self.tau
        f = self.amp * torch.exp(-0.5 * s2 * q - 0.5 * self.st ** 2 * dt ** 2)
        out_f.append(f)
        if not need_derivs:
            return torch.cat(out_f, 1)
        out_t.append(-self.st ** 2 * dt * f)
        out_L.append((s2 * (bx[:, 0:1] * d0 + bx[:, 1:2] * d1_) - 0.5 * A_DIFF * (s2 ** 2 * q - 2 * s2)) * f)
        return torch.cat(out_f, 1), torch.cat(out_t, 1), torch.cat(out_L, 1)


def loss_fn(gen, bank, M):
    t = EPS + (T_FINAL - EPS) * torch.rand(M, 1)
    x = gen(t, sample_ic(M), torch.randn(M, D_BASE))
    _, ft, Lf = bank(x, t)
    E_int = (T_FINAL - EPS) * (ft - Lf).mean(0)
    f0 = bank(sample_ic(M), torch.zeros(M, 1), need_derivs=False).mean(0)
    tT = torch.full((M, 1), T_FINAL)
    fT = bank(gen(tT, sample_ic(M), torch.randn(M, D_BASE)), tT, need_derivs=False).mean(0)
    return ((fT - f0 - E_int) ** 2).mean()


def simulate_mc(n, dt=1e-3, seed=0):
    """Direct Monte Carlo baseline: Euler-Maruyama for dX = b dt + sigma dW."""
    rng = np.random.default_rng(seed)
    x = np.array(MU0) + SIG0 * rng.standard_normal((n, 2))
    out = {0.0: x.copy()}
    steps = int(round(T_FINAL / dt))
    for i in range(1, steps + 1):
        r2 = (x ** 2).sum(1, keepdims=True)
        b = -KAPPA * (r2 - RL.R0 ** 2) * x + RL.OMEGA * np.stack([-x[:, 1], x[:, 0]], 1)
        x = x + b * dt + np.sqrt(A_DIFF * dt) * rng.standard_normal(x.shape)
        tt = round(i * dt, 6)
        if any(abs(tt - s) < 1e-9 for s in TIMES):
            out[tt] = x.copy()
    return out


def stats(x):
    r = np.hypot(x[:, 0], x[:, 1]); th = np.arctan2(x[:, 1], x[:, 0])
    C = np.exp(1j * th).mean()
    return dict(mean_x=float(x[:, 0].mean()), mean_y=float(x[:, 1].mean()),
                r_mean=float(r.mean()), r_std=float(r.std()),
                th_mean=float(np.angle(C)), th_std=float(np.sqrt(-2 * np.log(abs(C)))))


def sliced_w1(a, b, n_proj=64, seed=0):
    rng = np.random.default_rng(seed); q = np.linspace(0.005, 0.995, 199); tot = 0.0
    for _ in range(n_proj):
        ang = rng.uniform(0, np.pi); u = np.array([np.cos(ang), np.sin(ang)])
        tot += np.abs(np.quantile(a @ u, q) - np.quantile(b @ u, q)).mean()
    return float(tot / n_proj)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--hidden', type=int, nargs='+', default=[64])
    ap.add_argument('--epochs', type=int, default=20000)
    ap.add_argument('--M', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    gen = Gen(tuple(args.hidden)); bank = TimeBank(args.seed)
    g_opt = torch.optim.Adam(gen.parameters(), lr=1e-2)
    t_opt = torch.optim.SGD(bank.parameters(), lr=1e-2)
    losses = np.empty(args.epochs, dtype=np.float32); t0 = time.time()
    for ep in range(args.epochs):
        if ep % 10 == 0 and ep > 0:
            for _ in range(5):
                l = loss_fn(gen, bank, args.M); t_opt.zero_grad(); (-l).backward(); t_opt.step()
        l = loss_fn(gen, bank, args.M); g_opt.zero_grad(); l.backward(); g_opt.step()
        losses[ep] = l.item()
        if ep % 2000 == 0:
            print(f'  ep {ep:6d}  loss {losses[ep]:.4e}  ({time.time() - t0:.0f}s)', flush=True)
    train_time = time.time() - t0

    # evaluation against the direct Monte Carlo baseline
    N = 200000
    mc = simulate_mc(N)
    rec = dict(tag=args.tag, hidden=args.hidden, epochs=args.epochs, M=args.M, seed=args.seed,
               train_time_s=train_time, loss_first=float(losses[0]),
               loss_final=float(losses[-500:].mean()), times=TIMES, learned=[], mc=[], sw1=[])
    save = {}
    torch.manual_seed(999)
    for t in TIMES:
        with torch.no_grad():
            x = gen(torch.full((N, 1), t), sample_ic(N), torch.randn(N, D_BASE)).numpy()
        rec['learned'].append(stats(x)); rec['mc'].append(stats(mc[t])); rec['sw1'].append(sliced_w1(x, mc[t]))
        save[f'learned_{t:.1f}'] = x[:20000].astype(np.float32)
        save[f'mc_{t:.1f}'] = mc[t][:20000].astype(np.float32)
    np.savez_compressed(os.path.join(RL.OUTDIR, f'{args.tag}.npz'), losses=losses, **save)
    torch.save(gen.state_dict(), os.path.join(RL.OUTDIR, f'{args.tag}.pt'))
    json.dump(rec, open(os.path.join(RL.OUTDIR, f'{args.tag}.json'), 'w'), indent=1)
    print(f'{args.tag}: loss {rec["loss_first"]:.3f} -> {rec["loss_final"]:.4f}  ({train_time:.0f}s)')
    print('   t    | learned: <r> +- std   mean (x, y)      | direct MC: <r> +- std   mean (x, y)      | sliced W1')
    for t, a, b, s in zip(TIMES, rec['learned'], rec['mc'], rec['sw1']):
        print(f'  {t:.1f}   | {a["r_mean"]:.3f} +- {a["r_std"]:.3f}  ({a["mean_x"]:+.3f}, {a["mean_y"]:+.3f}) '
              f'| {b["r_mean"]:.3f} +- {b["r_std"]:.3f}  ({b["mean_x"]:+.3f}, {b["mean_y"]:+.3f}) | {s:.4f}')
