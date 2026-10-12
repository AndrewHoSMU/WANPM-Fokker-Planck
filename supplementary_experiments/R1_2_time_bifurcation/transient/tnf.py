"""
Temporal normalizing flow (t-NF) baseline: Feng, Zeng, Zhou, "Solving time dependent
Fokker-Planck equations via temporal normalizing flow", CiCP 32 (2022) 401-423.

Implemented from the paper (no reference code was available):
  * flow z = f(x,t) = spline o [ affine-coupling_L o actnorm_L ] o ... o [ affine-coupling_1 o actnorm_1 ]
      - actnorm  (eq 3.8):  y = a*x + b, data-dependent init
      - time-dependent affine coupling (eq 3.10, 3.11):
            x2' = x2*(1 + beta*tanh(s(x1,t))) + exp(zeta)*tanh(q(x1,t)),  beta = 0.6
        with (s,q) = NN(x1, t) (two hidden tanh layers), halves alternate each layer
      - polynomial spline layer (eq 3.14-3.17), optional (the paper turns it on only for its
        hardest example); time-independent
  * density p(x,t) = N(z;0,I) |det dz/dx|, so it is normalized and positive by construction
  * loss (eq 3.19-3.23):  lam_r * mean r^2 + lam_ic * mean (p(x,0) - p0(x))^2, with the strong-form
        residual r = p_t + div(b p) - (sigma^2/2) Lap p     (b = -grad V)
    computed by automatic differentiation of log p (third-order graph for the parameter gradient)
  * adaptive sampling (Algorithm 1): round 0 uses uniform points in a box; after every round the
    residual and initial-condition points are re-drawn from the current p_theta(.,t) by the
    inverse flow; epochs per round grow by alpha.

The paper does not state lam_r / lam_ic ("weights to balance the two terms"), so they are
hyper-parameters here (--lam-ic).

    python tnf.py --case E --tag E_tnf
    python tnf.py --selftest
"""
import sys, os, json, time, argparse, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import problem as P
import metrics as Mx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTDIR = os.path.join(HERE, 'results')
os.makedirs(OUTDIR, exist_ok=True)
LOG2PI = math.log(2 * math.pi)


# ---------------------------------------------------------------------------
# flow layers.  Every layer: forward(x, t) -> (y, logdet[batch]);  inverse(y, t) -> x
# ---------------------------------------------------------------------------
class Actnorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.log_a = nn.Parameter(torch.zeros(d))
        self.b = nn.Parameter(torch.zeros(d))

    @torch.no_grad()
    def init_from(self, h):
        mu, sd = h.mean(0), h.std(0).clamp(min=1e-3)
        self.log_a.copy_(-torch.log(sd))
        self.b.copy_(-mu / sd)

    def forward(self, x, t):
        return x * torch.exp(self.log_a) + self.b, self.log_a.sum().expand(x.shape[0])

    def inverse(self, y, t):
        return (y - self.b) * torch.exp(-self.log_a)


class Coupling(nn.Module):
    def __init__(self, d, first, width, beta=0.6):
        super().__init__()
        idx = list(range(d))
        half = d // 2 if d > 1 else 1
        keep = idx[:half] if first else idx[half:]
        self.i1 = keep
        self.i2 = [i for i in idx if i not in keep]
        self.beta = beta
        n1, n2 = len(self.i1), len(self.i2)
        self.net = nn.Sequential(nn.Linear(n1 + 1, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(),
                                 nn.Linear(width, 2 * n2))
        with torch.no_grad():                       # start close to the identity map
            self.net[-1].weight.mul_(1e-2)
            self.net[-1].bias.zero_()
        self.zeta = nn.Parameter(torch.zeros(n2))

    def _sq(self, x1, t):
        s, q = self.net(torch.cat([x1, t], 1)).chunk(2, dim=1)
        return 1 + self.beta * torch.tanh(s), torch.exp(self.zeta) * torch.tanh(q)

    def forward(self, x, t):
        x1, x2 = x[:, self.i1], x[:, self.i2]
        scale, shift = self._sq(x1, t)
        y = x.clone()
        y[:, self.i2] = x2 * scale + shift
        return y, torch.log(scale).sum(1)

    def inverse(self, y, t):
        y1, y2 = y[:, self.i1], y[:, self.i2]
        scale, shift = self._sq(y1, t)
        x = y.clone()
        x[:, self.i2] = (y2 - shift) / scale
        return x


class Spline(nn.Module):
    """Component-wise monotone map with piecewise-linear density on [-c,c] (eq 3.14-3.17):
    z = 2c*G_hat((y+c)/(2c)) - c inside, slope gamma outside."""

    def __init__(self, d, nodes=50, c=4.0, gamma=1e-6):
        super().__init__()
        self.m, self.c, self.gamma, self.d = nodes, c, gamma, d
        self.ktil = nn.Parameter(torch.zeros(d, nodes - 1))

    def _tables(self):
        m, h, g = self.m, 1.0 / self.m, self.gamma
        w = torch.softmax(self.ktil, dim=1)                       # exp(k~_j)/sum exp(k~_i)
        kint = (1 - g * h) * w / h                                # eq 3.17, uniform partition
        ends = torch.full((self.d, 1), g, device=kint.device, dtype=kint.dtype)
        k = torch.cat([ends, kint, ends], dim=1)                  # (d, m+1)
        C = torch.cat([torch.zeros(self.d, 1, device=k.device, dtype=k.dtype),
                       torch.cumsum(0.5 * (k[:, :-1] + k[:, 1:]) * h, dim=1)], dim=1)
        return k, C, h

    def forward(self, y, t):
        k, C, h = self._tables()
        c, m = self.c, self.m
        s = (y + c) / (2 * c)
        j = torch.clamp(torch.floor(s * m), 0, m - 1).long()
        ar = torch.arange(self.d, device=y.device)[None, :]
        k0, k1, C0 = k[ar, j], k[ar, j + 1], C[ar, j]
        u = s - j * h
        Gh = (k1 - k0) / (2 * h) * u ** 2 + k0 * u + C0
        p = k0 + (k1 - k0) / h * u
        inside = (y > -c) & (y < c)
        z_in, ld_in = 2 * c * Gh - c, torch.log(p.clamp(min=1e-30))
        z_out = torch.where(y >= c, self.gamma * (y - c) + c, self.gamma * (y + c) - c)
        z = torch.where(inside, z_in, z_out)
        ld = torch.where(inside, ld_in, torch.full_like(ld_in, math.log(self.gamma)))
        return z, ld.sum(1)

    def inverse(self, z, t):
        k, C, h = self._tables()
        c, m = self.c, self.m
        sy = ((z + c) / (2 * c)).clamp(1e-9, 1 - 1e-9)
        cols = []
        for i in range(self.d):
            j = (torch.searchsorted(C[i].contiguous(), sy[:, i].contiguous(), right=True) - 1).clamp(0, m - 1)
            k0, k1, C0 = k[i][j], k[i][j + 1], C[i][j]
            a = (k1 - k0) / (2 * h)
            Dl = sy[:, i] - C0
            u = 2 * Dl / (k0 + torch.sqrt((k0 ** 2 + 4 * a * Dl).clamp(min=1e-30)))
            cols.append(2 * c * (j * h + u) - c)
        y_in = torch.stack(cols, 1)
        y_out = torch.where(z >= c, c + (z - c) / self.gamma, -c + (z + c) / self.gamma)
        return torch.where((z > -c) & (z < c), y_in, y_out)


class TNF(nn.Module):
    def __init__(self, d=2, L=8, width=128, beta=0.6, spline_nodes=0, spline_c=4.0):
        super().__init__()
        self.d = d
        layers = []
        for i in range(L):
            layers += [Actnorm(d), Coupling(d, first=(i % 2 == 0), width=width, beta=beta)]
        if spline_nodes:
            layers.append(Spline(d, spline_nodes, spline_c))
        self.layers = nn.ModuleList(layers)

    @torch.no_grad()
    def data_init(self, x, t):
        h = x
        for l in self.layers:
            if isinstance(l, Actnorm):
                l.init_from(h)
            h, _ = l(h, t)

    def forward(self, x, t):
        ld = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for l in self.layers:
            x, d_ = l(x, t)
            ld = ld + d_
        return x, ld

    def log_prob(self, x, t):
        z, ld = self(x, t)
        return -0.5 * (z ** 2).sum(1) - 0.5 * self.d * LOG2PI + ld

    @torch.no_grad()
    def sample(self, n, t):
        z = torch.randn(n, self.d, device=t.device)
        for l in reversed(self.layers):
            z = l.inverse(z, t)
        return z


# ---------------------------------------------------------------------------
# strong-form residual  r = p_t + div(b p) - (sigma^2/2) Lap p,   b = -grad V
# via  r = p * [ l_t + div b + b.grad l - (sigma^2/2)(Lap l + |grad l|^2) ],   l = log p
# ---------------------------------------------------------------------------
def residual(logp_fn, drift_fn, divb_fn, x, t, sigma2):
    x = x.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)
    ell = logp_fn(x, t)
    gx, gt = torch.autograd.grad(ell.sum(), [x, t], create_graph=True)
    lap = 0.0
    for i in range(x.shape[1]):
        lap = lap + torch.autograd.grad(gx[:, i].sum(), x, create_graph=True)[0][:, i]
    b = drift_fn(x)
    bracket = gt[:, 0] + divb_fn(x) + (b * gx).sum(1) - 0.5 * sigma2 * (lap + (gx ** 2).sum(1))
    p = torch.exp(ell)
    return p * bracket, p


def tnf_residual(model, x, t, sigma2):
    return residual(model.log_prob, P.drift_torch, lambda x_: -P.lapV_torch(x_), x, t, sigma2)


# ---------------------------------------------------------------------------
def draw_points(model, case, n, T, box, uniform, n_ic, chunk=200_000):
    """Residual points (x,t) and initial-condition points x."""
    if uniform:
        x = (torch.rand(n, 2, device=device) * 2 - 1) * box
        t = T * torch.rand(n, 1, device=device)
        xi = (torch.rand(n_ic, 2, device=device) * 2 - 1) * box
        return x, t, xi
    xs, ts = [], []
    got = 0
    while got < n:
        tt = T * torch.rand(chunk, 1, device=device)
        xx = model.sample(chunk, tt)
        ok = torch.isfinite(xx).all(1)
        xs.append(xx[ok]); ts.append(tt[ok]); got += int(ok.sum())
    x, t = torch.cat(xs)[:n], torch.cat(ts)[:n]
    xi, got = [], 0
    while got < n_ic:
        xx = model.sample(chunk, torch.zeros(chunk, 1, device=device))
        ok = torch.isfinite(xx).all(1)
        xi.append(xx[ok]); got += int(ok.sum())
    return x, t, torch.cat(xi)[:n_ic]


@torch.no_grad()
def sample_at(model, t, n):
    out = []
    for i in range(0, n, 200_000):
        m = min(200_000, n - i)
        out.append(model.sample(m, t * torch.ones(m, 1, device=device)))
    x = torch.cat(out)
    x = x[torch.isfinite(x).all(1)]
    return x.cpu().numpy()


def selftest():
    dev = device
    # 1. analytic Laplacian of V
    x = (torch.randn(50, 2, device=dev, dtype=torch.float64)).requires_grad_(True)
    V = P.V_torch(x.float() if False else x).sum()
    g = torch.autograd.grad(V, x, create_graph=True)[0]
    lap = sum(torch.autograd.grad(g[:, i].sum(), x, retain_graph=True)[0][:, i] for i in range(2))
    print('  |Lap V analytic - autograd| max = %.2e' % (lap - P.lapV_torch(x)).abs().max().item())
    assert (lap - P.lapV_torch(x)).abs().max().item() < 1e-9

    # 2. residual formula on an exact solution: OU  dX = -X dt + sigma dW, X0~N(m0, s0^2 I)
    sig2, m0, s0 = 0.7, 1.3, 0.5
    def m_t(t): return m0 * torch.exp(-t)
    def v_t(t): return s0 ** 2 * torch.exp(-2 * t) + 0.5 * sig2 * (1 - torch.exp(-2 * t))
    def logp_ou(x, t):
        v = v_t(t)[:, 0]
        return -((x - m_t(t)) ** 2).sum(1) / (2 * v) - torch.log(2 * math.pi * v)
    xs = torch.randn(200, 2, device=dev, dtype=torch.float64) * 0.8 + 0.5
    ts = torch.rand(200, 1, device=dev, dtype=torch.float64)
    r, p = residual(logp_ou, lambda x_: -x_, lambda x_: -2.0 * torch.ones(x_.shape[0], device=dev, dtype=x_.dtype),
                    xs, ts, sig2)
    print('  residual of the exact OU solution: max|r|=%.2e   (max p = %.2f)' % (r.abs().max().item(), p.max().item()))
    assert r.abs().max().item() < 1e-9
    r_bad, _ = residual(logp_ou, lambda x_: -x_, lambda x_: -2.0 * torch.ones(x_.shape[0], device=dev, dtype=x_.dtype),
                        xs, ts, 2 * sig2)
    assert r_bad.abs().max().item() > 1e-3

    # 3. flow: inverse(forward(x)) = x and log|det| vs the autograd Jacobian
    torch.manual_seed(0)
    for spl in (0, 20):
        m = TNF(2, L=4, width=32, spline_nodes=spl).to(dev).double()
        for l in m.layers:                                   # make the layers non-trivial
            for p_ in l.parameters():
                p_.data += 0.3 * torch.randn_like(p_)
        xt = torch.randn(64, 2, device=dev, dtype=torch.float64) * 1.2
        tt = torch.rand(64, 1, device=dev, dtype=torch.float64)
        z, ld = m(xt, tt)
        xr = z
        for l in reversed(m.layers):
            xr = l.inverse(xr, tt)
        inv_err = (xr - xt).abs().max().item()
        ldj = []
        for i in range(8):
            J = torch.autograd.functional.jacobian(lambda a: m(a[None, :], tt[i:i + 1])[0][0], xt[i])
            ldj.append(torch.linalg.slogdet(J)[1].item())
        ld_err = np.abs(np.array(ldj) - ld[:8].detach().cpu().numpy()).max()
        print(f'  flow (spline nodes={spl:2d}): max|inv(fwd(x))-x| = {inv_err:.2e},  max|logdet - slogdet(J)| = {ld_err:.2e}')
        assert inv_err < 1e-6 and ld_err < 1e-6
    print('selftest OK')


def plot(tag):
    d = np.load(os.path.join(OUTDIR, f'{tag}.npz'))
    with open(os.path.join(OUTDIR, f'{tag}.json')) as fh:
        js = json.load(fh)
    case = js['args']['case']
    times, ref, _ = Mx.load_reference(case)
    n = len(times)
    fig, ax = plt.subplots(3, n, figsize=(3.1 * n, 9.2))
    e = np.linspace(-2.2, 2.2, 221)
    for j, t in enumerate(times):
        for i, (lab, s) in enumerate((('MC reference', ref[j]), ('t-NF', d[f'x_{float(t)}']))):
            h = ax[i, j]
            h.hist2d(s[:, 0], s[:, 1], bins=[e, e], cmap='viridis', norm=matplotlib.colors.PowerNorm(0.5))
            h.set_aspect('equal'); h.set_title(f'{lab}  t={t:g}', fontsize=9); h.plot([1, -1], [1, -1], 'r+', ms=8)
        m = js['metrics'][str(float(t))]
        ax[1, j].set_xlabel(f'pi+={m["pi_plus"]:.3f}  sw2={m["sw2"]:.3f}\nsaddle={m["saddle_mass"]:.3f}', fontsize=8)
        h = ax[2, j]
        b = np.linspace(-2.2, 2.2, 150)
        h.hist(P.u_coord(ref[j]), b, density=True, alpha=0.5, label='MC')
        h.hist(P.u_coord(d[f'x_{float(t)}']), b, density=True, histtype='step', lw=1.6, label='t-NF')
        h.set_yscale('log'); h.set_ylim(1e-3, None); h.set_xlabel('u=(x1+x2)/sqrt2', fontsize=8)
        if j == 0:
            h.legend(fontsize=8)
    a = js['args']
    fig.suptitle(f'{tag}: t-NF, case {case}, L={a["L"]}, spline={a["spline"]}, lam_ic={a["lam_ic"]}, '
                 f'rounds={a["rounds"]}', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(OUTDIR, f'{tag}.png'), dpi=110); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='E', choices=list(P.CASES))
    ap.add_argument('--tag', default=None)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--plot-only', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--L', type=int, default=8, help='number of (actnorm + coupling) layers')
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--spline', type=int, default=0, help='spline nodes (0 = off)')
    ap.add_argument('--spline-c', type=float, default=4.0)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=4000)
    ap.add_argument('--n-x', type=int, default=4000, help='spatial points per time slice')
    ap.add_argument('--n-t', type=int, default=100, help='time slices; N_r = n_x * n_t')
    ap.add_argument('--n-ic', type=int, default=20000)
    ap.add_argument('--epochs', type=int, default=30, help='epochs in adaptive round 0')
    ap.add_argument('--alpha', type=float, default=1.5, help='epoch growth factor per round')
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--lam-r', type=float, default=1.0)
    ap.add_argument('--lam-ic', type=float, default=10.0)
    ap.add_argument('--box', type=float, default=3.0)
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    tag = args.tag or f'{args.case}_tnf'
    if args.plot_only:
        plot(tag); return

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    c = P.CASES[args.case]
    s2, T = c['sigma'] ** 2, P.T_FINAL
    model = TNF(2, args.L, args.width, spline_nodes=args.spline, spline_c=args.spline_c).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    refs = Mx.load_reference(args.case)
    N_r = args.n_x * args.n_t
    print(f'[{tag}] t-NF case {args.case} sigma^2={s2}  L={args.L} width={args.width} spline={args.spline} '
          f'N_r={N_r} batch={args.batch} lam_ic={args.lam_ic}  params={sum(p.numel() for p in model.parameters())}',
          flush=True)

    x, t, xi = draw_points(model, args.case, N_r, T, args.box, True, args.n_ic)
    model.data_init(x[:20000], t[:20000])
    hist, t0 = [], time.time()
    n_ep = float(args.epochs)
    for rnd in range(args.rounds):
        if rnd > 0:
            x, t, xi = draw_points(model, args.case, N_r, T, args.box, False, args.n_ic)
        for ep in range(int(round(n_ep))):
            perm = torch.randperm(N_r, device=device)
            pic = torch.randperm(args.n_ic, device=device)
            tot_r = tot_ic = 0.0
            nb = 0
            for bi in range(0, N_r - args.batch + 1, args.batch):
                idx = perm[bi:bi + args.batch]
                r, _ = tnf_residual(model, x[idx], t[idx], s2)
                ic_idx = pic[(nb * args.batch) % args.n_ic:][:args.batch] if args.n_ic >= args.batch else pic
                xi_b = xi[ic_idx]
                p0 = model.log_prob(xi_b, torch.zeros(xi_b.shape[0], 1, device=device)).exp()
                l_ic = ((p0 - P.rho0_pdf_torch(args.case, xi_b)) ** 2).mean()
                l_r = (r ** 2).mean()
                loss = args.lam_r * l_r + args.lam_ic * l_ic
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                opt.step()
                tot_r += l_r.item(); tot_ic += l_ic.item(); nb += 1
            if ep % 5 == 0 or ep == int(round(n_ep)) - 1:
                print(f'  round {rnd} ep {ep:3d}/{int(round(n_ep))}  L_r={tot_r/nb:.3e}  L_ic={tot_ic/nb:.3e}  '
                      f'[{time.time()-t0:.0f}s]', flush=True)
        ev = {}
        for tt in (0.2, 1.0):
            i = int(np.argmin(np.abs(refs[0] - tt)))
            xs = sample_at(model, tt, 50_000)
            ev[tt] = (Mx.summary(xs)['pi_plus'], Mx.sliced_w2(xs, refs[1][i]), len(xs) / 50_000)
        hist.append(dict(round=rnd, **{f'pi_t{t_}': v[0] for t_, v in ev.items()},
                         **{f'sw2_t{t_}': v[1] for t_, v in ev.items()}))
        print(f'  == end of round {rnd}: t=0.2 pi+={ev[0.2][0]:.3f} sw2={ev[0.2][1]:.3f} | '
              f't=1.0 pi+={ev[1.0][0]:.3f} sw2={ev[1.0][1]:.3f}  (finite samples {ev[1.0][2]:.3f})', flush=True)
        n_ep *= args.alpha
    dt = time.time() - t0

    res, samples = {}, {}
    times, a_, _ = refs
    for i, t_ in enumerate(times):
        xs = sample_at(model, float(t_), 200_000)
        samples[f'x_{float(t_)}'] = xs.astype(np.float32)
        m = Mx.compare(xs, a_[i])
        xr = torch.tensor(a_[i][:100_000], device=device)
        with torch.no_grad():
            m['nll_ref'] = float(-model.log_prob(xr, float(t_) * torch.ones(xr.shape[0], 1, device=device)).mean())
        res[float(t_)] = m
    floor = Mx.noise_floor(args.case)
    np.savez_compressed(os.path.join(OUTDIR, f'{tag}.npz'), **samples)
    torch.save({'model': model.state_dict(), 'args': vars(args)}, os.path.join(OUTDIR, f'{tag}.pt'))
    with open(os.path.join(OUTDIR, f'{tag}.json'), 'w') as fh:
        json.dump(dict(args=vars(args), train_time_s=dt, metrics={str(k): v for k, v in res.items()},
                       noise_floor={str(k): v for k, v in floor.items()}, history=hist), fh, indent=1)
    print(f'\n[{tag}] done in {dt:.0f}s.  metrics vs reference (noise floor in brackets):')
    for t_, m in res.items():
        f_ = floor[t_]
        print(f'  t={t_:<5} pi+={m["pi_plus"]:.3f} sw2={m["sw2"]:.4f}[{f_["sw2"]:.4f}] '
              f'hell_u={m["hell_u"]:.3f}[{f_["hell_u"]:.3f}] saddle={m["saddle_mass"]:.4f} '
              f'well_std(u,v)=({m["well_std_u"]:.3f},{m["well_std_v"]:.3f})  nll_ref={m["nll_ref"]:.3f}')
    plot(tag)


if __name__ == '__main__':
    main()
