"""
WANPM (time-dependent) on the 2-D one-peak -> two-peak problem, carrying over the
recipe that made the Sec. 4.2 1-D steady double peak work. Defaults:

  * richer test-function class : a bank that is 60% sin plane waves and 40% localized
                                 (Gaussian) wavelets, f = phi(w.x + kappa t + b)
  * temporal frequencies       : kappa initialised as N(0, 10^2) (--kappa-scale). At scale 1
                                 the phase kappa*t barely changes over T=1 and the bank
                                 cannot resolve dynamics on the ~0.1 time scale
  * sigma-annealing curriculum : sigma^2 : 1.5 -> target (warm-up, cosine, hold)
  * generator optimizer        : Adam, lr 1e-3. Muon (lr 0.2 and 0.05) was tried and was
                                 worse here; it remains available via --opt muon /
                                 adamw_split (archived runs are in legacy/)

Everything else is the manuscript's time-dependent method: pushforward
F(t,x0,r) = x0 + sqrt(t) * Ftilde(t,x0,r), weak form with terminal / initial /
interior-integral terms, t ~ U(0, T), biased squared-mean loss (no U-statistic),
SGD adversary.

    python wanpm_transient.py --case B          # saves results/B_adam.{json,npz,png}
    python wanpm_transient.py --selftest
"""
import sys, os, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(CODES, 'rebuttal_stage1'))
sys.path.insert(0, os.path.join(CODES, 'rebuttal_stage4', 'R1_5_asymmetric_muon'))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytorch_optimizer as po

from common_wanpm import TimeDepPushforward
from wavelet_testfuncs import ACTIVATIONS
import problem as P
import metrics as Mx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE = 'gaussian'          # 'gaussian': N(0, I_d);  'uniform': U(0,1)^d   (set from --base)


def sample_base(n, d):
    return torch.rand(n, d, device=device) if BASE == 'uniform' else torch.randn(n, d, device=device)


OUTDIR = os.path.join(HERE, 'results')
os.makedirs(OUTDIR, exist_ok=True)


# ---------------------------------------------------------------------------
# space-time test-function bank  f_k(t,x) = phi_k(w_k . x + kappa_k t + b_k)
# ---------------------------------------------------------------------------
class TimeDepBank(nn.Module):
    """First (1-frac_wave)*K functions are sin plane waves (manuscript init),
    the rest use a localized activation. Same (w, kappa, b) parameterisation and
    closed-form Lf as the manuscript; only phi changes."""

    def __init__(self, n_in, K, frac_wave=0.4, activation='gauss', init_scale=4.0,
                 centre_spread=1.5, kappa_scale=10.0, sin_w_scale=1.0, T=1.0, seed=None,
                 frac_saddle=0.0, saddle_a=(6.0, 10.0), saddle_kappa=1.5, saddle_centre=0.3,
                 frac_path=0.0, path_a=(4.0, 8.0), path_kappa=1.5, path_range=(0.4, 1.2)):
        super().__init__()
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        self.n_saddle = int(round(frac_saddle * K))
        self.n_path = int(round(frac_path * K))
        n_wave_only = int(round(frac_wave * (K - self.n_saddle - self.n_path)))
        self.n_wave = n_wave_only + self.n_saddle + self.n_path   # all localized columns (used by _phi)
        self.n_sin = K - self.n_wave
        self.activation = activation
        K_ = K
        w = torch.randn(n_in, K_, generator=gen) * sin_w_scale
        kappa = torch.randn(K_, generator=gen) * kappa_scale
        b = torch.rand(K_, generator=gen) * 2 * np.pi
        if self.n_wave:
            s = slice(self.n_sin, K_)
            mag = init_scale * (0.5 + torch.rand(n_in, self.n_wave, generator=gen))
            sgn = torch.where(torch.rand(n_in, self.n_wave, generator=gen) < 0.5, -1.0, 1.0)
            w[:, s] = mag * sgn
            c = (torch.rand(n_in, self.n_wave, generator=gen) * 2 - 1) * centre_spread
            tc = torch.rand(self.n_wave, generator=gen) * T
            b[s] = -((w[:, s] * c).sum(0) + kappa[s] * tc)
        if self.n_saddle:
            # narrow Gaussian ridges whose normal points along the split direction u=(1,1)/sqrt2,
            # centred within +-saddle_centre of the saddle, small time frequency kappa
            ns = self.n_saddle
            s = slice(K_ - self.n_path - ns, K_ - self.n_path)
            ang = np.pi / 4 + 0.2 * torch.randn(ns, generator=gen)
            amp = saddle_a[0] + (saddle_a[1] - saddle_a[0]) * torch.rand(ns, generator=gen)
            sgn = torch.where(torch.rand(ns, generator=gen) < 0.5, -1.0, 1.0)
            w[:, s] = sgn * amp * torch.stack([torch.cos(ang), torch.sin(ang)])
            kappa[s] = torch.randn(ns, generator=gen) * saddle_kappa
            off = (torch.rand(ns, generator=gen) * 2 - 1) * saddle_centre
            tc = torch.rand(ns, generator=gen) * T
            b[s] = -(amp * off + kappa[s] * tc)
        if self.n_path:
            # same construction, centred on the connecting path between saddle and wells:
            # |ridge offset along u| in path_range, either side, small kappa
            npth = self.n_path
            s = slice(K_ - npth, K_)
            ang = np.pi / 4 + 0.2 * torch.randn(npth, generator=gen)
            amp = path_a[0] + (path_a[1] - path_a[0]) * torch.rand(npth, generator=gen)
            sgn = torch.where(torch.rand(npth, generator=gen) < 0.5, -1.0, 1.0)
            w[:, s] = sgn * amp * torch.stack([torch.cos(ang), torch.sin(ang)])
            kappa[s] = torch.randn(npth, generator=gen) * path_kappa
            mag = path_range[0] + (path_range[1] - path_range[0]) * torch.rand(npth, generator=gen)
            side = torch.where(torch.rand(npth, generator=gen) < 0.5, -1.0, 1.0)
            tc = torch.rand(npth, generator=gen) * T
            b[s] = -(amp * mag * side + kappa[s] * tc)
        self.w = nn.Parameter(w)
        self.kappa = nn.Parameter(kappa)
        self.b = nn.Parameter(b)

    def phase(self, x, t):
        return x @ self.w + self.kappa * t + self.b

    def _phi(self, z):
        zs, zw = z[:, :self.n_sin], z[:, self.n_sin:]
        s, c = torch.sin(zs), torch.cos(zs)
        parts = [(s, c, -s)]
        if self.n_wave:
            parts.append(ACTIVATIONS[self.activation](zw))
        return [torch.cat([p[i] for p in parts], dim=1) for i in range(3)]

    def f(self, x, t):
        return self._phi(self.phase(x, t))[0]

    def dfdt(self, x, t):
        return self.kappa * self._phi(self.phase(x, t))[1]

    def Lf(self, x, t, drift_fn, sigma2_fn):
        _, d1, d2 = self._phi(self.phase(x, t))
        b_dot_w = drift_fn(t, x) @ self.w
        w_sq = (self.w ** 2).sum(0, keepdim=True)
        return -b_dot_w * d1 - 0.5 * sigma2_fn(t, x) * w_sq * d2


# ---------------------------------------------------------------------------
# sigma curriculum (same shape as the 1-D study: warm-up, cosine, hold)
# ---------------------------------------------------------------------------
def sigma2_at(epoch, n_epochs, s2_start, s2_end, warm=0.10, anneal=0.55):
    a, b = int(warm * n_epochs), int((warm + anneal) * n_epochs)
    if epoch < a:
        return s2_start
    if epoch < b:
        p = (epoch - a) / max(1, b - a)
        return s2_start + (s2_end - s2_start) * 0.5 * (1 - np.cos(np.pi * p))
    return s2_end


# ---------------------------------------------------------------------------
# generator optimizers
# ---------------------------------------------------------------------------
def make_gen_opt(kind, model, lr, lr_head=2e-3):
    named = list(model.named_parameters())
    head = max(int(n.split('.')[1]) for n, _ in named)
    hidden = [p for n, p in named if p.ndim >= 2 and int(n.split('.')[1]) != head]
    rest = [p for n, p in named if not (p.ndim >= 2 and int(n.split('.')[1]) != head)]
    if kind == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    if kind == 'muon':
        return po.Muon([dict(params=hidden, lr=lr, weight_decay=0.0, use_muon=True),
                        dict(params=rest, lr=lr_head, weight_decay=0.0, use_muon=False)],
                       weight_decay=0.0)
    if kind == 'adamw_split':     # attribution control: same LR split, no orthogonalization
        return torch.optim.AdamW([dict(params=hidden, lr=lr), dict(params=rest, lr=lr_head)],
                                 weight_decay=0.0)
    raise ValueError(kind)



def weak_loss(pf, bank, M, T, rho0, drift_fn, s2_fn, d_base):
    """Manuscript's time-dependent weak-form loss (l. 286-311), with t ~ U(0, T).

    R_k = E_T[f_k] - E_0[f_k] - T * mean_m[(df_k/dt - L f_k)(t_m, x_m)],  loss = mean_k R_k^2.
    The interior term is df/dt - Lf with L the manuscript's (negated-generator) operator:
    that combination is the true identity for the exact solution, see the sign note in
    common_wanpm.time_dep_loss.
    """
    t = T * torch.rand(M, 1, device=device)
    x = pf(t, rho0(M), sample_base(M, d_base))
    E_int = T * (bank.dfdt(x, t) - bank.Lf(x, t, drift_fn, s2_fn)).mean(0)
    E_0 = bank.f(rho0(M), torch.zeros(M, 1, device=device)).mean(0)
    tT = T * torch.ones(M, 1, device=device)
    E_T = bank.f(pf(tT, rho0(M), sample_base(M, d_base)), tT).mean(0)
    R = E_T - E_0 - E_int
    return (R ** 2).mean(), x


@torch.no_grad()
def sample_at(pf, case, t, n, d_base):
    x0 = P.rho0_sampler(case, device)(n)
    tt = t * torch.ones(n, 1, device=device)
    return pf(tt, x0, sample_base(n, d_base)).cpu().numpy()


def evaluate(pf, case, d_base, refs, n=200_000):
    times, a, _ = refs
    out = {}
    for i, t in enumerate(times):
        out[float(t)] = Mx.compare(sample_at(pf, case, float(t), n, d_base), a[i])
    return out


# ---------------------------------------------------------------------------
def selftest():
    """(1) Lf/dfdt vs autograd for both activations; (2) pure-diffusion exact solution."""
    torch.manual_seed(0)
    dev = device
    for act in ('gauss', 'mexhat', 'morlet'):
        bank = TimeDepBank(2, 12, frac_wave=0.5, activation=act, seed=1).to(dev).double()
        x = torch.randn(40, 2, device=dev, dtype=torch.float64, requires_grad=True)
        t = torch.rand(40, 1, device=dev, dtype=torch.float64, requires_grad=True)
        f = bank.f(x, t)
        drift = lambda tt, xx: -0.7 * xx + 0.3
        s2 = lambda tt, xx: 0.37 * torch.ones(xx.shape[0], 1, device=dev, dtype=torch.float64)
        err = 0.0
        for k in range(f.shape[1]):
            gx, gt = torch.autograd.grad(f[:, k].sum(), [x, t], create_graph=True)
            lap = sum(torch.autograd.grad(gx[:, i].sum(), x, retain_graph=True)[0][:, i] for i in range(2))
            Lf_ad = -(drift(t, x) * gx).sum(1) - 0.5 * 0.37 * lap      # manuscript's negated generator
            err = max(err, (Lf_ad - bank.Lf(x, t, drift, s2)[:, k]).abs().max().item(),
                      (gt[:, 0] - bank.dfdt(x, t)[:, k]).abs().max().item())
        print(f'  {act:7s} max|analytic - autograd| (Lf, df/dt) = {err:.2e}')
        assert err < 1e-9
    # exact heat-equation solution: loss should be pure MC noise
    bank = TimeDepBank(2, 100, kappa_scale=1.0, seed=2).to(dev)
    s2c = 0.5
    drift0 = lambda tt, xx: torch.zeros_like(xx)
    s2f = lambda tt, xx: s2c * torch.ones(xx.shape[0], 1, device=dev)

    class Exact(nn.Module):
        n_dim, d_base = 2, 2
        def forward(self, t, x0, r):
            return x0 + torch.sqrt(s2c * t) * r
    rho0 = lambda n, device=dev: 0.5 * torch.randn(n, 2, device=device)
    l_ok, _ = weak_loss(Exact(), bank, 4000, 1.0, rho0, drift0, s2f, 2)

    class Wrong(Exact):
        def forward(self, t, x0, r):
            return x0 + torch.sqrt(2 * s2c * t) * r
    l_bad, _ = weak_loss(Wrong(), bank, 4000, 1.0, rho0, drift0, s2f, 2)
    print(f'  heat eq: loss(exact)={l_ok.item():.2e}  loss(wrong diffusivity)={l_bad.item():.2e}')
    assert l_bad.item() > 5 * l_ok.item()
    # information only: the same exact solution scored with the default kappa scale (10)
    bank10 = TimeDepBank(2, 100, seed=2).to(dev)
    l10, _ = weak_loss(Exact(), bank10, 4000, 1.0, rho0, drift0, s2f, 2)
    print(f'  loss of the EXACT solution, kappa-scale 1: {l_ok.item():.2e}, '
          f'default kappa-scale 10: {l10.item():.2e}  (pure Monte-Carlo noise floor)')
    print('selftest OK')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='B', choices=list(P.CASES))
    ap.add_argument('--tag', default=None)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--plot-only', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=30000)
    ap.add_argument('--M', type=int, default=2000)
    ap.add_argument('--K', type=int, default=300)
    ap.add_argument('--d-base', type=int, default=8)
    ap.add_argument('--base', default='gaussian', choices=['gaussian', 'uniform'], help='base distribution of r')
    ap.add_argument('--hidden', type=int, nargs='+', default=[128, 128])
    ap.add_argument('--power', type=float, default=0.5, help='time scaling t^power (0.5 = sqrt(t))')
    ap.add_argument('--opt', default='adam', choices=['adam', 'muon', 'adamw_split'])
    ap.add_argument('--lr-gen', type=float, default=None, help='Adam lr, or Muon/hidden lr')
    ap.add_argument('--lr-head', type=float, default=2e-3)
    ap.add_argument('--lr-test', type=float, default=1e-2)
    ap.add_argument('--adv-every', type=int, default=10)
    ap.add_argument('--adv-steps', type=int, default=5)
    ap.add_argument('--frac-wave', type=float, default=0.4, help='0 = plain sin bank (manuscript)')
    ap.add_argument('--wave-act', default='gauss', choices=['gauss', 'mexhat', 'morlet'])
    ap.add_argument('--init-scale', type=float, default=4.0)
    ap.add_argument('--frac-saddle', type=float, default=0.0, help='share of K that are low-kappa saddle wavelets')
    ap.add_argument('--saddle-a', type=float, nargs=2, default=[6.0, 10.0], help='|w| range (width = 1/|w|)')
    ap.add_argument('--saddle-kappa', type=float, default=1.5, help='init std of kappa for saddle wavelets')
    ap.add_argument('--saddle-centre', type=float, default=0.3, help='ridge offset from the saddle, +-')
    ap.add_argument('--frac-path', type=float, default=0.0, help='share of K that are low-kappa path wavelets')
    ap.add_argument('--path-a', type=float, nargs=2, default=[4.0, 8.0], help='|w| range (width = 1/|w|)')
    ap.add_argument('--path-kappa', type=float, default=1.5, help='init std of kappa for path wavelets')
    ap.add_argument('--path-range', type=float, nargs=2, default=[0.4, 1.2], help='|ridge offset along u| range')
    ap.add_argument('--centre-spread', type=float, default=1.5)
    ap.add_argument('--kappa-scale', type=float, default=10.0, help='init std of time frequencies kappa')
    ap.add_argument('--sin-w-scale', type=float, default=1.0)
    ap.add_argument('--anneal', type=int, default=1)
    ap.add_argument('--s2-start', type=float, default=1.5)
    ap.add_argument('--warm-frac', type=float, default=0.10, help='fraction of epochs held at s2-start')
    ap.add_argument('--anneal-frac', type=float, default=0.55, help='fraction of epochs for the cosine anneal; the rest is held at the target')
    ap.add_argument('--eval-every', type=int, default=5000)
    args = ap.parse_args()

    if args.selftest:
        selftest(); return
    global BASE
    BASE = args.base
    tag = args.tag or f'{args.case}_{args.opt}'
    npz_path = os.path.join(OUTDIR, f'{tag}.npz')
    if args.plot_only:
        plot(tag); return

    lr_gen = args.lr_gen if args.lr_gen is not None else (0.2 if args.opt != 'adam' else 1e-3)
    c = P.CASES[args.case]
    s2_end = c['sigma'] ** 2
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    pf = TimeDepPushforward(2, args.d_base, tuple(args.hidden), power=args.power).to(device)
    bank = TimeDepBank(2, args.K, args.frac_wave, args.wave_act, args.init_scale,
                       args.centre_spread, args.kappa_scale, args.sin_w_scale, T=P.T_FINAL, seed=args.seed,
                       frac_saddle=args.frac_saddle, saddle_a=tuple(args.saddle_a),
                       saddle_kappa=args.saddle_kappa, saddle_centre=args.saddle_centre,
                       frac_path=args.frac_path, path_a=tuple(args.path_a),
                       path_kappa=args.path_kappa, path_range=tuple(args.path_range)).to(device)
    gen_opt = make_gen_opt(args.opt, pf, lr_gen, args.lr_head)
    test_opt = torch.optim.SGD(bank.parameters(), lr=args.lr_test)
    rho0 = P.rho0_sampler(args.case, device)
    refs = Mx.load_reference(args.case)

    cur = [s2_end]
    drift_fn = lambda t, x: P.drift_torch(x)
    s2_fn = lambda t, x: cur[0] * torch.ones(x.shape[0], 1, device=x.device)

    def loss_fn():
        return weak_loss(pf, bank, args.M, P.T_FINAL, rho0, drift_fn, s2_fn, args.d_base)

    print(f'[{tag}] case {args.case} sigma={c["sigma"]} opt={args.opt} lr={lr_gen} '
          f'bank={args.K} (wave {args.frac_wave:.0%} {args.wave_act}) anneal={bool(args.anneal)} '
          f'M={args.M} epochs={args.epochs}', flush=True)
    losses = np.empty(args.epochs); sig2 = np.empty(args.epochs)
    hist = []
    t0 = time.time()
    for ep in range(args.epochs):
        cur[0] = (sigma2_at(ep, args.epochs, args.s2_start, s2_end, args.warm_frac, args.anneal_frac)
                  if args.anneal else s2_end)
        sig2[ep] = cur[0]
        if ep % args.adv_every == 0 and ep > 0:
            for _ in range(args.adv_steps):
                l, _ = loss_fn()
                test_opt.zero_grad(); (-l).backward(); test_opt.step()
        l, _ = loss_fn()
        gen_opt.zero_grad(); l.backward(); gen_opt.step()
        losses[ep] = l.item()
        if ep % args.eval_every == 0 or ep == args.epochs - 1:
            cur_eval = {}
            with torch.no_grad():
                for tt in (0.2, 1.0):
                    i = int(np.argmin(np.abs(refs[0] - tt)))
                    xs = sample_at(pf, args.case, tt, 50_000, args.d_base)
                    cur_eval[tt] = (Mx.summary(xs)['pi_plus'], Mx.sliced_w2(xs, refs[1][i]))
            hist.append((ep, cur_eval))
            print(f'  ep {ep:6d} s2={cur[0]:.3f} loss={losses[ep]:.3e} | '
                  f't=0.2 pi+={cur_eval[0.2][0]:.3f} sw2={cur_eval[0.2][1]:.3f} | '
                  f't=1.0 pi+={cur_eval[1.0][0]:.3f} sw2={cur_eval[1.0][1]:.3f} '
                  f'[{time.time()-t0:.0f}s]', flush=True)
    dt = time.time() - t0

    res = evaluate(pf, args.case, args.d_base, refs)
    floor = Mx.noise_floor(args.case)
    samples = {f'x_{t}': sample_at(pf, args.case, t, 200_000, args.d_base).astype(np.float32)
               for t in P.SNAP_TIMES}
    np.savez_compressed(npz_path, loss=losses[::20].astype(np.float32),
                        sigma2=sig2[::20].astype(np.float32), **samples)
    torch.save({'pf': pf.state_dict(), 'bank': bank.state_dict(), 'args': vars(args)},
               os.path.join(OUTDIR, f'{tag}.pt'))
    late = hist[-10:]
    late_mean = {str(t): float(np.mean([h[1][t][1] for h in late])) for t in (0.2, 1.0)}
    out = dict(args=vars(args), train_time_s=dt, metrics={str(k): v for k, v in res.items()},
               noise_floor={str(k): v for k, v in floor.items()},
               history=[dict(epoch=e, **{f'sw2_t{t}': v[1] for t, v in ev.items()},
                             **{f'pi_t{t}': v[0] for t, v in ev.items()}) for e, ev in hist],
               sw2_mean_last10_evals=late_mean)
    with open(os.path.join(OUTDIR, f'{tag}.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\n[{tag}] done in {dt:.0f}s.  metrics vs reference (noise floor in brackets):')
    for t, m in res.items():
        f_ = floor[t]
        print(f'  t={t:<5} pi+={m["pi_plus"]:.3f} sw2={m["sw2"]:.4f}[{f_["sw2"]:.4f}] '
              f'hell2d={m["hell2d"]:.3f}[{f_["hell2d"]:.3f}] hell_u={m["hell_u"]:.3f}[{f_["hell_u"]:.3f}] '
              f'saddle={m["saddle_mass"]:.4f} well_std(u,v)=({m["well_std_u"]:.3f},{m["well_std_v"]:.3f})')
    plot(tag)


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
        for i, (lab, s) in enumerate((('MC reference', ref[j]), ('WANPM', d[f'x_{float(t)}']))):
            h = ax[i, j]
            h.hist2d(s[:, 0], s[:, 1], bins=[e, e], cmap='viridis', norm=matplotlib.colors.PowerNorm(0.5))
            h.set_aspect('equal'); h.set_title(f'{lab}  t={t:g}', fontsize=9)
            h.plot([1, -1], [1, -1], 'r+', ms=8)
        m = js['metrics'][str(float(t))]
        ax[1, j].set_xlabel(f'pi+={m["pi_plus"]:.3f}  sw2={m["sw2"]:.3f}\n'
                            f'saddle={m["saddle_mass"]:.3f}', fontsize=8)
    # u-marginals
    for j, t in enumerate(times):
        h = ax[2, j]
        b = np.linspace(-2.2, 2.2, 150)
        h.hist(P.u_coord(ref[j]), b, density=True, alpha=0.5, label='MC')
        h.hist(P.u_coord(d[f'x_{float(t)}']), b, density=True, histtype='step', lw=1.6, label='WANPM')
        h.set_yscale('log'); h.set_ylim(1e-3, None); h.set_xlabel('u=(x1+x2)/sqrt2', fontsize=8)
        if j == 0:
            h.legend(fontsize=8)
    a = js['args']
    fig.suptitle(f'{tag}: case {case}, opt={a["opt"]}, bank wave={a["frac_wave"]:.0%} {a["wave_act"]}, '
                 f'anneal={bool(a["anneal"])}, M={a["M"]}, epochs={a["epochs"]}', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUTDIR, f'{tag}.png')
    plt.savefig(out, dpi=110); plt.close()
    print('wrote', out)


if __name__ == '__main__':
    main()
