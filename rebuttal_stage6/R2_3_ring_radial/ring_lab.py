"""
Sec. 4.6 ring (r0=2, omega=2, sigma=1): fixing the over-concentrated radial
marginal (manuscript Fig. 12 left; Reviewer 2 comment 3 / Reviewer 3 minor 5).

Everything stays inside the manuscript's framework:
  * loss            : the manuscript's biased squared-mean weak-form loss
  * test functions  : ridge functions f(x) = phi(w.x + b), phi in C^2, with the
                      closed-form generator
                          Lf = -(b.w) phi'(z) - (1/2) a |w|^2 phi''(z)
                      (isotropic a = sigma^2; rho ~ exp(-2V/a)).
Only three things are varied:
  1. the test-function BANK  (which phi, how many of each, at what scale)
  2. a sigma-ANNEALING curriculum (start at a larger diffusion, cosine down)
  3. ordinary training knobs (batch M, generator lr schedule, optimizer, d)

Bank spec string:  "kind:count:scale[,kind:count:scale...]"
    sin:200:1            manuscript bank  (w ~ N(0, scale^2 I), b ~ U(0,2pi))
    sin:120:1,gauss:80:4 mixed plane-wave / Gaussian ridge-wavelet bank
  For localized kinds, scale = |w| (ridge width ~ 1/scale), direction uniform
  on the circle, ridge offset c ~ U(-spread, spread) (b = -|w| c).
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'results')
os.makedirs(OUTDIR, exist_ok=True)

R0, OMEGA, A_TRUE = 2.0, 2.0, 1.0


# ---------------------------------------------------------------------------
# problem
# ---------------------------------------------------------------------------
def ring_drift(x, kappa=2.0):
    """b = -kappa (r^2 - r0^2) x + omega (-x2, x1).
    kappa = 2 is the drift as coded and as printed in Sec. 4.6 (-2(r^2-r0^2)x);
    kappa = 1 is -grad V for the stated potential V = (1/4)(r^2-r0^2)^2."""
    x1, x2 = x[:, 0:1], x[:, 1:2]
    r2 = x1 ** 2 + x2 ** 2
    return torch.cat([-kappa * x1 * (r2 - R0 ** 2) - OMEGA * x2,
                      -kappa * x2 * (r2 - R0 ** 2) + OMEGA * x1], dim=1)


def sample_true(n, a=A_TRUE, rng=None):
    """Exact sampler for rho ~ exp(-(r^2-r0^2)^2/(2 var_u)), with var_u passed
    as `a` (= a_diffusion / kappa): u = r^2 ~ N(r0^2, var_u) truncated to u > 0,
    theta uniform.  (p(r) ~ r rho  <=>  p(u) ~ exp(-(u-r0^2)^2/(2 var_u)).)"""
    rng = rng or np.random.default_rng(0)
    u = np.empty(0)
    while u.size < n:
        c = R0 ** 2 + np.sqrt(a) * rng.standard_normal(2 * n)
        u = np.concatenate([u, c[c > 0]])
    r = np.sqrt(u[:n]); th = rng.uniform(-np.pi, np.pi, n)
    return np.stack([r * np.cos(th), r * np.sin(th)], 1)


def true_radial(rg, a=A_TRUE):
    w = rg * np.exp(-(rg ** 2 - R0 ** 2) ** 2 / (2 * a))
    return w / np.trapezoid(w, rg)


# ---------------------------------------------------------------------------
# activations: (phi, phi', phi'')
# ---------------------------------------------------------------------------
def _sin(z):
    s, c = torch.sin(z), torch.cos(z); return s, c, -s

def _gauss(z):
    g = torch.exp(-z ** 2 / 2); return g, -z * g, (z ** 2 - 1) * g

def _mexhat(z):
    g = torch.exp(-z ** 2 / 2)
    return (1 - z ** 2) * g, (z ** 3 - 3 * z) * g, (-z ** 4 + 6 * z ** 2 - 3) * g

def _morlet(z, w0=5.0):
    g = torch.exp(-z ** 2 / 2); c, s = torch.cos(w0 * z), torch.sin(w0 * z)
    return g * c, g * (-z * c - w0 * s), g * ((z ** 2 - 1 - w0 ** 2) * c + 2 * w0 * z * s)

def _sech2(z):
    t = torch.tanh(z); s2 = 1 - t ** 2
    return s2, -2 * t * s2, 4 * s2 - 6 * s2 ** 2

def _herm1(z):
    g = torch.exp(-z ** 2 / 2)
    return z * g, (1 - z ** 2) * g, (z ** 3 - 3 * z) * g

PHI = {'sin': _sin, 'gauss': _gauss, 'mexhat': _mexhat, 'morlet': _morlet,
       'sech2': _sech2, 'herm1': _herm1}


class RidgeGroup(nn.Module):
    def __init__(self, kind, K, scale, spread, gen):
        super().__init__()
        self.kind, self.K = kind, K
        if kind == 'sin':
            w = torch.randn(2, K, generator=gen) * scale
            b = torch.rand(K, generator=gen) * 2 * np.pi
        else:
            mag = scale * (0.5 + torch.rand(K, generator=gen))     # |w| in [0.5,1.5]*scale
            ang = torch.rand(K, generator=gen) * 2 * np.pi
            w = torch.stack([mag * torch.cos(ang), mag * torch.sin(ang)], 0)
            c = (torch.rand(K, generator=gen) * 2 - 1) * spread     # ridge offset
            b = -mag * c
        self.w = nn.Parameter(w); self.b = nn.Parameter(b)

    def Lf(self, x, bx, a):
        z = x @ self.w + self.b
        _, d1, d2 = PHI[self.kind](z)
        return -(bx @ self.w) * d1 - 0.5 * a * (self.w ** 2).sum(0, keepdim=True) * d2


class RBFGroup(nn.Module):
    """Gaussian wavelets localized in EVERY direction (a ridge function is
    localized only along w):

        f(x) = exp(-s^2 |x - c|^2 / 2),   centre c in R^n, inverse width s > 0,
        grad f = -s^2 (x - c) f,          Lap f = (s^4 |x - c|^2 - n s^2) f,
        Lf = -b . grad f - (a/2) Lap f    (closed form, O(n) per sample).

    Equivalently a product of n one-dimensional Gaussian ridge functions.
    Centres start uniform in the disc |c| < spread; s starts in [0.5,1.5]*scale.
    Both are trained by the adversary, exactly like (w, b) of a ridge function."""

    def __init__(self, K, scale, spread, gen):
        super().__init__()
        self.kind, self.K = 'rbf', K
        rad = spread * torch.sqrt(torch.rand(K, generator=gen))
        ang = torch.rand(K, generator=gen) * 2 * np.pi
        self.c = nn.Parameter(torch.stack([rad * torch.cos(ang), rad * torch.sin(ang)], 0))
        self.w = nn.Parameter((scale * (0.5 + torch.rand(K, generator=gen))).unsqueeze(0))

    def Lf(self, x, bx, a):
        d0 = x[:, 0:1] - self.c[0:1]; d1 = x[:, 1:2] - self.c[1:2]      # (M, K)
        s2 = self.w ** 2                                                # (1, K)
        q = d0 ** 2 + d1 ** 2
        f = torch.exp(-0.5 * s2 * q)
        b_dot_d = bx[:, 0:1] * d0 + bx[:, 1:2] * d1
        return (s2 * b_dot_d - 0.5 * a * (s2 ** 2 * q - 2 * s2)) * f

    def f(self, x):
        d0 = x[:, 0:1] - self.c[0:1]; d1 = x[:, 1:2] - self.c[1:2]
        return torch.exp(-0.5 * self.w ** 2 * (d0 ** 2 + d1 ** 2))


class Bank(nn.Module):
    def __init__(self, spec, spread=3.0, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(10_000 + seed)
        groups = []
        self.amps = []
        for item in spec.split(','):
            # kind:count:scale[:amplitude]  -- amplitude A is a fixed constant
            # multiplying the test function, f -> A f (so Lf -> A Lf). It only
            # rebalances how much each family contributes to mean_k R_k^2.
            parts = item.split(':')
            kind, K, scale = parts[0], int(parts[1]), float(parts[2])
            self.amps.append(float(parts[3]) if len(parts) > 3 else 1.0)
            groups.append(RBFGroup(K, scale, spread, gen) if kind == 'rbf'
                          else RidgeGroup(kind, K, scale, spread, gen))
        self.groups = nn.ModuleList(groups)
        self.K = sum(g.K for g in groups)

    def Lf(self, x, a, kappa=2.0):
        bx = ring_drift(x, kappa)
        return torch.cat([A * g.Lf(x, bx, a) if A != 1.0 else g.Lf(x, bx, a)
                          for g, A in zip(self.groups, self.amps)], 1)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def metrics(x, a=A_TRUE):
    r = np.hypot(x[:, 0], x[:, 1]); th = np.arctan2(x[:, 1], x[:, 0])
    rg = np.linspace(0, 4.5, 4501)
    pr = true_radial(rg, a)
    rm = np.trapezoid(rg * pr, rg); rs = np.sqrt(np.trapezoid((rg - rm) ** 2 * pr, rg))
    out = dict(r_mean=float(r.mean()), r_std=float(r.std()),
               r_mean_true=float(rm), r_std_true=float(rs),
               r_std_ratio=float(r.std() / rs))
    # robust core width: inter-quartile range of r against the true IQR
    # (std is dominated by a handful of stray samples inside the ring)
    cdf = np.cumsum(pr) * (rg[1] - rg[0]); cdf /= cdf[-1]
    q_true = np.interp([0.25, 0.75], cdf, rg)
    q = np.quantile(r, [0.25, 0.75])
    out['r_iqr_ratio'] = float((q[1] - q[0]) / (q_true[1] - q_true[0]))
    # radial Hellinger + peak ratio (90 bins on [0,4.5], width 0.05)
    e = np.linspace(0, 4.5, 91); c = 0.5 * (e[:-1] + e[1:]); dr = e[1] - e[0]
    h, _ = np.histogram(r, bins=e); h = h / len(r)
    pt = np.interp(c, rg, pr) * dr; pt /= pt.sum()
    out['radial_hellinger'] = float(np.sqrt(max(0., 1 - np.sum(np.sqrt(h * pt)))))
    out['peak_ratio'] = float(h.max() / pt.max())
    out['frac_inside'] = float((r < 1.0).mean())       # true: ~1e-3
    out['frac_outside'] = float((r > 3.0).mean())
    hh, _ = np.histogram(th, bins=36, range=(-np.pi, np.pi)); hh = hh / hh.sum()
    out['angular_tv'] = float(0.5 * np.abs(hh - 1 / 36).sum())
    # 2-D Hellinger, same grid as rebuttal_stage3 lowdim_muon.metrics_2d
    lim, nb = 3.5, 120
    ed = np.linspace(-lim, lim, nb + 1); cc = 0.5 * (ed[:-1] + ed[1:])
    H, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=[ed, ed]); ph = H / max(1, H.sum())
    X, Y = np.meshgrid(cc, cc, indexing='ij')
    p2 = np.exp(-(X ** 2 + Y ** 2 - R0 ** 2) ** 2 / (2 * a)); p2 /= p2.sum()
    out['hellinger'] = float(np.sqrt(max(0., 1 - np.sum(np.sqrt(ph * p2)))))
    return out


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def mlp(d_in, hidden, d_out):
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.Tanh()]; d = h
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


def a_schedule(epoch, epochs, a0, a_end=A_TRUE, warm=0.10, ramp=0.55):
    """Diffusion coefficient: hold a0, cosine down to a_end, hold."""
    if a0 is None:
        return a_end
    w = int(warm * epochs); e = w + int(ramp * epochs)
    if epoch < w:
        return a0
    if epoch < e:
        p = 0.5 * (1 - np.cos(np.pi * (epoch - w) / max(1, e - w)))
        return a0 + (a_end - a0) * p
    return a_end


def lr_factor(epoch, epochs, decay_to, start=0.65):
    """Generator-lr multiplier: 1 until `start`, then cosine down to decay_to."""
    if decay_to is None or epoch < start * epochs:
        return 1.0
    p = (epoch - start * epochs) / max(1, (1 - start) * epochs)
    return decay_to + (1 - decay_to) * 0.5 * (1 + np.cos(np.pi * p))


DEFAULTS = dict(bank='sin:200:1', spread=3.0, M=1000, epochs=20000, d_base=4,
                hidden=(64,), opt='adam', lr=1e-2, head_lr=1e-2, anneal_from=None,
                lr_decay_to=None, adv_every=10, adv_steps=5, lr_test=1e-2,
                seed=0, n_eval=200000,
                # problem: b = -kappa (r^2-r0^2) x + rotation, generator (a/2) Laplacian,
                # stationary rho ~ exp(-(r^2-r0^2)^2 / (2 a/kappa))
                kappa=2.0, a=1.0, warm=0.10, ramp=0.55)


def build_generator_opt(pf, c):
    if c['opt'] == 'adam':
        return torch.optim.Adam(pf.parameters(), lr=c['lr'])
    if c['opt'] in ('muon', 'adam_split'):
        # two-group split: hidden 2-D weight matrices at `lr`, biases + output
        # layer at `head_lr`.  'muon' orthogonalises the hidden update;
        # 'adam_split' is the control with the same split and plain Adam.
        named = list(pf.named_parameters())
        head = max(int(n.split('.')[0]) for n, _ in named)
        is_hid = lambda n, p: p.ndim >= 2 and int(n.split('.')[0]) != head
        hid = [p for n, p in named if is_hid(n, p)]
        rest = [p for n, p in named if not is_hid(n, p)]
        if c['opt'] == 'adam_split':
            return torch.optim.Adam([dict(params=hid, lr=c['lr']),
                                     dict(params=rest, lr=c['head_lr'])])
        import pytorch_optimizer as po
        return po.Muon([dict(params=hid, lr=c['lr'], weight_decay=0.0, use_muon=True),
                        dict(params=rest, lr=c['head_lr'], weight_decay=0.0, use_muon=False)],
                       weight_decay=0.0)
    raise ValueError(c['opt'])


def train(cfg, device='cpu', verbose=False):
    c = dict(DEFAULTS); c.update(cfg)
    c['hidden'] = tuple(c['hidden'])
    torch.manual_seed(c['seed']); np.random.seed(c['seed'])
    pf = mlp(c['d_base'], c['hidden'], 2).to(device)
    tf = Bank(c['bank'], c['spread'], c['seed']).to(device)
    gen_opt = build_generator_opt(pf, c)
    base_lrs = [g['lr'] for g in gen_opt.param_groups]
    test_opt = torch.optim.SGD(tf.parameters(), lr=c['lr_test'])

    def loss_fn(a):
        x = pf(torch.randn(c['M'], c['d_base'], device=device))
        return (tf.Lf(x, a, c['kappa']).mean(0) ** 2).mean()

    E = c['epochs']; losses = np.empty(E, dtype=np.float32); t0 = time.time()
    for ep in range(E):
        a = a_schedule(ep, E, c['anneal_from'], c['a'], c['warm'], c['ramp'])
        f = lr_factor(ep, E, c['lr_decay_to'])
        for g, l0 in zip(gen_opt.param_groups, base_lrs):
            g['lr'] = l0 * f
        if ep % c['adv_every'] == 0 and ep > 0:
            for _ in range(c['adv_steps']):
                l = loss_fn(a); test_opt.zero_grad(); (-l).backward(); test_opt.step()
        l = loss_fn(a); gen_opt.zero_grad(); l.backward(); gen_opt.step()
        losses[ep] = l.item()
        if not np.isfinite(losses[ep]):
            losses[ep:] = np.nan; break
        if verbose and ep % 2000 == 0:
            print(f"   ep {ep:6d} a={a:.3f} loss={losses[ep]:.4e}", flush=True)
    dt = time.time() - t0
    diverged = bool(not np.isfinite(losses[-1]))
    with torch.no_grad():
        x = pf(torch.randn(c['n_eval'], c['d_base'], device=device)).cpu().numpy()
    wn = torch.cat([g.w.detach().norm(dim=0) for g in tf.groups]).cpu().numpy()
    rec = dict(cfg=c, train_time_s=dt, final_loss=float(np.nanmean(losses[-500:])),
               diverged=diverged or bool(not np.isfinite(x).all()),
               w_norm_mean=float(np.nanmean(wn)), w_norm_max=float(np.nanmax(wn)))
    if not rec['diverged']:
        rec.update(metrics(x, c['a'] / c['kappa']))
    return rec, x, losses, pf


def run_tagged(tag, cfg, device='cpu', save_samples=True):
    path = os.path.join(OUTDIR, f'{tag}.json')
    if os.path.exists(path):
        return json.load(open(path))
    rec, x, losses, pf = train(cfg, device)
    rec['tag'] = tag
    if save_samples:
        np.savez_compressed(os.path.join(OUTDIR, f'{tag}.npz'),
                            x=x[:100000].astype(np.float32), losses=losses)
        torch.save(pf.state_dict(), os.path.join(OUTDIR, f'{tag}.pt'))
    json.dump(rec, open(path, 'w'), indent=1)
    return rec


def fmt(rec):
    if rec.get('diverged'):
        return f"{rec['tag']:40s} DIVERGED"
    return (f"{rec['tag']:40s} std={rec['r_std_ratio']:.3f} iqr={rec['r_iqr_ratio']:.3f} peak={rec['peak_ratio']:.2f} "
            f"Hr={rec['radial_hellinger']:.3f} H2d={rec['hellinger']:.3f} "
            f"angTV={rec['angular_tv']:.3f} in={rec['frac_inside']:.4f} "
            f"rmean={rec['r_mean']:.3f} loss={rec['final_loss']:.2e} "
            f"|w|={rec['w_norm_mean']:.2f} ({rec['train_time_s']:.0f}s)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--cfg', default='{}', help='JSON dict of overrides')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    torch.set_num_threads(1)
    rec = run_tagged(args.tag, json.loads(args.cfg), args.device)
    print(fmt(rec), flush=True)
