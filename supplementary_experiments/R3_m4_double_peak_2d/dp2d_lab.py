"""
Sec. 4.3 two-dimensional double peak (Reviewer 3, minor comment 4: stray
samples near the origin in the left panel of Fig. 5).

    V(x) = |x - c|^2 |x + c|^2,  c = (1, 1),   b = -grad V,   a = sigma^2 I,
    sigma = 0.4,   rho ~ exp(-2 V / sigma^2).

Relative barrier 2 V(0) / sigma^2 = 2*4/0.16 = 50, so the saddle at the origin
carries no visible mass in the true density; the samples the manuscript's run
leaves there are a training artefact.

Same framework as the manuscript (squared-mean weak-form loss, (16,32,32,2)
tanh generator, K = 200, M = 1000, SGD adversary 1e-2 with 5 ascent steps every
10 epochs).  Changed, as in the other re-run examples:
  1. sigma-annealing: a = sigma^2 held at 1.5, cosine down to 0.16, held.
  2. mixed test-function bank: plane waves + localized ridge wavelets
     (spec string as in ../R2_3_ring_radial/ring_lab.py).
  3. generator optimizer: Muon split (hidden matrices lr 1e-1, rest 1e-2).

    python dp2d_lab.py --tag mixed_s0 --seed 0
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'R2_3_ring_radial'))
from ring_lab import RidgeGroup, RBFGroup, mlp, a_schedule, build_generator_opt

OUTDIR = os.path.join(HERE, 'results')
os.makedirs(OUTDIR, exist_ok=True)

A_TRUE = 0.16
MIXED = 'sin:120:1,gauss:20:4,mexhat:20:4,sech2:20:4,herm1:20:4'


def drift(x):
    x1, x2 = x[:, 0:1], x[:, 1:2]
    p = (x1 + 1) ** 2 + (x2 + 1) ** 2
    m = (x1 - 1) ** 2 + (x2 - 1) ** 2
    return torch.cat([-2 * (x1 - 1) * p - 2 * (x1 + 1) * m,
                      -2 * (x2 - 1) * p - 2 * (x2 + 1) * m], dim=1)


def V(X, Y):
    return ((X - 1) ** 2 + (Y - 1) ** 2) * ((X + 1) ** 2 + (Y + 1) ** 2)


class Bank(nn.Module):
    def __init__(self, spec, spread, seed):
        super().__init__()
        gen = torch.Generator().manual_seed(10_000 + seed)
        groups = []
        for item in spec.split(','):
            kind, K, scale = item.split(':')
            groups.append(RBFGroup(int(K), float(scale), spread, gen) if kind == 'rbf'
                          else RidgeGroup(kind, int(K), float(scale), spread, gen))
        self.groups = nn.ModuleList(groups)
        self.K = sum(g.K for g in groups)

    def Lf(self, x, a):
        bx = drift(x)
        return torch.cat([g.Lf(x, bx, a) for g in self.groups], 1)


# ---------------------------------------------------------------------------
def metrics(x, a=A_TRUE):
    # 2-D Hellinger on the grid of R3_M5_optimizer_comparison/lowdim_muon.metrics_2d
    lim, nb = 2.5, 120
    ed = np.linspace(-lim, lim, nb + 1); c = 0.5 * (ed[:-1] + ed[1:])
    H, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=[ed, ed]); ph = H / max(1, H.sum())
    X, Y = np.meshgrid(c, c, indexing='ij')
    pt = np.exp(-2 * V(X, Y) / a); pt /= pt.sum()
    out = dict(hellinger=float(np.sqrt(max(0., 1 - np.sum(np.sqrt(ph * pt))))),
               tv=float(0.5 * np.abs(ph - pt).sum()))
    up = (x[:, 0] + x[:, 1]) > 0
    out['frac_upper'] = float(up.mean())
    # mass away from both wells: the quantity Reviewer 3 asks about
    d = np.minimum(np.hypot(x[:, 0] - 1, x[:, 1] - 1), np.hypot(x[:, 0] + 1, x[:, 1] + 1))
    out['stray_frac'] = float((d > 0.35).mean())
    out['saddle_frac'] = float((np.hypot(x[:, 0], x[:, 1]) < 0.5).mean())
    # the same two numbers for the analytical density (fine grid)
    g = np.linspace(-2.5, 2.5, 2001); Xf, Yf = np.meshgrid(g, g, indexing='ij')
    pf = np.exp(-2 * V(Xf, Yf) / a); pf /= pf.sum()
    df = np.minimum(np.hypot(Xf - 1, Yf - 1), np.hypot(Xf + 1, Yf + 1))
    out['stray_frac_true'] = float(pf[df > 0.35].sum())
    out['saddle_frac_true'] = float(pf[np.hypot(Xf, Yf) < 0.5].sum())
    # shape of one well: std along / across the diagonal, upper well
    s = (Xf + Yf) / np.sqrt(2); t = (Xf - Yf) / np.sqrt(2); mk = (Xf + Yf) > 0
    wt = pf[mk] / pf[mk].sum()
    sm = (s[mk] * wt).sum()
    out['well_std_along_true'] = float(np.sqrt((((s[mk] - sm) ** 2) * wt).sum()))
    out['well_std_across_true'] = float(np.sqrt(((t[mk] ** 2) * wt).sum()))
    out['well_centre_true'] = float(sm / np.sqrt(2))
    for name, sel, sg in (('upper', up, 1.0), ('lower', ~up, -1.0)):
        y = x[sel & (d <= 0.35)]
        if len(y) < 10:
            continue
        ss = (y[:, 0] + y[:, 1]) / np.sqrt(2); tt = (y[:, 0] - y[:, 1]) / np.sqrt(2)
        out[f'{name}_centre'] = [float(y[:, 0].mean()), float(y[:, 1].mean())]
        out[f'{name}_std_along'] = float(ss.std()); out[f'{name}_std_across'] = float(tt.std())
    return out


DEFAULTS = dict(bank=MIXED, spread=2.0, M=1000, epochs=20000, d_base=16,
                hidden=(32, 32), opt='muon', lr=0.1, head_lr=1e-2,
                anneal_from=1.5, a=A_TRUE, warm=0.10, ramp=0.55,
                adv_every=10, adv_steps=5, lr_test=1e-2, seed=0, n_eval=200000)


def train(cfg, device='cpu', verbose=True):
    c = dict(DEFAULTS); c.update(cfg); c['hidden'] = tuple(c['hidden'])
    torch.manual_seed(c['seed']); np.random.seed(c['seed'])
    pf = mlp(c['d_base'], c['hidden'], 2).to(device)
    tf = Bank(c['bank'], c['spread'], c['seed']).to(device)
    gen_opt = build_generator_opt(pf, c)
    test_opt = torch.optim.SGD(tf.parameters(), lr=c['lr_test'])

    def loss_fn(a):
        x = pf(torch.randn(c['M'], c['d_base'], device=device))
        return (tf.Lf(x, a).mean(0) ** 2).mean()

    eval_gen = torch.Generator(device=device).manual_seed(777)
    E = c['epochs']; losses = np.empty(E, dtype=np.float32); avals = np.empty(E, dtype=np.float32)
    t0 = time.time()
    for ep in range(E):
        a = a_schedule(ep, E, c['anneal_from'], c['a'], c['warm'], c['ramp']); avals[ep] = a
        if ep % c['adv_every'] == 0 and ep > 0:
            for _ in range(c['adv_steps']):
                l = loss_fn(a); test_opt.zero_grad(); (-l).backward(); test_opt.step()
        l = loss_fn(a); gen_opt.zero_grad(); l.backward(); gen_opt.step()
        losses[ep] = l.item()
        if not np.isfinite(losses[ep]):
            losses[ep:] = np.nan; break
        if verbose and ep % 2000 == 0:
            with torch.no_grad():       # own generator: leaves the training random stream untouched
                xe = pf(torch.randn(20000, c['d_base'], generator=eval_gen, device=device))
                de = torch.minimum((xe - 1).norm(dim=1), (xe + 1).norm(dim=1))
                stray = (de > 0.35).float().mean().item(); up = (xe.sum(1) > 0).float().mean().item()
            print(f"   ep {ep:6d} a={a:.3f} loss={losses[ep]:.4e} stray={stray:.4f} upper={up:.3f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
    dt = time.time() - t0
    with torch.no_grad():
        x = pf(torch.randn(c['n_eval'], c['d_base'], device=device)).cpu().numpy()
    rec = dict(cfg=c, train_time_s=dt, loss_first=float(losses[0]),
               loss_final=float(np.nanmean(losses[-500:])),
               diverged=bool(not np.isfinite(losses[-1]) or not np.isfinite(x).all()))
    if not rec['diverged']:
        rec.update(metrics(x, c['a']))
    return rec, x, losses, avals, pf


def fmt(r):
    if r.get('diverged'):
        return f"{r['tag']:24s} DIVERGED"
    return (f"{r['tag']:24s} H={r['hellinger']:.4f} TV={r['tv']:.4f} upper={r['frac_upper']:.4f} "
            f"stray={r['stray_frac']:.2e} (true {r['stray_frac_true']:.1e}) "
            f"saddle={r['saddle_frac']:.2e} loss={r['loss_final']:.2e} ({r['train_time_s']:.0f}s)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--bank', default=MIXED)
    ap.add_argument('--epochs', type=int, default=DEFAULTS['epochs'])
    ap.add_argument('--opt', default=DEFAULTS['opt'], choices=['muon', 'adam', 'adam_split'])
    ap.add_argument('--lr', type=float, default=None,
                    help='generator lr (default: 0.1 for the muon split, 1e-2 for adam)')
    ap.add_argument('--M', type=int, default=DEFAULTS['M'])
    ap.add_argument('--no-anneal', action='store_true', help='train at sigma^2 = 0.16 throughout')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    torch.set_num_threads(1)
    lr = args.lr if args.lr is not None else (1e-2 if args.opt == 'adam' else DEFAULTS['lr'])
    cfg = dict(seed=args.seed, bank=args.bank, epochs=args.epochs, opt=args.opt, lr=lr, M=args.M)
    if args.no_anneal:
        cfg['anneal_from'] = None
    rec, x, losses, avals, pf = train(cfg, args.device)
    rec['tag'] = args.tag
    np.savez_compressed(os.path.join(OUTDIR, f'{args.tag}.npz'),
                        x=x.astype(np.float32), losses=losses, a=avals)
    torch.save(pf.state_dict(), os.path.join(OUTDIR, f'{args.tag}.pt'))
    json.dump(rec, open(os.path.join(OUTDIR, f'{args.tag}.json'), 'w'), indent=1)
    print(fmt(rec), flush=True)
