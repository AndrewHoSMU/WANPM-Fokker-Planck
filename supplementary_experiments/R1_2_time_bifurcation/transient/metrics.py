"""
Sample-based metrics shared by every method (WANPM has no pointwise density, so
everything here works from samples only).

    pi_plus        fraction with u=(x1+x2)/sqrt2 > 0          (well population)
    sw2            sliced Wasserstein-2 vs reference samples   (bin-free)
    hell2d         Hellinger distance of 2-D histograms        (bin 0.05 on [-2.5,2.5]^2)
    hell_u         Hellinger distance of the u-marginal        (bin 0.02)
    saddle_mass    P(|x| < 0.3)                                (barrier over-population)
    well_std_u/v   std of u, v inside the u>0 well             (peak sharpness)
    EV             E[V(X)]
"""
import os
import numpy as np
import problem as P

HERE = os.path.dirname(os.path.abspath(__file__))


def load_reference(case, kind='mc'):
    """kind='mc': Euler-Maruyama samples (reference_mc.py); 'fv': samples of the finite-volume
    solution (reference_fv.py). Either way two independent sample sets a, b."""
    name = f'ref_{case}.npz' if kind == 'mc' else f'ref_{case}_fv.npz'
    d = np.load(os.path.join(HERE, 'reference', name))
    return np.round(d['times'].astype(np.float64), 6), d['a'], d['b']


def sliced_w2(x, y, n_proj=256, n=100_000, seed=0):
    rng = np.random.default_rng(seed)
    m = min(n, len(x), len(y))
    x = x[rng.permutation(len(x))[:m]]
    y = y[rng.permutation(len(y))[:m]]
    th = rng.uniform(0, 2 * np.pi, n_proj)
    dirs = np.stack([np.cos(th), np.sin(th)])            # (2, n_proj)
    px, py = np.sort(x @ dirs, axis=0), np.sort(y @ dirs, axis=0)
    return float(np.sqrt(np.mean((px - py) ** 2)))


def _hell(p, q):
    return float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(p * q)))))


def hellinger_2d(x, y, lo=-2.5, hi=2.5, nb=100):
    e = np.linspace(lo, hi, nb + 1)
    p, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=[e, e])
    q, _, _ = np.histogram2d(y[:, 0], y[:, 1], bins=[e, e])
    return _hell(p / max(p.sum(), 1), q / max(q.sum(), 1))


def hellinger_u(x, y, lo=-2.5, hi=2.5, nb=250):
    e = np.linspace(lo, hi, nb + 1)
    p, _ = np.histogram(P.u_coord(x), bins=e)
    q, _ = np.histogram(P.u_coord(y), bins=e)
    return _hell(p / max(p.sum(), 1), q / max(q.sum(), 1))


def summary(x):
    u, v = P.u_coord(x), P.v_coord(x)
    right = u > 0
    return {
        'pi_plus': float(right.mean()),
        'saddle_mass': float((np.linalg.norm(x, axis=1) < 0.3).mean()),
        'well_std_u': float(u[right].std()) if right.sum() > 10 else float('nan'),
        'well_std_v': float(v[right].std()) if right.sum() > 10 else float('nan'),
        'EV': float(P.V_np(x[:, 0], x[:, 1]).mean()),
        'frac_in_box': float((np.abs(x) < 2.5).all(axis=1).mean()),
    }


def compare(x, ref):
    """Metrics of learned samples x against reference samples ref."""
    m = summary(x)
    m['sw2'] = sliced_w2(x, ref)
    m['hell2d'] = hellinger_2d(x, ref)
    m['hell_u'] = hellinger_u(x, ref)
    return m


def noise_floor(case, kind='mc'):
    """Reference-half vs reference-half, same metrics: the finite-sample floor."""
    times, a, b = load_reference(case, kind)
    return {float(t): compare(a[i], b[i]) for i, t in enumerate(times)}
