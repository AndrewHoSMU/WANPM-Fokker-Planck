"""
Deterministic reference for the transient bifurcation problem: the Fokker-Planck equation

    d rho/dt = div(rho grad V) + D Laplacian(rho),      D = sigma^2 / 2,

solved on [-L, L]^2 with Scharfetter-Gummel finite volumes (exactly mass-conserving, positivity-
preserving under the step bound asserted below, exp(-V/D) is its discrete steady state) and Heun
in time. Saves the density at the snapshot times, the free energy F(t) = int V rho + D int rho log rho,
and -- so that the sample-based metrics in metrics.py work unchanged -- two independent sets of
samples drawn from the computed density (cell picked by its mass, uniform position inside it).

    python reference_fv.py --case E          # reference/ref_E_fv.npz
"""
import os, argparse, time
import numpy as np
import torch
import problem as P

HERE = os.path.dirname(os.path.abspath(__file__))
REFDIR = os.path.join(HERE, 'reference')
os.makedirs(REFDIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _bern(z):
    """Bernoulli function z / (e^z - 1)."""
    small = z.abs() < 1e-6
    return torch.where(small, 1 - z / 2, z / torch.expm1(torch.where(small, torch.ones_like(z), z)))


def solve(case, L=3.0, n=300, dt=1e-5, n_out=200, snap_times=()):
    """Returns t, F(t) on n_out+1 output times, F_inf, the cell centres x and rho at snap_times."""
    c = P.CASES[case]
    D = 0.5 * c['sigma'] ** 2
    h = 2 * L / n
    xc = -L + h * (torch.arange(n, device=device, dtype=torch.float64) + 0.5)
    X1, X2 = torch.meshgrid(xc, xc, indexing='ij')
    V = P.V_torch(torch.stack([X1.reshape(-1), X2.reshape(-1)], 1)).reshape(n, n)
    # flux through the face between cells i, i+1:  (D/h) [B(-z) rho_{i+1} - B(z) rho_i],  z = dV/D
    z1, z2 = (V[1:, :] - V[:-1, :]) / D, (V[:, 1:] - V[:, :-1]) / D
    up1, dn1, up2, dn2 = _bern(-z1), _bern(z1), _bern(-z2), _bern(z2)

    def rhs(rho):
        f1 = (D / h ** 2) * (up1 * rho[1:, :] - dn1 * rho[:-1, :])
        f2 = (D / h ** 2) * (up2 * rho[:, 1:] - dn2 * rho[:, :-1])
        out = torch.zeros_like(rho)
        out[:-1, :] += f1; out[1:, :] -= f1
        out[:, :-1] += f2; out[:, 1:] -= f2
        return out

    # forward-Euler positivity bound (Heun is a convex combination of Euler steps)
    outflow = torch.zeros_like(V)
    outflow[:-1, :] += dn1; outflow[1:, :] += up1
    outflow[:, :-1] += dn2; outflow[:, 1:] += up2
    dt_max = h ** 2 / (D * outflow.max().item())
    assert dt < dt_max, f'dt={dt} exceeds the positivity bound {dt_max:.2e}'

    def free_energy(rho):
        return (h ** 2 * (V * rho + D * torch.xlogy(rho, rho)).sum()).item()

    mu = torch.tensor(c['mu0'], device=device, dtype=torch.float64)
    rho = torch.exp(-((X1 - mu[0]) ** 2 + (X2 - mu[1]) ** 2) / (2 * c['s0'] ** 2))
    rho /= rho.sum() * h ** 2
    n_steps = int(round(P.T_FINAL / dt))
    every = n_steps // n_out
    snap_steps = {int(round(t / dt)): i for i, t in enumerate(snap_times)}
    snaps = [None] * len(snap_times)
    ts, Fs = [0.0], [free_energy(rho)]
    for k in range(1, n_steps + 1):
        k1 = rhs(rho)
        rho = rho + 0.5 * dt * (k1 + rhs(rho + dt * k1))
        if k % every == 0:
            ts.append(k * dt); Fs.append(free_energy(rho))
        if k in snap_steps:
            snaps[snap_steps[k]] = rho.cpu().numpy()
    Z = h ** 2 * torch.exp(-V / D).sum().item()
    return dict(t=np.array(ts), F=np.array(Fs), F_inf=-D * np.log(Z), mass=rho.sum().item() * h ** 2,
                x=xc.cpu().numpy(), rho=np.array(snaps))


def sample_density(rho, x, n, seed):
    """n samples of the piecewise-constant density rho on the uniform grid with centres x."""
    g = torch.Generator(device=device).manual_seed(seed)
    h = x[1] - x[0]
    p = torch.as_tensor(rho, device=device).reshape(-1).float()
    idx = torch.multinomial(p, n, replacement=True, generator=g)
    xc = torch.as_tensor(x, device=device, dtype=torch.float32)
    centre = torch.stack([xc[idx // len(x)], xc[idx % len(x)]], 1)
    return (centre + h * (torch.rand(n, 2, device=device, generator=g) - 0.5)).cpu().numpy()


def load_density(case):
    """times, cell centres x, rho[i_time, i_x1, i_x2]."""
    d = np.load(os.path.join(REFDIR, f'ref_{case}_fv.npz'))
    return np.round(d['times'].astype(np.float64), 6), d['x'], d['rho']


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='E', choices=list(P.CASES))
    ap.add_argument('--n', type=int, default=1_000_000, help='samples per independent half')
    ap.add_argument('--cells', type=int, default=300, help='cells per direction on [-3, 3]')
    args = ap.parse_args()
    t0 = time.time()
    r = solve(args.case, n=args.cells, snap_times=P.SNAP_TIMES)
    a = np.stack([sample_density(rho, r['x'], args.n, seed=10 + i) for i, rho in enumerate(r['rho'])])
    b = np.stack([sample_density(rho, r['x'], args.n, seed=20 + i) for i, rho in enumerate(r['rho'])])
    np.savez_compressed(os.path.join(REFDIR, f'ref_{args.case}_fv.npz'),
                        times=np.array(P.SNAP_TIMES, dtype=np.float32), a=a, b=b,
                        x=r['x'], rho=r['rho'].astype(np.float32), t_F=r['t'], F=r['F'], F_inf=r['F_inf'])
    print(f'case {args.case}: wrote finite-volume reference ({time.time()-t0:.0f}s), '
          f'mass={r["mass"]:.6f} F_inf={r["F_inf"]:.4f}')
    h = r['x'][1] - r['x'][0]
    X1, X2 = np.meshgrid(r['x'], r['x'], indexing='ij')
    for t, rho in zip(P.SNAP_TIMES, r['rho']):
        print(f'  t={t:<5} P(u>0)={rho[X1 + X2 > 0].sum() * h * h:.4f}  E[V]={(P.V_np(X1, X2) * rho).sum() * h * h:.4f}'
              f'  saddle-mass(|x|<0.3)={rho[X1 ** 2 + X2 ** 2 < 0.09].sum() * h * h:.4f}')
