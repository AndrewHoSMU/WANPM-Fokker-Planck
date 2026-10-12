"""
Monte-Carlo reference for the transient bifurcation problem: Euler-Maruyama on
GPU, dt=1e-4 (rate*dt <= 1.6e-3 at the wells, so the weak error is negligible
next to sampling noise), N particles per snapshot. Two independent halves are
kept so metrics.py can report a reference-vs-reference noise floor.

    python reference_mc.py --case B
"""
import os, argparse, time
import numpy as np
import torch
import problem as P

HERE = os.path.dirname(os.path.abspath(__file__))
REFDIR = os.path.join(HERE, 'reference')
os.makedirs(REFDIR, exist_ok=True)


def simulate(case, n, dt=1e-4, seed=0, device='cuda'):
    c = P.CASES[case]
    g = torch.Generator(device=device).manual_seed(seed)
    mu = torch.tensor(c['mu0'], device=device)
    x = mu + c['s0'] * torch.randn(n, 2, device=device, generator=g)
    steps = {int(round(t / dt)): t for t in P.SNAP_TIMES}
    out = {}
    amp = c['sigma'] * np.sqrt(dt)
    for k in range(1, int(round(P.T_FINAL / dt)) + 1):
        x = x + P.drift_torch(x) * dt + amp * torch.randn(n, 2, device=device, generator=g)
        if k in steps:
            out[steps[k]] = x.cpu().numpy().astype(np.float32)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='B', choices=list(P.CASES))
    ap.add_argument('--n', type=int, default=1_000_000, help='particles per independent half')
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.time()
    a = simulate(args.case, args.n, seed=1, device=dev)
    b = simulate(args.case, args.n, seed=2, device=dev)
    ts = np.array(P.SNAP_TIMES, dtype=np.float32)
    np.savez_compressed(os.path.join(REFDIR, f'ref_{args.case}.npz'), times=ts,
                        a=np.stack([a[t] for t in P.SNAP_TIMES]),
                        b=np.stack([b[t] for t in P.SNAP_TIMES]))
    print(f'case {args.case}: wrote reference ({time.time()-t0:.0f}s)')
    for t in P.SNAP_TIMES:
        u = P.u_coord(a[t])
        print(f'  t={t:<5} P(u>0)={np.mean(u>0):.4f}  E[V]={P.V_np(a[t][:,0],a[t][:,1]).mean():.4f}'
              f'  saddle-mass(|x|<0.3)={np.mean(np.linalg.norm(a[t],axis=1)<0.3):.4f}')
