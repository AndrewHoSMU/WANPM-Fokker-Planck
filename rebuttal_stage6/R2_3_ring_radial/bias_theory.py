"""
Why is the learned ring too narrow?  A training-free diagnostic.

The manuscript's loss is  L = mean_k (M^-1 sum_m L f_k(x_m))^2, whose
expectation over the batch is

    E[L] = mean_k [ (E_rho L f_k)^2  +  Var_rho(L f_k) / M ].
             ^ weak-form residual      ^ finite-batch bias

Take the EXACT ring samples and rescale only their radial spread,
r -> rbar + s (r - rbar).  s = 1 is the true solution.  For each test-function
bank we evaluate E[L](s) and report the s that minimises it: that is where the
generator is driven, optimiser aside.  If argmin s < 1, the objective itself
prefers an over-concentrated ring.

NOTE (what this script actually found).  The exact samples here are drawn from
the density PRINTED in the manuscript, exp(-(r^2-r0^2)^2/2), while Bank.Lf
defaults to the drift as CODED (kappa = 2).  The residual is minimised at
s ~ 0.71 = 1/sqrt(2) for every bank and every M, i.e. not a batch-bias effect at
all: the printed density is simply not the stationary solution of the coded
drift.  See verify_reference.py and RESULTS.md Sec. 1.
"""
import os, json
import numpy as np, torch
import ring_lab as RL

torch.manual_seed(0)
N = 200000
x0 = RL.sample_true(N)
r = np.hypot(x0[:, 0], x0[:, 1]); th = np.arctan2(x0[:, 1], x0[:, 0]); rbar = r.mean()
S = np.array([0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3])

BANKS = {
    'sin s=1 (manuscript)': 'sin:600:1',
    'sin s=2': 'sin:600:2',
    'sin s=3': 'sin:600:3',
    'sin s=4': 'sin:600:4',
    'gauss s=2': 'gauss:600:2',
    'gauss s=4': 'gauss:600:4',
    'gauss s=8': 'gauss:600:8',
    'mexhat s=4': 'mexhat:600:4',
    'herm1 s=4': 'herm1:600:4',
    'sech2 s=4': 'sech2:600:4',
    'mix sin1:60% gauss4:40%': 'sin:360:1,gauss:240:4',
    'mix sin1:75% gauss4:25%': 'sin:450:1,gauss:150:4',
    'mix sin1:50% sin3:50%': 'sin:300:1,sin:300:3',
}


def curves(spec):
    bank = RL.Bank(spec, spread=3.0, seed=0).double()
    res2, var = [], []
    for s in S:
        rs = rbar + s * (r - rbar)
        x = torch.tensor(np.stack([rs * np.cos(th), rs * np.sin(th)], 1))
        with torch.no_grad():
            B = 20000
            sums = torch.zeros(bank.K, dtype=torch.float64); sq = torch.zeros_like(sums)
            for i in range(0, N, B):
                L = bank.Lf(x[i:i + B], RL.A_TRUE)
                sums += L.sum(0); sq += (L ** 2).sum(0)
            mean = sums / N; v = sq / N - mean ** 2
        # debias the squared mean estimated from N samples
        res2.append(float((mean ** 2 - v / N).mean())); var.append(float(v.mean()))
    return np.array(res2), np.array(var)


if __name__ == '__main__':
    print(f"true radial std = {r.std():.4f}; s rescales it. M = batch size.\n")
    print(f"{'bank':26s} {'argmin s':>9s} {'argmin s':>9s} {'argmin s':>9s}")
    print(f"{'':26s} {'M=1000':>9s} {'M=4000':>9s} {'M=16000':>9s}")
    out = {}
    for name, spec in BANKS.items():
        res2, var = curves(spec)
        sf = np.linspace(S[0], S[-1], 901)
        row = []
        for M in (1000, 4000, 16000):
            tot = np.interp(sf, S, res2 + var / M)
            row.append(float(sf[tot.argmin()]))
        i8, i1 = list(S).index(0.8), list(S).index(1.0)
        # batch size at which the true solution starts to beat the s=0.8 ring
        Mreq = (var[i1] - var[i8]) / max(res2[i8] - res2[i1], 1e-30)
        print(f"{name:26s} {row[0]:9.3f} {row[1]:9.3f} {row[2]:9.3f}   "
              f"M_req(vs s=0.8) = {Mreq:8.0f}   res2(0.8)={res2[i8]:.3e} "
              f"Var(1)={var[i1]:.3g} Var(.8)={var[i8]:.3g}", flush=True)
        out[name] = dict(S=S.tolist(), res2=res2.tolist(), var=var.tolist(), argmin=row)
    json.dump(out, open(os.path.join(RL.OUTDIR, 'bias_theory.json'), 'w'), indent=1)
