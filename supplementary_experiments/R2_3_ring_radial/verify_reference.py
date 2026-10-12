"""
Which density is the stationary solution of the ring problem AS STATED / AS CODED?

Sec. 4.6 of the manuscript states
    V = (1/4)(r^2 - r0^2)^2,     b_radial = -grad V = -2 (r^2 - r0^2) x,
    a_ij = sigma^2 delta_ij / 2, rho_true ~ exp(-(r^2 - r0^2)^2 / (2 sigma^2)).
But grad[(1/4)(r^2-r0^2)^2] = (r^2 - r0^2) x, so the stated drift is the gradient
of (1/2)(r^2-r0^2)^2 -- twice the stated potential.

Write the radial drift as  b = -kappa (r^2 - r0^2) x  and the generator's
diffusion as (a/2) Laplacian.  Then rho ~ exp(-kappa (r^2-r0^2)^2 / (2a)), i.e.
u = r^2 ~ N(r0^2, a/kappa):

    kappa/a = 1 : the density the manuscript PLOTS as "analytical"   (std_r ~ 0.259)
    kappa/a = 2 : drift as coded (kappa=2), generator (sigma^2/2) Lap (std_r ~ 0.183)
    kappa/a = 4 : drift as coded, a_ij = sigma^2/2 taken literally    (std_r ~ 0.129)

Two independent checks, neither involving a neural network:
  (1) Euler-Maruyama simulation of dX = b dt + sqrt(a) dW;
  (2) the weak-form residual E[Lf] on exact samples of each candidate density.
"""
import numpy as np, torch
import ring_lab as RL

rng = np.random.default_rng(0)


def em_radial_std(kappa, a, n=20000, dt=5e-4, T=6.0):
    x = RL.sample_true(n, 1.0, rng)                     # start from the manuscript's density
    rs = []
    steps = int(T / dt)
    for i in range(steps):
        r2 = (x ** 2).sum(1, keepdims=True)
        b = -kappa * (r2 - RL.R0 ** 2) * x + RL.OMEGA * np.stack([-x[:, 1], x[:, 0]], 1)
        x = x + b * dt + np.sqrt(a * dt) * rng.standard_normal(x.shape)
        if i > steps // 2 and i % 200 == 0:
            rs.append(np.hypot(x[:, 0], x[:, 1]))
    r = np.concatenate(rs)
    return r.mean(), r.std()


def exact_radial(var_u):
    r = np.hypot(*RL.sample_true(2_000_000, var_u, rng).T)
    return r.mean(), r.std()


print("(1) Euler-Maruyama vs candidate stationary densities  [radial mean, radial std]")
for kappa, a, label in [(2.0, 1.0, 'drift as coded, a = sigma^2       (rebuttal reproductions)'),
                        (2.0, 0.5, 'drift as coded, a = sigma^2/2     (manuscript text, literal)'),
                        (1.0, 1.0, 'drift = -grad V of stated V, a = sigma^2')]:
    m, s = em_radial_std(kappa, a)
    print(f"  kappa={kappa:g} a={a:g}  {label}")
    print(f"      simulated SDE          : mean {m:.4f}  std {s:.4f}")
    for ratio in (1.0, 2.0, 4.0):
        me, se = exact_radial(1.0 / ratio)
        mark = '  <-- matches' if abs(se - s) < 0.01 else ''
        print(f"      exp(-{ratio:g}(r^2-r0^2)^2/2)   : mean {me:.4f}  std {se:.4f}{mark}")

print("\n(2) weak-form residual  mean_k (E[L f_k])^2  on EXACT samples, 300 plane waves, N=400k")
torch.manual_seed(0)
bank = RL.Bank('sin:300:1', seed=0).double()
for kappa, a in [(2.0, 1.0), (2.0, 0.5), (1.0, 1.0)]:
    row = []
    for ratio in (1.0, 2.0, 4.0):
        x = torch.tensor(RL.sample_true(400000, 1.0 / ratio, rng))
        with torch.no_grad():
            L = bank.Lf(x, a, kappa)
        m = L.mean(0); v = L.var(0)
        row.append(float((m ** 2 - v / len(x)).mean()))
    print(f"  kappa={kappa:g} a={a:g}:  " +
          "   ".join(f"ratio {r:g}: {v:+.2e}" for r, v in zip((1, 2, 4), row)))
print("  (the true stationary density is the one whose residual is ~0)")
