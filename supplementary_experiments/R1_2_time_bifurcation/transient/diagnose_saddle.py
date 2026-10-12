"""
Can the training loss see the saddle blob?

For a trained WANPM (results/<tag>.pt) this computes, for every test function k, the
weak-form residual with (almost) no Monte-Carlo noise,

    R_k = E_T[f_k] - E_0[f_k] - T * E_{t~U(0,T)} E_{rho(t)}[ d_t f_k - L f_k ],

(i) for the LEARNED distribution (pushforward samples, N_L draws) and
(ii) for the TRUE solution (Euler-Maruyama SDE particles, each recorded at its own
     uniformly random time, N_T draws) -- a control: R_true must be 0 within noise.

signal_k = R_learned_k - R_true_k.  The noise the generator faces at batch size M is
sigma_k^2 = (Var_T + Var_0 + Var_int)_k / M, taken from per-sample variances.
SNR_k = |signal_k| / sigma_k.   E[training loss] = mean_k(R_learned_k^2) + mean_k sigma_k^2.

Also scores a family of candidate saddle-centred Gaussian wavelets (ridge along the split
direction u=(x1+x2)/sqrt2, width 1/a, time frequency kappa) the same way.

    python diagnose_saddle.py --tag D_adam
"""
import sys, os, json, argparse, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import wanpm_transient as W
import problem as P
from common_wanpm import TimeDepPushforward

dev = W.device
T = P.T_FINAL


def load(tag):
    ck = torch.load(os.path.join(W.OUTDIR, f'{tag}.pt'), map_location=dev)
    a = ck['args']
    pf = TimeDepPushforward(2, a['d_base'], tuple(a['hidden']), power=a['power']).to(dev)
    pf.load_state_dict(ck['pf'])
    bank = W.TimeDepBank(2, a['K'], a['frac_wave'], a['wave_act'], a['init_scale'],
                         a['centre_spread'], a['kappa_scale'], a['sin_w_scale'],
                         T=T, seed=a['seed'], frac_saddle=a.get('frac_saddle', 0.0),
                         saddle_a=tuple(a.get('saddle_a', [6.0, 10.0])),
                         saddle_kappa=a.get('saddle_kappa', 1.5),
                         saddle_centre=a.get('saddle_centre', 0.3),
                         frac_path=a.get('frac_path', 0.0), path_a=tuple(a.get('path_a', [4.0, 8.0])),
                         path_kappa=a.get('path_kappa', 1.5),
                         path_range=tuple(a.get('path_range', [0.4, 1.2]))).to(dev)
    bank.load_state_dict(ck['bank'])
    pf.eval()
    return pf, bank, a


class Acc:
    """running mean / variance per test function, float64"""
    def __init__(self, K):
        self.n = 0
        self.s = torch.zeros(K, dtype=torch.float64, device=dev)
        self.q = torch.zeros(K, dtype=torch.float64, device=dev)

    def add(self, c):                       # c: (B, K)
        self.n += c.shape[0]
        self.s += c.sum(0).double()
        self.q += (c * c).sum(0).double()

    def mean(self):
        return self.s / self.n

    def var(self):
        return (self.q / self.n - self.mean() ** 2).clamp(min=0)


def make_fields(s2):
    drift = lambda t, x: P.drift_torch(x)
    s2f = lambda t, x: s2 * torch.ones(x.shape[0], 1, device=x.device)
    return drift, s2f


@torch.no_grad()
def residual_learned(pf, bank, case, d_base, s2, N, B=50_000):
    drift, s2f = make_fields(s2)
    rho0 = P.rho0_sampler(case, dev)
    K = bank.w.shape[1]
    aI, aT, a0 = Acc(K), Acc(K), Acc(K)
    for _ in range(N // B):
        t = T * torch.rand(B, 1, device=dev)
        x = pf(t, rho0(B), W.sample_base(B, d_base))
        aI.add(T * (bank.dfdt(x, t) - bank.Lf(x, t, drift, s2f)))
        tT = T * torch.ones(B, 1, device=dev)
        aT.add(bank.f(pf(tT, rho0(B), W.sample_base(B, d_base)), tT))
        a0.add(bank.f(rho0(B), torch.zeros(B, 1, device=dev)))
    return finish(aT, a0, aI)


def finish(aT, a0, aI):
    R = aT.mean() - a0.mean() - aI.mean()
    var_sum = aT.var() + a0.var() + aI.var()          # per-sample variances
    return dict(R=R.cpu().numpy(), var_sum=var_sum.cpu().numpy(), N=aI.n)


@torch.no_grad()
def simulate_true(case, N, dt=1e-4, seed=1):
    """Euler-Maruyama; each particle recorded once, at its own uniform random step."""
    c = P.CASES[case]
    g = torch.Generator(device=dev).manual_seed(seed)
    n_steps = int(round(T / dt))
    x = torch.tensor(c['mu0'], device=dev) + c['s0'] * torch.randn(N, 2, device=dev, generator=g)
    x_init = x.clone()
    idx = torch.randint(0, n_steps, (N,), device=dev, generator=g)
    order = torch.argsort(idx)
    bounds = torch.searchsorted(idx[order], torch.arange(n_steps + 1, device=dev)).tolist()
    x_rec = torch.empty(N, 2, device=dev)
    amp = c['sigma'] * np.sqrt(dt)
    for j in range(n_steps):
        sl = order[bounds[j]:bounds[j + 1]]
        x_rec[sl] = x[sl]
        x = x + P.drift_torch(x) * dt + amp * torch.randn(N, 2, device=dev, generator=g)
    return dict(x_rec=x_rec, t_rec=(idx.float() * dt).unsqueeze(1), xT=x, x0=x_init)


@torch.no_grad()
def residual_true(bank, s2, sim, B=50_000):
    drift, s2f = make_fields(s2)
    K = bank.w.shape[1]
    aI, aT, a0 = Acc(K), Acc(K), Acc(K)
    N = sim['x_rec'].shape[0]
    for i in range(0, N - B + 1, B):
        x, t = sim['x_rec'][i:i + B], sim['t_rec'][i:i + B]
        aI.add(T * (bank.dfdt(x, t) - bank.Lf(x, t, drift, s2f)))
        aT.add(bank.f(sim['xT'][i:i + B], T * torch.ones(B, 1, device=dev)))
        a0.add(bank.f(sim['x0'][i:i + B], torch.zeros(B, 1, device=dev)))
    return finish(aT, a0, aI)


def candidate_bank(a_list, kappa_list):
    pairs = [(a, k) for a in a_list for k in kappa_list]
    bank = W.TimeDepBank(2, len(pairs), frac_wave=1.0, activation='gauss', seed=0).to(dev)
    with torch.no_grad():
        for j, (a, k) in enumerate(pairs):
            bank.w[:, j] = a * torch.tensor([1.0, 1.0]) / np.sqrt(2.0)
            bank.kappa[j] = k
            bank.b[j] = -k * 0.5                    # ridge passes through the saddle at t=0.5
    return bank, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='D_adam')
    ap.add_argument('--M', type=int, default=2000, help='training batch size the noise is quoted for')
    ap.add_argument('--n-learned', type=int, default=16_000_000)
    ap.add_argument('--n-true', type=int, default=4_000_000)
    args = ap.parse_args()

    pf, bank, a = load(args.tag)
    W.BASE = a.get('base', 'gaussian')
    case = a['case']
    s2 = P.CASES[case]['sigma'] ** 2
    K = bank.w.shape[1]
    print(f'[{args.tag}] case {case}  sigma^2={s2}  K={K}  d_base={a["d_base"]}  M={args.M}', flush=True)

    t0 = time.time()
    sim = simulate_true(case, args.n_true)
    print(f'  true SDE particles simulated ({time.time()-t0:.0f}s)', flush=True)

    out = {}

    def analyse(name, bank_):
        L = residual_learned(pf, bank_, case, a['d_base'], s2, args.n_learned)
        Tr = residual_true(bank_, s2, sim)
        sigma = np.sqrt(L['var_sum'] / args.M)                       # noise at batch size M
        sigma_true = np.sqrt(Tr['var_sum'] / args.M)                 # same, at the TRUE solution
        se_true = np.sqrt(Tr['var_sum'] / Tr['N'])
        se_learn = np.sqrt(L['var_sum'] / L['N'])
        signal = L['R'] - Tr['R']
        se_sig = np.sqrt(se_true ** 2 + se_learn ** 2)
        return dict(R_learned=L['R'], R_true=Tr['R'], signal=signal, sigma=sigma, sigma_true=sigma_true,
                    se_true=se_true, se_signal=se_sig, snr=np.abs(signal) / sigma,
                    z_true=Tr['R'] / se_true)

    res = analyse('trained', bank)
    n_sin = bank.n_sin
    typ = np.array(['sin'] * n_sin + ['wavelet'] * (K - n_sin - bank.n_saddle - bank.n_path)
                   + ['saddle'] * bank.n_saddle + ['path'] * bank.n_path)

    # ---- control ----
    z = res['z_true']
    print('\nCONTROL  (true solution should give R_k = 0):')
    print(f'  R_true_k / SE: mean z^2 = {np.mean(z**2):.2f} (1 expected),  max|z| = {np.abs(z).max():.1f},'
          f'  frac |z|>3 = {np.mean(np.abs(z) > 3):.3f}')

    # ---- loss decomposition ----
    sig_loss = float(np.mean(res['signal'] ** 2))
    learn_R2 = float(np.mean(res['R_learned'] ** 2))
    floor = float(np.mean(res['sigma'] ** 2))
    npz = np.load(os.path.join(W.OUTDIR, f'{args.tag}.npz'))
    logged = float(np.mean(npz['loss'][-50:]))            # last ~1000 epochs (stored every 20)
    print('\nLOSS DECOMPOSITION at the trained parameters (batch size M=%d):' % args.M)
    print(f'  systematic loss  mean_k (R_learned - R_true)^2 = {sig_loss:.3e}')
    print(f'  noise floor      mean_k sigma_k^2              = {floor:.3e}')
    print(f'  predicted training loss (R_learned^2 + floor)  = {learn_R2 + floor:.3e}   '
          f'(logged, last ~1000 epochs: {logged:.3e})')
    print(f'  systematic / noise = {sig_loss / floor:.3f}')
    floor_true = float(np.mean(res['sigma_true'] ** 2))
    exp_true = float(np.mean(res['R_true'] ** 2)) + floor_true
    print('\nDOES THE LOSS PREFER THE LEARNED SOLUTION TO THE TRUTH?  (expected loss E[R_hat^2] = R^2 + Var/M)')
    print(f'  learned solution: R^2 = {learn_R2:.3e}  +  Var/M = {floor:.3e}  =  {learn_R2 + floor:.3e}')
    print(f'  TRUE solution:    R^2 = {float(np.mean(res["R_true"]**2)):.3e}  +  Var/M = {floor_true:.3e}  =  {exp_true:.3e}')
    print(f'  => expected loss at the truth minus at the learned solution = {exp_true - (learn_R2 + floor):+.3e}'
          f'   (positive: the loss prefers the WRONG solution)')
    for Mx_ in (2000, 8000, 32000, 128000):
        f_l = float(np.mean(res['sigma'] ** 2)) * args.M / Mx_
        f_t = floor_true * args.M / Mx_
        print(f'     at M={Mx_:6d}: loss(truth) - loss(learned) = {(np.mean(res["R_true"]**2) + f_t) - (learn_R2 + f_l):+.3e}')

    # ---- per-type SNR ----
    print('\nPER-FUNCTION SNR = |signal_k| / sigma_k:')
    for tp in ('sin', 'wavelet', 'saddle', 'path'):
        m = typ == tp
        if m.sum() == 0:
            continue
        s = res['snr'][m]
        print(f'  {tp:8s} n={m.sum():4d}  median {np.median(s):.3f}  90th pct {np.percentile(s, 90):.3f}'
              f'  max {s.max():.3f}   #SNR>1: {(s > 1).sum():3d}   #SNR>0.3: {(s > 0.3).sum():3d}'
              f'   share of systematic loss {np.sum(res["signal"][m]**2) / np.sum(res["signal"]**2):.2f}')

    # top functions
    w = bank.w.detach().cpu().numpy(); kap = bank.kappa.detach().cpu().numpy(); b = bank.b.detach().cpu().numpy()
    order = np.argsort(-res['snr'])[:10]
    print('\nTOP-10 functions by SNR:  k  type   SNR   signal   sigma   |w|   kappa   ridge offset from saddle at t=0.5   width~1/|w|')
    for k in order:
        wn = np.linalg.norm(w[:, k])
        off = abs(kap[k] * 0.5 + b[k]) / wn
        print(f'  {k:4d} {typ[k]:7s} {res["snr"][k]:.3f} {res["signal"][k]:+.4f} {res["sigma"][k]:.4f} '
              f'{wn:6.2f} {kap[k]:+7.2f}   {off:6.3f}   {1 / wn:.3f}')

    # ---- candidate saddle wavelets ----
    a_list = [2, 3, 4, 6, 8, 10, 14]
    k_list = [0, 2, 5, 10]
    cb, pairs = candidate_bank(a_list, k_list)
    cres = analyse('candidates', cb)
    print('\nCANDIDATE saddle-centred Gaussian wavelets: SNR = |signal|/sigma at M=%d' % args.M)
    print('   width\\kappa ' + ''.join(f'{k:>9d}' for k in k_list))
    grid = np.zeros((len(a_list), len(k_list)))
    sgrid = np.zeros_like(grid)
    for j, (av, kv) in enumerate(pairs):
        grid[a_list.index(av), k_list.index(kv)] = cres['snr'][j]
        sgrid[a_list.index(av), k_list.index(kv)] = cres['signal'][j]
    for i, av in enumerate(a_list):
        print(f'   {1/av:6.3f}      ' + ''.join(f'{grid[i, j]:9.3f}' for j in range(len(k_list))))
    print(f'  best candidate SNR = {grid.max():.3f}; best trained-bank SNR = {res["snr"].max():.3f}')

    # ---- save ----
    js = dict(tag=args.tag, M=args.M, n_learned=args.n_learned, n_true=args.n_true,
              control=dict(mean_z2=float(np.mean(z ** 2)), max_abs_z=float(np.abs(z).max()),
                           frac_gt3=float(np.mean(np.abs(z) > 3))),
              loss=dict(systematic=sig_loss, noise_floor=floor, predicted=learn_R2 + floor, logged=logged,
                        noise_floor_true=floor_true, expected_loss_true=exp_true),
              snr_by_type={tp: dict(n=int((typ == tp).sum()), median=float(np.median(res['snr'][typ == tp])),
                                    max=float(res['snr'][typ == tp].max()),
                                    n_gt1=int((res['snr'][typ == tp] > 1).sum()))
                           for tp in ('sin', 'wavelet', 'saddle', 'path') if (typ == tp).any()},
              candidates=dict(a=a_list, kappa=k_list, snr=grid.tolist(), signal=sgrid.tolist()))
    with open(os.path.join(W.OUTDIR, f'{args.tag}_diagnosis.json'), 'w') as fh:
        json.dump(js, fh, indent=1)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for tp, col in (('sin', '#3b6fb6'), ('wavelet', '#d95f02'), ('saddle', '#2ca02c'), ('path', '#9467bd')):
        m = typ == tp
        if m.any():
            ax[0].scatter(res['sigma'][m], np.abs(res['signal'][m]), s=9, alpha=0.6, c=col, label=tp)
    lim = [1e-4, max(res['sigma'].max(), np.abs(res['signal']).max()) * 1.5]
    ax[0].plot(lim, lim, 'k--', lw=1, label='SNR = 1')
    ax[0].set_xscale('log'); ax[0].set_yscale('log')
    ax[0].set_xlabel(r'noise $\sigma_k$ at $M=%d$' % args.M); ax[0].set_ylabel(r'|signal$_k$| = |R$_k$(learned) - R$_k$(true)|')
    ax[0].set_title('per test function: signal vs noise'); ax[0].legend(fontsize=8)
    ax[1].hist(np.log10(np.maximum(res['snr'], 1e-4)), bins=40, color='#7ba7d1')
    ax[1].axvline(0, c='k', ls='--'); ax[1].set_xlabel(r'$\log_{10}$ SNR$_k$'); ax[1].set_title('SNR of the trained bank')
    im = ax[2].imshow(grid, origin='lower', cmap='viridis', aspect='auto')
    ax[2].set_xticks(range(len(k_list))); ax[2].set_xticklabels(k_list)
    ax[2].set_yticks(range(len(a_list))); ax[2].set_yticklabels([f'{1/v:.2f}' for v in a_list])
    ax[2].set_xlabel(r'time frequency $\kappa$'); ax[2].set_ylabel('wavelet width 1/a')
    for i in range(len(a_list)):
        for j in range(len(k_list)):
            ax[2].text(j, i, f'{grid[i, j]:.2f}', ha='center', va='center', color='w', fontsize=8)
    ax[2].set_title('candidate saddle wavelets: SNR'); plt.colorbar(im, ax=ax[2])
    plt.tight_layout()
    fig.savefig(os.path.join(W.OUTDIR, f'{args.tag}_diagnosis.png'), dpi=120)
    print('\nwrote', os.path.join(W.OUTDIR, f'{args.tag}_diagnosis.png'))


if __name__ == '__main__':
    main()
