"""
Standalone re-run of the mixed test-function bank experiment, this time also
logging the training loss curve (not saved by the original run).

Reuses closer_wells.py's problem definition (a=0.7) and 'mixed' config
(75/25 plane-wave/Gaussian-wavelet bank, Adam lr=5e-3) exactly, but with the
training loop inlined here so the loss can be recorded every epoch.
"""
import os, json, time
import numpy as np
import torch

import closer_wells as C
import asym_muon as A

OUTDIR = C.OUTDIR
device = C.device
SEEDS = [0, 1, 2]
LOG_EVERY = 20  # subsample the loss curve for storage/plotting


def run_logged(seed, n_epochs=16000):
    torch.manual_seed(seed); np.random.seed(seed)
    pf = A.SteadyPushforward(d_base=C.D_BASE, n_out=1, hidden=C.HIDDEN).to(device)
    tf = C.make_tf('mix-gauss', seed=seed)
    gen_opt = torch.optim.Adam(pf.parameters(), lr=5e-3)
    test_opt = torch.optim.SGD(tf.parameters(), lr=A.LR_TEST_PAPER)
    cur = [C.SIGMA2]

    def s2fn(x):
        return cur[0] * torch.ones(x.shape[0], 1, device=x.device)

    def loss_fn():
        return A.steady_loss(pf, tf, M=C.BATCH_M, drift_fn=C.drift_fn,
                             sigma2_fn=s2fn, base_sampler=A.gaussian_base_sampler,
                             device=device)

    loss_log, sigma_log = [], []
    t0 = time.time()
    diverged = False
    for ep in range(n_epochs):
        cur[0] = A.sigma2_at_epoch(ep, n_epochs, True, A.SIGMA2_START, C.SIGMA2)
        if ep % A.ADV_EVERY == 0 and ep > 0:
            for _ in range(A.ADV_STEPS):
                l, _ = loss_fn()
                test_opt.zero_grad(); (-l).backward(); test_opt.step()
        l, _ = loss_fn()
        gen_opt.zero_grad(); l.backward(); gen_opt.step()
        lv = l.item()
        if not np.isfinite(lv):
            diverged = True
            break
        if ep % LOG_EVERY == 0 or ep == n_epochs - 1:
            loss_log.append(lv)
            sigma_log.append(cur[0])
    dt = time.time() - t0

    rec = {'seed': seed, 'n_epochs': n_epochs, 'train_time_s': dt,
          'diverged': diverged}
    x = None
    if not diverged:
        with torch.no_grad():
            x = pf(torch.randn(C.N_EVAL, C.D_BASE, device=device)).cpu().numpy().ravel()
        rec.update(C.evaluate(x))
    rec['loss_log'] = loss_log
    rec['sigma_log'] = sigma_log
    rec['log_every'] = LOG_EVERY
    return rec, x


def main():
    print(f"a_well={C.A_WELL}  true_frac_right={C.TRUE_FRAC_RIGHT:.4f}  "
          f"true_mean={C.TRUE_MEAN:.4f}")
    recs, samples = [], {}
    for seed in SEEDS:
        print(f"[mixed-bank] seed={seed} ...", flush=True)
        rec, x = run_logged(seed)
        print(f"    frac_right={rec['frac_right']:.4f} (true {C.TRUE_FRAC_RIGHT:.4f})  "
              f"H={rec['hellinger']:.4f}  captured={100*rec['asymmetry_captured']:.1f}%  "
              f"final_loss={rec['loss_log'][-1]:.3e}  ({rec['train_time_s']:.0f}s)",
              flush=True)
        recs.append(rec)
        if x is not None:
            samples[f"s{seed}"] = x.astype(np.float32)
        with open(os.path.join(OUTDIR, 'mixed_bank_standalone.json'), 'w') as fh:
            json.dump({'a_well': C.A_WELL, 'true_frac_right': C.TRUE_FRAC_RIGHT,
                      'true_frac_left': C.TRUE_FRAC_LEFT, 'true_mean': C.TRUE_MEAN,
                      'true_std': C.TRUE_STD, 'records': recs}, fh, indent=2)
        np.savez_compressed(os.path.join(OUTDIR, 'mixed_bank_standalone_samples.npz'),
                            **samples)

    cap = np.array([100 * r['asymmetry_captured'] for r in recs])
    h = np.array([r['hellinger'] for r in recs])
    print()
    print(f"captured {cap.mean():.1f}% +/- {cap.std():.1f}   H={h.mean():.4f}")


if __name__ == '__main__':
    main()
