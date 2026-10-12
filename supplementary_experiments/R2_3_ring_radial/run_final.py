"""
Sec. 4.6 ring: the single final run (split learning rate + wavelet bank).

    python run_final.py            # trains once (~5 min on one CPU core), writes figs/final/

Problem   : V = (1/4)(r^2 - r0^2)^2,  b = -grad V + omega (-x2, x1) = -(r^2 - r0^2) x + ...,
            generator (sigma^2/2) Laplacian,  r0 = 2, omega = 2, sigma = 1.
            Stationary density  rho ~ exp(-(r^2 - r0^2)^2 / (2 sigma^2))   (as printed in the paper).
Generator : (4, 64, 2) tanh, base N(0, I_4)                                 (unchanged)
Loss      : the manuscript's squared-mean weak-form loss, M = 1000          (unchanged)
Adversary : SGD 1e-2, 5 ascent steps every 10 epochs, 20,000 epochs         (unchanged)

Changed relative to the manuscript:
  1. Test functions: K = 200 ridge functions f = phi(w.x + b), of which
       120 plane waves        phi(z) = sin z,           w ~ N(0, I)
        80 Hermite wavelets   phi(z) = z exp(-z^2/2),   |w| in [2, 6] (width ~ ring width),
                              random direction, ridge offset uniform in [-3, 3].
  2. Generator optimizer: two-group learning-rate split (Muon convention):
       hidden weight matrix  lr 1e-1 (Muon update),
       biases + output layer lr 1e-2 (the manuscript's Adam rate).
"""
import os, sys, json, subprocess
import numpy as np
import torch
import ring_lab as RL

TAG = 'final_split_lr_wavelet'
CFG = dict(kappa=1.0, a=1.0,                       # drift = -grad V of the stated V; a = sigma^2
           bank='sin:120:1,herm1:80:4', spread=3.0,
           opt='muon', lr=0.1, head_lr=1e-2,
           M=1000, epochs=20000, d_base=4, hidden=(64,),
           adv_every=10, adv_steps=5, lr_test=1e-2, seed=3)

if __name__ == '__main__':
    torch.set_num_threads(1)
    rec, x, losses, pf = RL.train(CFG, device='cpu', verbose=True)
    rec['tag'] = TAG
    np.savez_compressed(os.path.join(RL.OUTDIR, f'{TAG}.npz'),
                        x=x[:100000].astype(np.float32), losses=losses)
    torch.save(pf.state_dict(), os.path.join(RL.OUTDIR, f'{TAG}.pt'))
    json.dump(rec, open(os.path.join(RL.OUTDIR, f'{TAG}.json'), 'w'), indent=1)
    print(RL.fmt(rec))
    print(f"loss: first {losses[0]:.3f} -> mean of last 500 epochs {rec['final_loss']:.4f}")
    print(f"<r> = {rec['r_mean']:.4f} (true {rec['r_mean_true']:.4f}),  "
          f"std(r) = {rec['r_std']:.4f} (true {rec['r_std_true']:.4f}),  "
          f"sample mean = ({x[:, 0].mean():+.4f}, {x[:, 1].mean():+.4f})")
    for script in ('make_paper_figs.py', 'make_mc_fig.py'):
        subprocess.run([sys.executable, os.path.join(RL.HERE, script), '--tag', TAG,
                        '--out', 'figs/final'], cwd=RL.HERE, check=True)
