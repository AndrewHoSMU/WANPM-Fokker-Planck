"""
Reviewer 3, major comment 5: "Could the authors provide more intuition behind
their choice of optimizers? Specifically, why is Adam used for optimizing
theta, while SGD is used for the test functions? Have the authors
experimented with other optimizers, such as AdamW or more recent
geometry-aware methods including SHAMPOO, SOAP, or Muon...?"

We answer this on main.tex's OWN Sec. 4.2 example (the 1D double-peak /
double-well potential), reproduced at exactly the configuration the paper
specifies, changing ONLY the generator optimizer:

    V(x) = (x^2-1)^2,  sigma = 0.6,  b(x) = -4x^3 + 4x
    pushforward  (8, 100, 1) Tanh MLP, 8-dim Gaussian base
    test functions  f^(k)(x) = sin(w^(k) x + b^(k)), K = 100
    16,000 epochs, batch M = 1000
    baseline: Adam lr 2e-3 (generator) + SGD lr 1e-2 (test functions),
              5 adversary ascent steps per generator update, every 10 epochs

Compared generator optimizers: SGD, Adam (paper), AdamW, Shampoo, SOAP,
Muon. The adversary is held at the paper's SGD lr 1e-2 throughout, so the
comparison isolates the theta-side question the reviewer actually asks
(a separate adversary-side sweep is run by --mode adversary).

FAIRNESS: optimizers have wildly different natural learning-rate scales
(Muon's default is 2e-2, Adam's 2e-3), so comparing at default LRs would be
meaningless. --mode sweep runs a per-optimizer LR grid; --mode seeds then
re-runs each optimizer at ITS OWN best LR across several seeds. Only the
seed-averaged numbers should be quoted.

METRIC: the training loss is NOT comparable across optimizers here -- it is
a minimax objective whose value depends on the adversary's current state, so
a lower loss can just mean a weaker adversary. We therefore score every run
by an external, optimizer-independent accuracy metric: the Hellinger and
total-variation distance between the learned sample density and the
analytical Boltzmann solution rho ~ exp(-2V/sigma^2), plus the well-population
balance (exactly 0.5/0.5 by symmetry) and wall-clock cost.

Usage:
    python run_experiment.py --mode sweep       # per-optimizer LR grid
    python run_experiment.py --mode seeds       # best-LR, multi-seed
    python run_experiment.py --mode adversary   # test-function-side sweep
    python run_experiment.py --mode plot        # figures + summary from JSON
"""
import sys, os, json, time, argparse, itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'rebuttal_stage1'))
import numpy as np
import torch
from scipy.integrate import quad

from common_wanpm import (SteadyPushforward, SteadyTestFunctions, steady_loss,
                          gaussian_base_sampler)

import pytorch_optimizer as po
from pytorch_optimizer.optimizer.shampoo_utils import LayerWiseGrafting

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'results')
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# main.tex Sec. 4.2 problem + training configuration, verbatim
# ---------------------------------------------------------------------------
SIGMA = 0.6
SIGMA2 = SIGMA ** 2
D_BASE = 8
HIDDEN = (100,)
K_TEST = 100
N_EPOCHS = 16000
BATCH_M = 1000
ADV_EVERY = 10          # adversary acts every 10 epochs ...
ADV_STEPS = 5           # ... 5 ascent steps each time
LR_TEST_PAPER = 1e-2    # SGD on the test functions
LR_GEN_PAPER = 2e-3     # Adam on theta


def V(x):
    return (x ** 2 - 1) ** 2


def drift_fn(x):
    return -4 * x ** 3 + 4 * x


def sigma2_fn(x):
    return SIGMA2 * torch.ones(x.shape[0], 1, device=x.device)


# ---------------------------------------------------------------------------
# sigma-annealing curriculum.
#
# The relative barrier here is 2*Delta_V/sigma^2 = 2/0.36 = 5.6, squarely in
# the range where stage-1 (R1_4) and stage-2 (R1_5) both found that training
# this method at a fixed sigma collapses catastrophically -- and it does so
# here too: at fixed sigma=0.6 the generator collapses to a point mass
# (learned std ~2e-4 against a true std of 0.972) for every optimizer we
# tried, so a fixed-sigma comparison would only be ranking failure modes.
# We therefore use the same curriculum stage 2 documents as "load-bearing,
# not optional": start from an easy, low-barrier version of the SAME
# potential and anneal sigma^2 down to the target.
# ---------------------------------------------------------------------------
SIGMA2_START = 1.5
WARMUP_FRAC, ANNEAL_FRAC = 0.10, 0.55   # remainder is fine-tuning at target


def sigma2_at_epoch(epoch, n_epochs, anneal=True):
    if not anneal:
        return SIGMA2
    w_end = int(WARMUP_FRAC * n_epochs)
    a_end = w_end + int(ANNEAL_FRAC * n_epochs)
    if epoch < w_end:
        return SIGMA2_START
    if epoch < a_end:
        progress = (epoch - w_end) / max(1, a_end - w_end)
        cos_progress = 0.5 * (1 - np.cos(np.pi * progress))
        return SIGMA2_START + (SIGMA2 - SIGMA2_START) * cos_progress
    return SIGMA2


def unnorm_density(x):
    return np.exp(-2 * V(x) / SIGMA2)


_Z, _ = quad(unnorm_density, -6, 6, limit=400)


def true_pdf(x):
    return unnorm_density(x) / _Z


# ---------------------------------------------------------------------------
# Optimizer factories.
#
# Weight decay is set to 0 for every method EXCEPT AdamW, whose decoupled
# weight decay is the entire point of including it (at wd=0 AdamW is
# algorithmically identical to Adam and would be a wasted row). Reported.
# ---------------------------------------------------------------------------
def _split_muon_groups(model, lr, adamw_lr):
    """Standard Muon convention: Muon on hidden 2-D weight matrices only;
    AdamW on all 1-D params (biases) and on the output/head layer.

    For the Sec. 4.2 net (8,100,1) this means Muon governs exactly ONE
    matrix, net[0].weight of shape (100, 8) -- worth keeping in mind when
    reading the result, since Muon is designed for deep stacks of hidden
    matrices and a single-hidden-layer MLP gives it very little to act on.
    """
    named = list(model.named_parameters())
    linear_idx = sorted({int(n.split('.')[1]) for n, _ in named})
    head_idx = linear_idx[-1]
    muon_p, adamw_p = [], []
    for n, p in named:
        idx = int(n.split('.')[1])
        (muon_p if (p.ndim >= 2 and idx != head_idx) else adamw_p).append(p)
    groups = [
        dict(params=muon_p, lr=lr, weight_decay=0.0, use_muon=True),
        dict(params=adamw_p, lr=adamw_lr, weight_decay=0.0, use_muon=False),
    ]
    return groups


def make_gen_opt(name, model, lr):
    p = list(model.parameters())
    if name == 'SGD':
        return torch.optim.SGD(p, lr=lr)
    if name == 'Adam':
        return torch.optim.Adam(p, lr=lr)
    if name == 'AdamW':
        return torch.optim.AdamW(p, lr=lr, weight_decay=1e-2)
    if name == 'Shampoo':
        # preconditioning_compute_steps MUST be 1 here. pytorch-optimizer's
        # plain Shampoo leaves `inv_pre_cond` at all-zeros until the first
        # recompute (shampoo.py:140-141), so any value > 1 applies a zero
        # (then stale) preconditioner for the opening steps and the run
        # NaNs out immediately at every learning rate we tried. Recomputing
        # the inverse root every step is stable but ~3x slower per epoch.
        return po.Shampoo(p, lr=lr, momentum=0.9, weight_decay=0.0,
                          preconditioning_compute_steps=1)
    if name == 'ScalableShampoo':
        # The production Shampoo of Anil et al.: identical preconditioner
        # geometry, but the update is *grafted* onto SGD's step magnitude.
        # Grafting is what makes Shampoo robust when the preconditioner is
        # badly rank-deficient -- which it always is here, since e.g. the
        # (100,8) weight yields a 100x100 statistic of rank <= 8, leaving 92
        # eigenvalues sitting at matrix_eps.
        return po.ScalableShampoo(p, lr=lr, weight_decay=0.0,
                                  start_preconditioning_step=25,
                                  preconditioning_compute_steps=10,
                                  graft_type=LayerWiseGrafting.SGD)
    if name == 'SOAP':
        return po.SOAP(p, lr=lr, weight_decay=0.0, precondition_frequency=10)
    if name == 'Muon':
        return po.Muon(_split_muon_groups(model, lr, adamw_lr=LR_GEN_PAPER),
                       weight_decay=0.0)
    raise ValueError(name)


def make_test_opt(name, params, lr):
    if name == 'SGD':
        return torch.optim.SGD(params, lr=lr)
    if name == 'Adam':
        return torch.optim.Adam(params, lr=lr)
    if name == 'AdamW':
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    if name == 'Shampoo':
        return po.Shampoo(params, lr=lr, momentum=0.9, weight_decay=0.0,
                          preconditioning_compute_steps=10)
    if name == 'SOAP':
        # w is (1, K) and b is (K,); 1-D params need precondition_1d to be
        # touched at all by SOAP's preconditioner.
        return po.SOAP(params, lr=lr, weight_decay=0.0,
                       precondition_frequency=10, precondition_1d=True)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Evaluation against the analytical Boltzmann solution
# ---------------------------------------------------------------------------
GRID_LO, GRID_HI, N_BINS, N_EVAL = -2.5, 2.5, 250, 200000


def evaluate(pf):
    with torch.no_grad():
        r = torch.randn(N_EVAL, D_BASE, device=device)
        x = pf(r).cpu().numpy().flatten()

    edges = np.linspace(GRID_LO, GRID_HI, N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dx = edges[1] - edges[0]
    p_hat, _ = np.histogram(x, bins=edges, density=True)
    p_true = true_pdf(centers)
    # renormalise the truth over the same finite window so both integrate to 1
    p_true = p_true / (p_true.sum() * dx)
    frac_in_window = float(((x >= GRID_LO) & (x <= GRID_HI)).mean())
    p_hat = p_hat * frac_in_window  # mass outside the window counts as error

    hellinger = float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(p_hat * p_true)) * dx)))
    tv = float(0.5 * np.sum(np.abs(p_hat - p_true)) * dx)
    frac_right = float((x > 0).mean())
    return {
        'hellinger': hellinger,
        'tv': tv,
        'frac_right': frac_right,
        'well_balance_err': abs(frac_right - 0.5),
        'mean': float(x.mean()),
        'std': float(x.std()),
        'frac_in_window': frac_in_window,
    }


# ---------------------------------------------------------------------------
# One training run
# ---------------------------------------------------------------------------
def run(gen_name, lr_gen, seed=0, test_name='SGD', lr_test=LR_TEST_PAPER,
        n_epochs=N_EPOCHS, anneal=True, adv_steps=None, verbose=False):
    adv_steps = ADV_STEPS if adv_steps is None else adv_steps
    torch.manual_seed(seed)
    np.random.seed(seed)
    pf = SteadyPushforward(d_base=D_BASE, n_out=1, hidden=HIDDEN).to(device)
    tf = SteadyTestFunctions(n_in=1, K=K_TEST).to(device)

    gen_opt = make_gen_opt(gen_name, pf, lr_gen)
    test_opt = make_test_opt(test_name, tf.parameters(), lr_test)

    cur_sigma2 = [SIGMA2]

    def sigma2_cur(x):
        return cur_sigma2[0] * torch.ones(x.shape[0], 1, device=x.device)

    def loss_fn():
        return steady_loss(pf, tf, M=BATCH_M, drift_fn=drift_fn,
                           sigma2_fn=sigma2_cur,
                           base_sampler=gaussian_base_sampler, device=device)

    loss_log = np.empty(n_epochs, dtype=np.float64)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    diverged = False
    for epoch in range(n_epochs):
        cur_sigma2[0] = sigma2_at_epoch(epoch, n_epochs, anneal)

        if epoch % ADV_EVERY == 0 and epoch > 0:
            for _ in range(adv_steps):
                loss, _ = loss_fn()
                test_opt.zero_grad(); (-loss).backward(); test_opt.step()

        loss, _ = loss_fn()
        gen_opt.zero_grad(); loss.backward(); gen_opt.step()
        lv = loss.item()
        loss_log[epoch] = lv
        if not np.isfinite(lv):
            diverged = True
            loss_log[epoch:] = np.nan
            break
        if verbose and (epoch % 4000 == 0 or epoch == n_epochs - 1):
            print(f"    epoch {epoch:6d}/{n_epochs}  loss={lv:.6e}")

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.time() - t0

    rec = {
        'gen_opt': gen_name, 'lr_gen': lr_gen, 'test_opt': test_name,
        'lr_test': lr_test, 'seed': seed, 'n_epochs': n_epochs,
        'anneal': anneal, 'adv_steps': adv_steps,
        'train_time_s': train_time, 'diverged': diverged,
    }
    if diverged:
        rec.update({'hellinger': float('nan'), 'tv': float('nan'),
                    'frac_right': float('nan'), 'well_balance_err': float('nan'),
                    'mean': float('nan'), 'std': float('nan'),
                    'frac_in_window': float('nan'),
                    'final_loss': float('nan')})
    else:
        rec.update(evaluate(pf))
        rec['final_loss'] = float(np.nanmean(loss_log[-500:]))
    rec['loss_log_every50'] = loss_log[::50].tolist()
    return rec


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
SWEEP_GRID = {
    'SGD':     [1e-3, 1e-2, 5e-2, 1e-1, 3e-1],
    'Adam':    [5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
    'AdamW':   [5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
    'Shampoo': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    'ScalableShampoo': [5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
    'SOAP':    [5e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    'Muon':    [2e-3, 5e-3, 1e-2, 2e-2, 5e-2],
}

# SGD and plain Shampoo both went straight from undertrained to collapsed
# across adjacent points of the coarse grid, so we refine between those points
# before concluding they have no usable learning rate.
FINE_GRID = {
    'SGD':     [2e-2, 3e-2],
    'Shampoo': [2e-3, 3e-3],
    # SOAP's best coarse result (0.108 at 3e-2) sat on the TOP edge of its
    # grid, so the coarse optimum is a boundary value and cannot be quoted as
    # SOAP's best without extending upward until it turns over.
    'SOAP':    [5e-2, 1e-1, 2e-1],
    # Every coarse Muon LR (2e-3 .. 5e-2) collapsed. Probe below the grid
    # before concluding Muon simply fails here. Note the structural caveat:
    # on this (8,100,1) net Muon governs exactly ONE matrix (net[0].weight,
    # 100x8) -- the head and all biases are always on its internal AdamW at
    # LR_GEN_PAPER -- so this sweep moves far less of the network than the
    # other optimizers' sweeps do.
    'Muon':    [1e-1, 2e-1, 5e-1],
}

ADV_GRID = {
    'SGD':     [1e-3, 1e-2, 1e-1],   # 1e-2 is the paper's choice
    'Adam':    [1e-4, 1e-3, 1e-2],
    'AdamW':   [1e-4, 1e-3, 1e-2],
    'Shampoo': [1e-4, 1e-3, 1e-2],
    'SOAP':    [1e-4, 1e-3, 1e-2],
}

SEEDS = [0, 1, 2]


def _save(records, fname):
    with open(os.path.join(OUTDIR, fname), 'w') as fh:
        json.dump(records, fh, indent=2)
    print(f"  -> wrote {fname} ({len(records)} runs)")


def mode_sweep(args):
    records = []
    for name, lrs in SWEEP_GRID.items():
        for lr in lrs:
            print(f"[sweep] {name:8s} lr={lr:g} ...", flush=True)
            rec = run(name, lr, seed=0, n_epochs=args.epochs)
            print(f"         hellinger={rec['hellinger']:.4f} "
                  f"tv={rec['tv']:.4f} bal_err={rec['well_balance_err']:.4f} "
                  f"t={rec['train_time_s']:.1f}s"
                  f"{'  [DIVERGED]' if rec['diverged'] else ''}", flush=True)
            records.append(rec)
            _save(records, 'sweep.json')
    return records


def mode_fine(args):
    """Finer LR grids for the optimizers whose coarse sweep jumped straight
    from 'undertrained' to 'collapsed', so we can't claim they have no working
    setting without ruling out a narrow window between grid points."""
    records = []
    for name, lrs in FINE_GRID.items():
        for lr in lrs:
            print(f"[fine] {name:16s} lr={lr:g} ...", flush=True)
            rec = run(name, lr, seed=0, n_epochs=args.epochs)
            print(f"        hellinger={rec['hellinger']:.4f} "
                  f"tv={rec['tv']:.4f} bal_err={rec['well_balance_err']:.4f} "
                  f"t={rec['train_time_s']:.1f}s"
                  f"{'  [DIVERGED]' if rec['diverged'] else ''}", flush=True)
            records.append(rec)
            _save(records, 'fine.json')
    return records


# A run is "collapsed" when essentially all mass sits in one well. Collapse
# must be detected by MODE BALANCE, not by Hellinger: a collapsed run scores
# Hellinger ~0.54, which is numerically *better* than an honest but
# under-trained run (plain Shampoo reaches only 0.62-0.92 while still finding
# both wells), so ranking failures by Hellinger alone would select the
# qualitatively worse failure and report it as the method's best showing.
#
# Collapse is also stochastic rather than a clean function of LR (Adam
# collapses at 5e-3 yet succeeds at both 2e-3 and 1e-2), so single-seed LR
# selection is unreliable. We therefore carry the top-2 non-collapsed LRs per
# optimizer into the multi-seed stage and rank by seed-mean, not seed-0.
COLLAPSE_BALANCE_ERR = 0.40   # |P(x>0) - 0.5| this large => one well only
N_LR_FINALISTS = 2


def mode_seeds(args):
    sweep = []
    for fname in ('sweep.json', 'fine.json'):
        path = os.path.join(OUTDIR, fname)
        if os.path.exists(path):
            with open(path) as fh:
                sweep.extend(json.load(fh))

    finalists = {}
    for name in SWEEP_GRID:
        cand = [r for r in sweep if r['gen_opt'] == name and not r['diverged']
                and np.isfinite(r['hellinger'])]
        good = [r for r in cand
                if r['well_balance_err'] < COLLAPSE_BALANCE_ERR]
        pool = good if good else cand   # fall back so failures are still reported
        if not pool:
            print(f"[seeds] {name}: no finite-loss LR at all, skipping")
            continue
        pool = sorted(pool, key=lambda r: r['hellinger'])[:N_LR_FINALISTS]
        finalists[name] = [r['lr_gen'] for r in pool]
        tag = '' if good else '  (every LR collapsed into one well)'
        print(f"[seeds] {name:16s} finalist lrs = "
              f"{', '.join(f'{l:g}' for l in finalists[name])}{tag}")

    records = []
    for name, lrs in finalists.items():
        for lr in lrs:
            for seed in SEEDS:
                print(f"[seeds] {name:16s} lr={lr:g} seed={seed} ...", flush=True)
                rec = run(name, lr, seed=seed, n_epochs=args.epochs)
                print(f"         hellinger={rec['hellinger']:.4f} "
                      f"tv={rec['tv']:.4f} bal_err={rec['well_balance_err']:.4f} "
                      f"t={rec['train_time_s']:.1f}s"
                      f"{'  [DIVERGED]' if rec['diverged'] else ''}", flush=True)
                records.append(rec)
                _save(records, 'seeds.json')
    return records


def mode_adversary(args):
    """Hold the generator at the paper's Adam lr 2e-3; vary the test-function
    optimizer. Muon is excluded by construction: the test functions are not a
    network, they are raw plane-wave parameters (w of shape (1,K), b of shape
    (K,)), and Muon's orthogonalized update on a (1,K) matrix would force a
    single uniform singular value, destroying exactly the per-mode frequency
    selectivity the adversary needs."""
    records = []
    for name, lrs in ADV_GRID.items():
        for lr in lrs:
            print(f"[adv] test={name:8s} lr={lr:g} ...", flush=True)
            rec = run('Adam', LR_GEN_PAPER, seed=0, test_name=name,
                      lr_test=lr, n_epochs=args.epochs)
            print(f"       hellinger={rec['hellinger']:.4f} "
                  f"tv={rec['tv']:.4f} bal_err={rec['well_balance_err']:.4f} "
                  f"t={rec['train_time_s']:.1f}s"
                  f"{'  [DIVERGED]' if rec['diverged'] else ''}", flush=True)
            records.append(rec)
            _save(records, 'adversary.json')
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True,
                    choices=['sweep', 'fine', 'seeds', 'adversary', 'plot', 'smoke'])
    ap.add_argument('--epochs', type=int, default=N_EPOCHS)
    args = ap.parse_args()

    print(f"device: {device}")
    if device.type == 'cuda':
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"Sec 4.2 config: sigma={SIGMA}, d_base={D_BASE}, hidden={HIDDEN}, "
          f"K={K_TEST}, epochs={args.epochs}, M={BATCH_M}")

    if args.mode == 'smoke':
        for name in SWEEP_GRID:
            lr = SWEEP_GRID[name][2]
            t0 = time.time()
            rec = run(name, lr, seed=0, n_epochs=200)
            print(f"  {name:8s} lr={lr:g}  ok, {time.time()-t0:.1f}s for 200 epochs "
                  f"-> est {(time.time()-t0)*N_EPOCHS/200/60:.1f} min for {N_EPOCHS}")
    elif args.mode == 'sweep':
        mode_sweep(args)
    elif args.mode == 'fine':
        mode_fine(args)
    elif args.mode == 'seeds':
        mode_seeds(args)
    elif args.mode == 'adversary':
        mode_adversary(args)
    elif args.mode == 'plot':
        import make_plots
        make_plots.main(OUTDIR)


if __name__ == '__main__':
    main()
