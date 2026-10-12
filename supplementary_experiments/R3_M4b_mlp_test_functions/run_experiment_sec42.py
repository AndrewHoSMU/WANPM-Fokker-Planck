"""
Reviewer 3, major comment 4 (second half), REDESIGNED on the revised
manuscript's Sec. 4.2 instance: "... whether the quality of the results could
be improved by using a richer class of test functions, such as multilayer
perceptrons with depth greater than two."

The first version (run_experiment.py) used an ad-hoc sigma^2=1 double well on
which the plane-wave baseline itself was poor (L2 0.19, loss blow-up), so the
comparison ranked two failures. This version runs both test-function classes
inside the exact training configuration of main.tex Sec. 4.2, where the
plane-wave bank is known to work (Hellinger 0.110 +- 0.025 over three seeds):

    V(x) = (x^2-1)^2, sigma = 0.6, b(x) = -4x^3 + 4x
    pushforward (8, 100, 1) tanh MLP, 8-dim Gaussian base
    K = 100 test functions, 16,000 epochs, batch M = 1000
    generator: Muon lr 2e-1 on the hidden matrix, AdamW lr 2e-3 on the rest
    adversary: SGD lr 1e-2, 5 ascent steps every 10 epochs
    sigma^2 annealed 1.5 -> 0.36 (10% hold, 55% cosine, 35% hold)

ONLY the test-function class changes between the two arms:
    pw   f^(k)(x) = sin(w^(k) x + b^(k))            (the manuscript's bank)
    mlp  f^(k)(x) = k-th output of a depth-3 tanh MLP 1 -> 100 -> 100 -> 100 -> K
The last hidden width equals K so that the K test functions can be linearly
independent (the first version's width-16 trunk capped the bank at rank 16).

Three things the first version got wrong and this one fixes:
  * Lf for the MLP bank is computed by exact second-order forward propagation
    (f, f', f'' pushed through each tanh layer), not by a K-fold loop of
    double-backward calls. The "70x slower" of the first version measured
    that loop, not the test-function class. `--check` verifies the forward
    propagation against autograd and times both.
  * Accuracy is the Sec. 4.2 metric set (Hellinger, TV, well balance, bridge
    mass) from muon_only.metrics, not a histogram L2 error.
  * The two banks' own training losses are NOT comparable (the loss is not
    normalised by test-function magnitude), so each trained generator is also
    scored by a COMMON referee: the weak-form residual against one frozen
    plane-wave probe bank, in units of its Monte Carlo noise under the exact
    density (rms z ~ 1 means "indistinguishable from the true solution").

Usage:
    python run_experiment_sec42.py --check                 # derivative check + step timing, no training
    python run_experiment_sec42.py --arms pw mlp --seeds 0 # train
    python run_experiment_sec42.py --plot --seeds 0        # figure from saved runs
Optional adversary-tuning control for the MLP arm:
    python run_experiment_sec42.py --arms mlp --test-opt Adam --lr-test 1e-3 --tag _adam
"""
import sys, os, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'R3_M5_optimizer_comparison'))
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common_wanpm import mlp
import run_experiment as R          # Sec. 4.2 problem + training configuration
from muon_only import metrics, MUON_LR

OUTDIR = os.path.join(HERE, 'results_sec42')
os.makedirs(OUTDIR, exist_ok=True)
RESULTS_JSON = os.path.join(OUTDIR, 'results.json')

MLP_HIDDEN = (100, 100, 100)
N_SAMPLES = 200000
PROBE_K, PROBE_W_SCALE, PROBE_SEED = 2000, 4.0, 12345


class MLPTestFunctions(nn.Module):
    """f^(k)(x) = MLP(x)_k for scalar x, with f' and f'' propagated in closed form."""

    def __init__(self, K, hidden=MLP_HIDDEN):
        super().__init__()
        self.net = mlp(1, K, hidden, activation=nn.Tanh)

    def jet(self, x):
        """x: (M, 1) -> f, f', f'' each (M, K)."""
        lin = [m for m in self.net if isinstance(m, nn.Linear)]
        z, z1, z2 = x, torch.ones_like(x), torch.zeros_like(x)
        for layer in lin[:-1]:
            h = layer(z)
            h1 = z1 @ layer.weight.T
            h2 = z2 @ layer.weight.T
            t = torch.tanh(h)
            s = 1 - t * t
            z, z1, z2 = t, s * h1, s * h2 - 2 * t * s * h1 * h1
        W = lin[-1].weight
        return lin[-1](z), z1 @ W.T, z2 @ W.T

    def Lf(self, x, drift_fn, sigma2_fn):
        _, f1, f2 = self.jet(x)
        return -drift_fn(x) * f1 - 0.5 * sigma2_fn(x) * f2

    def Lf_autograd(self, x, drift_fn, sigma2_fn):
        """Reference implementation (the first version's K-fold double backward)."""
        f = self.net(x)
        f1, f2 = [], []
        for k in range(f.shape[1]):
            g = torch.autograd.grad(f[:, k].sum(), x, create_graph=True)[0]
            f1.append(g[:, 0])
            f2.append(torch.autograd.grad(g.sum(), x, create_graph=True)[0][:, 0])
        f1, f2 = torch.stack(f1, dim=-1), torch.stack(f2, dim=-1)
        return -drift_fn(x) * f1 - 0.5 * sigma2_fn(x) * f2


def make_testfuncs(arm, hidden):
    if arm == 'pw':
        return R.SteadyTestFunctions(n_in=1, K=R.K_TEST)
    if arm == 'mlp':
        return MLPTestFunctions(K=R.K_TEST, hidden=hidden)
    raise ValueError(arm)


# ---------------------------------------------------------------------------
# Common referee: frozen plane-wave probe bank, residual in noise units
# ---------------------------------------------------------------------------
def sample_true(n, rng):
    g = np.linspace(-3, 3, 60001)
    cdf = np.cumsum(R.unnorm_density(g))
    cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, g)


def _probe_moments(probe, x_np, chunk=20000):
    s1 = torch.zeros(PROBE_K, dtype=torch.float64, device=R.device)
    s2 = torch.zeros_like(s1)
    with torch.no_grad():
        for i in range(0, len(x_np), chunk):
            x = torch.as_tensor(x_np[i:i + chunk], dtype=torch.float32, device=R.device)[:, None]
            Lf = probe.Lf(x, R.drift_fn, R.sigma2_fn).double()
            s1 += Lf.sum(dim=0)
            s2 += (Lf ** 2).sum(dim=0)
    mean = s1 / len(x_np)
    return mean, (s2 / len(x_np) - mean ** 2).clamp_min(0).sqrt()


def probe_rms_z(x_np):
    """rms over the probe bank of sqrt(N) * E_learned[Lf_k] / std_true[Lf_k].

    Returns (learned, reference), the reference being the same statistic for
    an independent exact sample of the same size (expected ~1).
    """
    g = torch.Generator().manual_seed(PROBE_SEED)
    probe = R.SteadyTestFunctions(n_in=1, K=PROBE_K).to(R.device)
    with torch.no_grad():
        probe.w.copy_(torch.randn(1, PROBE_K, generator=g) * PROBE_W_SCALE)
        probe.b.copy_(torch.rand(PROBE_K, generator=g) * 2 * np.pi)
    rng = np.random.default_rng(PROBE_SEED)
    n = len(x_np)
    _, std_true = _probe_moments(probe, sample_true(n, rng))
    scale = np.sqrt(n) / std_true
    z_learned = _probe_moments(probe, x_np)[0] * scale
    z_ref = _probe_moments(probe, sample_true(n, rng))[0] * scale
    return float((z_learned ** 2).mean().sqrt()), float((z_ref ** 2).mean().sqrt())


# ---------------------------------------------------------------------------
# One training run: identical loop to muon_only.train, test-function class swapped
# ---------------------------------------------------------------------------
def train(arm, seed, hidden, test_opt_name, lr_test, tag):
    torch.manual_seed(seed)
    np.random.seed(seed)
    pf = R.SteadyPushforward(d_base=R.D_BASE, n_out=1, hidden=R.HIDDEN).to(R.device)
    tf = make_testfuncs(arm, hidden).to(R.device)
    gen_opt = R.make_gen_opt('Muon', pf, MUON_LR)
    test_opt = R.make_test_opt(test_opt_name, tf.parameters(), lr_test)

    cur = [R.SIGMA2]

    def sigma2_cur(x):
        return cur[0] * torch.ones(x.shape[0], 1, device=x.device)

    def loss_fn():
        return R.steady_loss(pf, tf, M=R.BATCH_M, drift_fn=R.drift_fn,
                             sigma2_fn=sigma2_cur,
                             base_sampler=R.gaussian_base_sampler, device=R.device)

    n_epochs = R.N_EPOCHS
    losses = np.empty(n_epochs)
    sigmas = np.empty(n_epochs)
    if R.device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for epoch in range(n_epochs):
        cur[0] = R.sigma2_at_epoch(epoch, n_epochs, anneal=True)
        sigmas[epoch] = cur[0]
        if epoch % R.ADV_EVERY == 0 and epoch > 0:
            for _ in range(R.ADV_STEPS):
                l, _ = loss_fn()
                test_opt.zero_grad(); (-l).backward(); test_opt.step()
        l, _ = loss_fn()
        gen_opt.zero_grad(); l.backward(); gen_opt.step()
        losses[epoch] = l.item()
        if epoch % 2000 == 0 or epoch == n_epochs - 1:
            print(f"    epoch {epoch:6d}/{n_epochs}  sigma^2={cur[0]:.3f}  "
                  f"loss={losses[epoch]:.4e}", flush=True)
    if R.device.type == 'cuda':
        torch.cuda.synchronize()
    dt = time.time() - t0

    with torch.no_grad():
        x = pf(torch.randn(N_SAMPLES, R.D_BASE, device=R.device)).cpu().numpy().ravel()

    key = f"{arm}{tag}_seed{seed}"
    np.savez_compressed(os.path.join(OUTDIR, key + '.npz'), x=x.astype(np.float32),
                        loss=losses[::25].astype(np.float32),
                        sigma2=sigmas[::25].astype(np.float32))

    z_learned, z_ref = probe_rms_z(x)
    rec = {'arm': arm, 'seed': seed, 'test_opt': test_opt_name, 'lr_test': lr_test,
           'hidden': list(hidden) if arm == 'mlp' else None,
           'n_test_params': sum(p.numel() for p in tf.parameters()),
           'train_time_s': dt, 'own_bank_final_loss': float(losses[-500:].mean()),
           'probe_rms_z': z_learned, 'probe_rms_z_exact_sample': z_ref,
           **metrics(x)}
    allrec = json.load(open(RESULTS_JSON)) if os.path.exists(RESULTS_JSON) else {}
    allrec[key] = rec
    with open(RESULTS_JSON, 'w') as fh:
        json.dump(allrec, fh, indent=2)
    print(f"  [{key}] {dt:.0f}s  Hellinger={rec['hellinger']:.3f}  TV={rec['tv']:.3f}  "
          f"P(x>0)={rec['frac_right']:.3f}  bridge x{rec['bridge_ratio']:.2f}  "
          f"probe rms z={z_learned:.1f} (exact sample {z_ref:.1f})")
    return rec


# ---------------------------------------------------------------------------
# --check: forward-propagated derivatives vs autograd, and per-step cost
# ---------------------------------------------------------------------------
def check():
    torch.manual_seed(0)
    tf = MLPTestFunctions(K=R.K_TEST).double()
    x = (torch.randn(64, 1, dtype=torch.float64) * 1.2).requires_grad_(True)
    s2 = lambda x: 0.36 * torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
    a, b = tf.Lf(x, R.drift_fn, s2), tf.Lf_autograd(x, R.drift_fn, s2)
    print(f"  max |Lf_forward - Lf_autograd| = {(a - b).abs().max().item():.2e}  "
          f"(max |Lf| = {b.abs().max().item():.2e})")

    def step_time(tf, lf_name, n):
        pf = R.SteadyPushforward(d_base=R.D_BASE, n_out=1, hidden=R.HIDDEN).to(R.device)
        params = list(pf.parameters()) + list(tf.parameters())
        t = []
        for i in range(n + 2):
            if R.device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            x = pf(torch.randn(R.BATCH_M, R.D_BASE, device=R.device))
            loss = (getattr(tf, lf_name)(x, R.drift_fn, R.sigma2_fn).mean(dim=0) ** 2).mean()
            # allow_unused: the MLP's output bias drops out of Lf (L of a constant is 0)
            torch.autograd.grad(loss, params, allow_unused=True)
            if R.device.type == 'cuda':
                torch.cuda.synchronize()
            t.append(time.time() - t0)
        return 1e3 * float(np.median(t[2:]))

    print(f"  per-step cost on {R.device} (M={R.BATCH_M}, K={R.K_TEST}):")
    t_pw = step_time(make_testfuncs('pw', None).to(R.device), 'Lf', 50)
    t_fw = step_time(MLPTestFunctions(K=R.K_TEST).to(R.device), 'Lf', 50)
    t_ag = step_time(MLPTestFunctions(K=R.K_TEST).to(R.device), 'Lf_autograd', 3)
    print(f"    plane wave, closed form        {t_pw:8.2f} ms")
    print(f"    MLP, forward-propagated        {t_fw:8.2f} ms  ({t_fw / t_pw:.1f}x)")
    print(f"    MLP, K-fold double backward    {t_ag:8.2f} ms  ({t_ag / t_pw:.0f}x)")


# ---------------------------------------------------------------------------
# Figure: (a) plane-wave density, (b) MLP density, (c) losses + schedule
# ---------------------------------------------------------------------------
def plot(seed, tag):
    allrec = json.load(open(RESULTS_JSON))
    grid = np.linspace(-2, 2, 800)
    pt = R.true_pdf(grid)
    arms = [('pw', '', 'plane-wave test functions', '#7ba7d1'),
            ('mlp', tag, 'depth-3 MLP test functions', '#f0a35e')]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, (arm, tg, title, col) in zip(axes[:2], arms):
        key = f"{arm}{tg}_seed{seed}"
        d, m = np.load(os.path.join(OUTDIR, key + '.npz')), allrec[key]
        ax.hist(d['x'], bins=140, range=(-2, 2), density=True, color=col, alpha=0.8,
                edgecolor='none', label='learned samples')
        ax.plot(grid, pt, 'k-', lw=2.2, label='analytical Boltzmann')
        ax.set_xlim(-2, 2); ax.set_xlabel('$x$'); ax.set_ylabel('density')
        ax.set_title(f"{title}\nHellinger {m['hellinger']:.3f}, TV {m['tv']:.3f}, "
                     f"P(x>0) {m['frac_right']:.3f}, {m['train_time_s']:.0f} s", fontsize=10.5)
        ax.legend(fontsize=9, loc='upper center'); ax.grid(alpha=0.25)
        ep = np.arange(len(d['loss'])) * 25
        axes[2].semilogy(ep, d['loss'], color=col, lw=1.0, alpha=0.9, label=f'{title} (own bank)')
        sig = d['sigma2']
    ax = axes[2]
    ax.set_xlabel('epoch'); ax.set_ylabel('training loss'); ax.grid(alpha=0.25, which='both')
    ax.set_title('Training loss (not comparable across banks)', fontsize=10.5)
    ax2 = ax.twinx()
    ax2.plot(ep, sig, color='#d95f02', lw=1.5, ls='--', label=r'$\sigma^2$ schedule')
    ax2.set_ylabel(r'$\sigma^2$', color='#d95f02'); ax2.tick_params(axis='y', labelcolor='#d95f02')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc='upper right')
    plt.tight_layout()
    out = os.path.join(OUTDIR, f'mlp_vs_planewave_sec42{tag}_seed{seed}.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f"  wrote {out}")


def plot_reply(seed, tag, dest):
    """Reply-letter version: statistics move to the LaTeX caption, both density
    panels share one y-range, fonts sized for inclusion at \\textwidth."""
    plt.rcParams.update({'axes.labelsize': 15, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
                         'legend.fontsize': 12, 'axes.titlesize': 15})
    grid = np.linspace(-2, 2, 800)
    pt = R.true_pdf(grid)
    arms = [('pw', '', '(a) plane-wave test functions', '#7ba7d1', 'plane wave'),
            ('mlp', tag, '(b) depth-3 MLP test functions', '#f0a35e', 'depth-3 MLP')]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9))
    for ax, (arm, tg, title, col, short) in zip(axes[:2], arms):
        d = np.load(os.path.join(OUTDIR, f"{arm}{tg}_seed{seed}.npz"))
        ax.hist(d['x'], bins=140, range=(-2, 2), density=True, color=col, alpha=0.85,
                edgecolor='none', label='learned samples')
        ax.plot(grid, pt, 'k-', lw=2.2, label='analytical Boltzmann')
        ax.set_xlim(-2, 2); ax.set_ylim(0, 2.15)
        ax.set_xlabel('$x$'); ax.set_ylabel('density'); ax.set_title(title)
        ax.legend(loc='upper center', framealpha=0.95); ax.grid(alpha=0.25)
        ep = np.arange(len(d['loss'])) * 25
        axes[2].semilogy(ep, d['loss'], color=col, lw=1.0, alpha=0.95, label=short)
        sig = d['sigma2']
    ax = axes[2]
    ax.set_xlabel('epoch'); ax.set_ylabel('training loss'); ax.set_title('(c) training loss')
    ax.set_ylim(top=3e2); ax.grid(alpha=0.25, which='both')
    ax2 = ax.twinx()
    ax2.plot(ep, sig, color='#d95f02', lw=1.6, ls='--', label=r'$\sigma^2$ schedule')
    ax2.set_ylabel(r'$\sigma^2$', color='#d95f02'); ax2.tick_params(axis='y', labelcolor='#d95f02')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right', framealpha=0.95)
    plt.tight_layout()
    plt.savefig(dest, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  wrote {dest}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--reply', metavar='DEST',
                    help='write the reply-letter version of the figure (first seed) to DEST')
    ap.add_argument('--check', action='store_true', help='derivative check + step timing only')
    ap.add_argument('--plot', action='store_true', help='figure from saved runs only')
    ap.add_argument('--arms', nargs='+', default=['pw', 'mlp'], choices=['pw', 'mlp'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--hidden', nargs='+', type=int, default=list(MLP_HIDDEN))
    ap.add_argument('--test-opt', default='SGD', choices=['SGD', 'Adam'])
    ap.add_argument('--lr-test', type=float, default=R.LR_TEST_PAPER)
    ap.add_argument('--tag', default='', help='suffix for MLP-arm variants, e.g. _adam')
    args = ap.parse_args()
    print(f"device: {R.device}")
    if args.check:
        check()
    elif args.reply:
        plot_reply(args.seeds[0], args.tag, args.reply)
    elif args.plot:
        for s in args.seeds:
            plot(s, args.tag)
    else:
        for s in args.seeds:
            for arm in args.arms:
                is_mlp = arm == 'mlp'
                print(f"=== {arm}, seed {s} ===")
                train(arm, s, tuple(args.hidden),
                      args.test_opt if is_mlp else 'SGD',
                      args.lr_test if is_mlp else R.LR_TEST_PAPER,
                      args.tag if is_mlp else '')
