"""
Wavelet-activation test functions for the WANPM weak form.

The manuscript (main.tex, eq. following l.136) defines the test function as a
RIDGE function of a learnable affine form,

    f(x) = phi(w . x + b),      phi : R -> R  a C^2 activation,

for which the Fokker-Planck generator has the closed form

    Lf(x) = -(b_drift(x) . w) phi'(z)  -  (1/2) (sum_ij a_ij w_i w_j) phi''(z),
    z = w . x + b.

Only phi changes here; the parameterisation, the learnable parameters (w, b),
and the O(1) cost of the derivatives are exactly as in the paper. Every phi
below is C-infinity, so the C^2 requirement is satisfied with room to spare.

WHY A WAVELET.  In ridge form a localized phi is automatically a wavelet
family: f is centred at x = -b/w with width proportional to 1/|w|, so the
existing (w, b) already parameterise translation and dilation -- no extra
parameters are needed. residual_theory.py shows that for ANY C^2 test function
the weak-form residual of a wrong well split is

    R(lambda) = (a/2) rho*(0) f'(0) (lambda - lambda*) / [lambda*(1-lambda*)],

i.e. the signal depends only on the slope at the transition state, while the
noise Var_rho(Lf) is paid wherever rho has mass -- inside the wells. A plane
wave oscillates across the whole domain and so pays full variance in the wells
while collecting signal only at x = 0; a wavelet centred on the transition
state should collect the same signal for far less variance.

Activations (phi, phi', phi'' all closed form):
    sin      sin z                        the manuscript's plane wave
    gauss    exp(-z^2/2)                  localized, non-oscillatory
    mexhat   (1 - z^2) exp(-z^2/2)        Ricker / Mexican hat
    morlet   exp(-z^2/2) cos(w0 z)        windowed plane wave (default w0 = 5)
"""
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# activations: return (phi, phi', phi'')
# ---------------------------------------------------------------------------
def _sin(z, w0=None):
    s, c = torch.sin(z), torch.cos(z)
    return s, c, -s


def _gauss(z, w0=None):
    g = torch.exp(-z ** 2 / 2)
    return g, -z * g, (z ** 2 - 1) * g


def _mexhat(z, w0=None):
    g = torch.exp(-z ** 2 / 2)
    return (1 - z ** 2) * g, (z ** 3 - 3 * z) * g, (-z ** 4 + 6 * z ** 2 - 3) * g


def _morlet(z, w0=5.0):
    g = torch.exp(-z ** 2 / 2)
    c, s = torch.cos(w0 * z), torch.sin(w0 * z)
    return (g * c,
            g * (-z * c - w0 * s),
            g * ((z ** 2 - 1 - w0 ** 2) * c + 2 * w0 * z * s))


ACTIVATIONS = {'sin': _sin, 'gauss': _gauss, 'mexhat': _mexhat, 'morlet': _morlet}


class MixedTestFunctions(nn.Module):
    """A bank that is part global plane wave, part localized wavelet.

    WHY.  Training with a purely localized bank fails in a specific and
    instructive way: a localized test function only constrains the density
    WHERE IT HAS SUPPORT, so the generator can drive the loss to ~0 by moving
    its mass out of the wavelets' reach entirely. Empirically (mexhat, morlet)
    only 2-6% of samples stayed in [-2.5, 2.5] and the learned std blew up to
    9-39 against a true 0.97. Plane waves do not have this failure mode because
    they are global.

    So use both: `frac_wave` of the K functions are plane waves, which keep the
    density honest everywhere, and the rest are wavelets, which supply the
    transition-state sensitivity that wavelet_snr.py shows is worth 35-70x in
    detection SNR. Both halves are the manuscript's own ridge form
    f = phi(w x + b) with closed-form derivatives; only phi differs.
    """

    def __init__(self, n_in, K, activation='gauss', w0=5.0, frac_wave=0.5,
                 init_scale=4.0, init_centre_spread=1.5, seed=None):
        super().__init__()
        self.n_wave = int(round(frac_wave * K))
        self.n_loc = K - self.n_wave
        self.plane = WaveletTestFunctions(n_in, self.n_wave, activation='sin',
                                          seed=seed) if self.n_wave else None
        self.local = WaveletTestFunctions(
            n_in, self.n_loc, activation=activation, w0=w0,
            init_scale=init_scale, init_centre_spread=init_centre_spread,
            seed=None if seed is None else seed + 1) if self.n_loc else None
        self.activation = f'mixed({activation},{frac_wave:g})'

    def Lf(self, x, drift_fn, sigma2_fn):
        parts = [m.Lf(x, drift_fn, sigma2_fn)
                 for m in (self.plane, self.local) if m is not None]
        return torch.cat(parts, dim=1)


class WaveletTestFunctions(nn.Module):
    """Drop-in replacement for common_wanpm.SteadyTestFunctions.

    Same interface (`w`, `b` parameters; `Lf(x, drift_fn, sigma2_fn)`), same
    isotropic-diffusion assumption a(x) = sigma2 * I, so it can be swapped in
    without touching steady_loss.

    init_scale controls the initial |w|, i.e. the initial wavelet WIDTH ~ 1/|w|.
    init_centre_spread controls where the wavelets start: b is drawn so that the
    centre -b/w is spread over roughly [-c, c]. For the plane wave ('sin') the
    manuscript's own initialisation is used instead (w ~ N(0,1), b ~ U(0,2pi)),
    so `activation='sin'` reproduces SteadyTestFunctions exactly.
    """

    def __init__(self, n_in, K, activation='morlet', w0=5.0,
                 init_scale=1.0, init_centre_spread=1.5, seed=None):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; "
                             f"choose from {sorted(ACTIVATIONS)}")
        self.activation = activation
        self.w0 = w0
        self.n_in = n_in
        self.K = K
        gen = None
        if seed is not None:
            gen = torch.Generator().manual_seed(seed)

        if activation == 'sin':
            w = torch.randn(n_in, K, generator=gen)
            b = torch.rand(K, generator=gen) * 2 * np.pi
        else:
            # |w| ~ init_scale sets the width; sign is random.
            mag = init_scale * (0.5 + torch.rand(n_in, K, generator=gen))
            sign = torch.where(torch.rand(n_in, K, generator=gen) < 0.5, -1.0, 1.0)
            w = mag * sign
            # centre = -b/w spread over [-spread, spread]
            centre = (torch.rand(K, generator=gen) * 2 - 1) * init_centre_spread
            b = -(w[0] * centre) if n_in == 1 else -(w.mean(dim=0) * centre)
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(b)

    def phase(self, x):
        return x @ self.w + self.b

    def _phi(self, z):
        return ACTIVATIONS[self.activation](z, self.w0)

    def forward(self, x):
        return self._phi(self.phase(x))[0]

    def Lf(self, x, drift_fn, sigma2_fn):
        """Lf = -(b.w) phi'(z) - (1/2) (sum a_ij w_i w_j) phi''(z),
        with isotropic a(x) = sigma2(x) I so that sum_ij a_ij w_i w_j =
        sigma2 * |w|^2 -- identical in structure to the plane-wave case, only
        cos/sin replaced by phi'/phi''."""
        z = self.phase(x)                                    # (batch, K)
        _, d1, d2 = self._phi(z)
        b_dot_w = drift_fn(x) @ self.w                       # (batch, K)
        w_sq = (self.w ** 2).sum(dim=0, keepdim=True)        # (1, K)
        sigma2 = sigma2_fn(x)                                # (batch, 1)
        return -b_dot_w * d1 - 0.5 * sigma2 * w_sq * d2


# ---------------------------------------------------------------------------
# self-test: verify phi' and phi'' against autograd, and verify that
# activation='sin' reproduces the manuscript's Lf exactly.
# ---------------------------------------------------------------------------
def _selftest():
    torch.manual_seed(0)
    z = torch.randn(200, dtype=torch.float64, requires_grad=True)
    print("derivative check (analytic vs autograd), float64:")
    for name in ACTIVATIONS:
        tf = WaveletTestFunctions(1, 1, activation=name)
        f, d1, d2 = tf._phi(z)
        g1 = torch.autograd.grad(f.sum(), z, create_graph=True)[0]
        g2 = torch.autograd.grad(g1.sum(), z)[0]
        print("  %-8s max|phi'-ad| = %.3e   max|phi''-ad| = %.3e"
              % (name, (d1 - g1).abs().max().item(), (d2 - g2).abs().max().item()))

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..'))
    from common_wanpm import SteadyTestFunctions
    torch.manual_seed(7)
    ref = SteadyTestFunctions(n_in=1, K=16)
    torch.manual_seed(7)
    new = WaveletTestFunctions(1, 16, activation='sin')
    x = torch.randn(64, 1)
    drift = lambda t: -4 * t ** 3 + 4 * t - 0.1
    s2 = lambda t: 0.3 * torch.ones(t.shape[0], 1)
    d = (ref.Lf(x, drift, s2) - new.Lf(x, drift, s2)).abs().max().item()
    print("\nactivation='sin' vs manuscript SteadyTestFunctions: max|dLf| = %.3e" % d)
    assert d < 1e-6, "sin activation must reproduce the manuscript exactly"
    print("OK")


if __name__ == '__main__':
    _selftest()
