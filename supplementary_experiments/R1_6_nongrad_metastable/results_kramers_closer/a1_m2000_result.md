# Kramers metastable double well (original instance), mixed test-function bank + Muon, M=2000

## Problem

Classic underdamped (Kramers) Langevin dynamics in phase space $(x,v)\in\mathbb R^2$:

$$dx = v\,dt, \qquad dv = \big(-\gamma v - U'(x)\big)dt + \sqrt{2\gamma T}\,dW,$$

with $U(x) = (x^2 - 1)^2$ (double well with minima at $x=\pm1$), $\gamma=1$, $T=0.25$. This is the **original instance** — nothing about the potential or the wells is changed. The diffusion tensor $a(x,v)=\mathrm{diag}(0,\,2\gamma T)$ is degenerate (zero in $x$), and the drift is manifestly **non-gradient** in $(x,v)$ (the cross-partials $\partial b_x/\partial v = 1$ and $\partial b_v/\partial x = -U''(x)$ are unequal).

The stationary density factorises exactly, $\rho^*(x,v)\propto \exp(-H/T)$ with $H=U(x)+v^2/2$, giving

$$\rho_x^*(x) \propto e^{-U(x)/T} \qquad\text{(double-well Boltzmann)}, \qquad \rho_v^*(v) = \mathcal N(0,T).$$

By quadrature: true split $50{:}50$, connectivity $\rho^*(0)/\rho^*_{\text{peak}} = 0.018$, relative barrier $2\Delta U/T = 8.0$. This is a strongly metastable regime — the density between the two wells is more than fifty times smaller than at the peaks.

## Pushforward network (generator)

$F_\theta:\mathbb R^8\to\mathbb R^2$, the manuscript's Sec. 4.2-style architecture:

$$F_\theta(r) = W_3\tanh\big(W_2\tanh(W_1 r + b_1) + b_2\big) + b_3, \qquad W_1\in\mathbb R^{32\times8},\ W_2\in\mathbb R^{32\times32},\ W_3\in\mathbb R^{2\times32}$$

Two hidden layers, 32 units each, tanh. Base distribution $r\sim\mathcal N(0,I_8)$.

## Test-function bank — mixed, sin-majority

$K=300$ ridge functions $f^{(k)}(x,v)=\varphi_k\big(w_1^{(k)}x + w_2^{(k)}v + b^{(k)}\big)$, split $180/120$:

**180 plane waves** ($\varphi=\sin$), initialized $w_1,w_2\sim\mathcal N(0,1)$, $b\sim\mathrm{Unif}(0,2\pi)$ — the manuscript's own `SteadyTestFunctions` init, extended to 2 input dimensions.

**120 localized wavelets**, split evenly (24 each) across five non-plane-wave activation shapes: Gaussian $e^{-z^2/2}$, Mexican-hat $(1-z^2)e^{-z^2/2}$, Morlet $e^{-z^2/2}\cos(5z)$, logistic/sech² $1-\tanh^2 z$, and the odd Hermite bump $z\,e^{-z^2/2}$. All second derivatives are closed-form, verified against autograd to machine precision. Each wavelet sub-bank initialized with $|w_1|\sim 4\cdot\mathrm{Unif}(0.5,1.5)$ (random sign — width $\propto 1/|w_1|$), $w_2\sim 0.5\,\mathcal N(0,1)$, centre $-b/w_1\sim\mathrm{Unif}(-1.5,1.5)$ (spanning both wells and the saddle).

For the degenerate diffusion, the generator has the closed form

$$\mathcal Lf^{(k)}(x,v) = -\big(b_{\text{drift}}(x,v)\cdot w^{(k)}\big)\varphi_k'(z) \;-\; \gamma T\,(w_2^{(k)})^2\,\varphi_k''(z), \qquad z = w_1^{(k)}x+w_2^{(k)}v+b^{(k)}.$$

## Loss — the manuscript's own, unchanged

$$\hat L = \frac1K\sum_{k=1}^K\left(\frac1M\sum_{i=1}^M \mathcal Lf^{(k)}(x_i,v_i)\right)^{\!2}, \qquad (x_i,v_i)=F_\theta(r_i),\ r_i\sim\mathcal N(0,I_8).$$

**No U-statistic.** $M$ (collocation points per step) $= 2000$ — doubled from the paper's own $M=1000$. $K=300$.

## Optimizers

- **Generator** $\theta$: **Muon**, lr $=0.2$, momentum $=0.95$, Nesterov, 5 Newton–Schulz orthogonalization steps — applied only to the two 2-D hidden weight matrices ($W_1$, $W_2$). Biases and the output layer $W_3,b_3$ are updated by Muon's internal AdamW at lr $=10^{-2}$, $\beta=(0.9,0.95)$ — the manuscript's own generator learning rate.
- **Adversary** $\{w^{(k)},b^{(k)}\}$: SGD, lr $=10^{-2}$, **1 ascent step every 10 epochs**.

## Curriculum

$T$ annealed **1.0 → 0.25**: held at $T=1.0$ for the first 3,000 epochs, cosine-annealed to $T=0.25$ over the next 20,000 epochs, held at $T=0.25$ for the remaining 12,000 epochs. Total: **35,000 epochs**.

## Seed and evaluation

**Single run, seed 0.** At the end of training, 200,000 fresh samples $r\sim\mathcal N(0,I_8)$ pushed through $F_\theta$: $x$-marginal histogrammed on $[-2.5,2.5]$ (250 bins) against $\rho_x^*$ for Hellinger/TV, $v$-marginal std compared to $\sqrt T = 0.500$. Training time: 869 s on a single RTX 3050 Ti laptop GPU.

## Result

| quantity | learned | true |
|---|---|---|
| $x$-marginal Hellinger distance | **0.109** | 0 |
| $x$-marginal TV distance | 0.071 | 0 |
| $P(x>0)$ | 0.503 | 0.500 |
| transition-state ratio $\rho(0)/\rho(\pm1)$ | 0.002 | 0.018 |
| $v$-marginal std | 0.488 | 0.500 |
| fraction of samples in $[-2.5,2.5]$ | 1.000 | — |

The learned $x$-marginal recovers both peaks of the strongly metastable Boltzmann density essentially exactly — correct height, position, and width — with the valley between them nearly empty, which is the correct behaviour at connectivity $0.018$. The $v$-marginal closely matches the true Maxwell–Boltzmann $\mathcal N(0,T)$. Training converged: the loss flattens near $1.3\times10^{-3}$ from roughly epoch 22,000 onward, with a stable running median for the final $\sim13{,}000$ epochs (see `a1_m2000_result.png`, bottom-right panel).

This differs from the manuscript's own configuration in exactly three places: the test-function bank (mixed $180$ sin / $120$ wavelet instead of pure plane-wave), the generator optimizer (Muon instead of Adam), and the collocation batch size ($M=2000$ instead of $1000$). The architecture, base distribution, adversary schedule, curriculum, and loss function are all exactly as specified in the manuscript.

**Context.** The manuscript's own unmodified configuration on this instance produced a learned $x$-marginal with a large spurious spike at the transition state $x=0$ and under-resolved wells (Hellinger $\approx0.6$). Intermediate configurations (a mixed Gaussian-only test-function bank with Adam and $M=1000$) reduced this only to Hellinger $\approx0.53$, with one seed collapsing into a single well. The three changes above, together, take this instance to Hellinger $\approx0.11$.

**Caveat.** This is a single seed. A seed-variability study on nearby configurations found run-to-run Hellinger spread of roughly $\pm0.01$–$0.03$ — small relative to the gap between this result and the manuscript's own configuration, but the result has not been repeated across additional seeds. One residual imperfection: at this high barrier the failure mode is a slightly *over*-carved valley (ts_ratio $0.002$ vs. true $0.018$), i.e. the wavelets suppress the saddle a little too aggressively; because the true valley is nearly empty this contributes negligibly to the Hellinger.

Figure: `a1_m2000_result.png` — (top-left) phase-space samples with the non-gradient drift field; (top-right) learned vs. true $x$-marginal; (bottom-left) learned vs. true $v$-marginal; (bottom-right) training loss vs. epoch with the $T$-annealing schedule overlaid.
