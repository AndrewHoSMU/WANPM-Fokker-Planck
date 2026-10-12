# Mixed test-function bank on a tilted double well

## Problem

Steady-state Fokker–Planck equation in one dimension,

$$\partial_t\rho = -\partial_x\big(b(x)\rho\big) + \tfrac12\sigma^2\partial_x^2\rho, \qquad \rho^*_t = 0 \text{ at steady state},$$

with drift and diffusion

$$b(x) = -4x^3 + 4(0.7)^2x - 0.1, \qquad \sigma^2 = 0.3,$$

corresponding to the potential $V(x) = (x^2-0.7^2)^2 + 0.1x$ (a tilted double well with minima near $x=\pm0.7$). The analytical steady state is $\rho^*(x)\propto\exp(-2V(x)/\sigma^2)$; by numerical quadrature this places **68.85%** of the mass in the left well ($x<0$) and **31.15%** in the right well ($x>0$).

## Pushforward network

$F_\theta:\mathbb R^8\to\mathbb R^1$:

$$F_\theta(r) = W_2\tanh(W_1 r + b_1) + b_2, \qquad W_1\in\mathbb R^{100\times8},\ W_2\in\mathbb R^{1\times100}$$

One hidden layer, 100 units, tanh activation — **1001 parameters** total. Base distribution $r\sim\mathcal N(0,I_8)$, an 8-dimensional standard Gaussian.

## Test functions

$K=100$ test functions of the ridge form $f^{(k)}(x)=\varphi_k(w^{(k)}x+b^{(k)})$, split into two groups:

- **75 functions** with $\varphi=\sin$, initialized $w^{(k)}\sim\mathcal N(0,1)$, $b^{(k)}\sim\mathrm{Unif}(0,2\pi)$.
- **25 functions** with $\varphi(z)=e^{-z^2/2}$ (so $\varphi'(z)=-ze^{-z^2/2}$, $\varphi''(z)=(z^2-1)e^{-z^2/2}$), initialized with $|w^{(k)}|\sim\mathrm{Unif}(2,6)$ (random sign) and centre $-b^{(k)}/w^{(k)}\sim\mathrm{Unif}(-1.5,1.5)$.

For each test function, the generator acting on it has the closed form

$$\mathcal Lf^{(k)}(x) = -b(x)\,w^{(k)}\varphi_k'(w^{(k)}x+b^{(k)}) - \tfrac12\sigma^2(w^{(k)})^2\varphi_k''(w^{(k)}x+b^{(k)}).$$

## Loss

$$\hat L(\theta,\{w^{(k)},b^{(k)}\}) = \frac1K\sum_{k=1}^K\left(\frac1M\sum_{i=1}^M \mathcal Lf^{(k)}(x_i)\right)^{\!2}, \qquad x_i = F_\theta(r_i),\ r_i\overset{\text{iid}}{\sim}\mathcal N(0,I_8),$$

with $M=1000$ samples per gradient step.

## Optimizers

- **Generator** ($\theta$): Adam, learning rate $5\times10^{-3}$, default $\beta=(0.9,0.999)$.
- **Test functions** ($\{w^{(k)},b^{(k)}\}$): SGD, learning rate $10^{-2}$, 5 ascent steps every 10 epochs (each ascent step maximizes the same $\hat L$ with respect to the test-function parameters only).

## Training schedule

$\sigma^2$ annealed from 1.5 to its target value of 0.3 over training: held at 1.5 for the first 10% of epochs, cosine-annealed down to 0.3 over the next 55%, then held at 0.3 for the remainder. Total: **16,000 epochs**.

## Seeds and evaluation

Three independent runs (seeds 0, 1, 2), each with an independently initialized network and test-function bank and independent training-batch draws. At the end of training, 200,000 samples $r\sim\mathcal N(0,I_8)$ are pushed through $F_\theta$ and compared to the analytical density $\rho^*$ on the interval $[-2.5,2.5]$ (250 histogram bins): Hellinger distance, and the ratio of the learned sample mean to the true mean ("asymmetry captured": 100% = exact recovery, 0% = a symmetric density).

## Result

| seed | learned $P(x>0)$ | Hellinger distance | asymmetry captured |
|---|---|---|---|
| 0 | 0.341 | 0.035 | 87.8% |
| 1 | 0.304 | 0.055 | 103.9% |
| 2 | 0.367 | 0.051 | 75.9% |
| **mean ± sd** | 0.337 ± 0.026 | 0.047 ± 0.009 | **89.2% ± 11.5%** |

True $P(x>0) = 0.3115$. Direction of the asymmetry (which well holds the majority of the mass) was correct on all three seeds.

Figure: `mixed_bank_result.png` — (a) learned density against the analytical steady state, seed 0; (b) training loss vs. epoch, all three seeds, with the $\sigma^2$ annealing schedule overlaid.
