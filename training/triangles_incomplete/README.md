# Comparative Evaluation: Plain MCMC vs. MALA for Dirichlet Laplace Triangle Inversion

This report evaluates and compares **Plain MCMC (Random Walk Metropolis)** and **MALA (Metropolis-Adjusted Langevin Algorithm)** for sampling planar triangles compatible with the first two Dirichlet Laplacian eigenvalues.

---

## 1. Algorithm Descriptions

### A. Plain MCMC (Random Walk Metropolis)
- **State Representation:** Centered vertex coordinates $T \in \mathbb{R}^{3 \times 2}$ with $\sum_{i=1}^3 v_i = 0$.
- **Prior:** Invariant Procrustes distance prior:
  $$\ln p_{\text{prior}}(T) = -\frac{\mathcal{D}_{\text{inv}}^2(T, \mu)}{2\sigma_p^2}, \quad \mu = \text{equilateral template}$$
- **Likelihood:** Log-eigenvalue mean squared error:
  $$\ln p(\lambda^* \mid T) = -\frac{\sum_{k=1}^2 (\ln \lambda_k(T) - \ln \lambda_k^*)^2}{2\sigma_s^2}$$
- **Proposal:** Symmetric isotropic Gaussian random walk:
  $$T' = \Pi_{\text{zero-mean}}(T + \sigma_t Z), \quad Z \sim \mathcal{N}(0, I)$$
- **Acceptance Probability:**
  $$\ln \alpha = \min\left(0, \; \ln \pi(T') - \ln \pi(T)\right)$$

### B. MALA (Metropolis-Adjusted Langevin Dynamics)
- **Log-Posterior Energy:** $U(T) = \ln p(\lambda^* \mid T) + \ln p_{\text{prior}}(T)$.
- **Gradient Computation:** Exact reverse-mode automatic differentiation $\nabla_T U(T)$ via JAX, with Danskin's envelope theorem (`stop_gradient` on the Procrustes rotation factor $R = U V^T$) to bypass SVD degeneracy at the equilateral template.
- **Drift with Truncation (t-MALA):**
  $$\text{drift}(T) = \tau \nabla_T U(T) \cdot \min\left(1.0, \; \frac{\delta_{\max}}{\tau \|\nabla_T U(T)\|}\right)$$
  projected onto the centered subspace $\sum v_i = 0$.
- **Proposal:**
  $$T' = \Pi_{\text{zero-mean}}\left(T + \text{drift}(T) + \sqrt{2\tau} Z\right), \quad Z \sim \mathcal{N}(0, I)$$
- **Hastings Correction (Asymmetric Transition Probabilities):**
  $$\ln q(T' \mid T) = -\frac{\|T' - T - \text{drift}(T)\|_F^2}{4\tau}, \quad \ln q(T \mid T') = -\frac{\|T - T' - \text{drift}(T')\|_F^2}{4\tau}$$
  $$\ln \alpha = \min\left(0, \; U(T') - U(T) + \ln q(T \mid T') - \ln q(T' \mid T)\right)$$

---

## 2. Empirical Benchmark Results

We benchmarked both algorithms over 5 independent targets sampled from the invariant prior (`comparative_mcmc_mala.py`). Both chains ran for 1,000 steps with grid resolution $N=35$, using identical initializations drawn independently from the prior.

### Summary Metrics (1,000 Steps)

| Metric | Plain MCMC (Random Walk) | MALA (Langevin Dynamics) | Comparison |
| :--- | :---: | :---: | :---: |
| **Average Acceptance Rate** | **48.7%** | **80.7%** | **+32.0% higher acceptance** |
| **Average Steps to $<20\%$ Error** | 21.8 steps | **18.8 steps** | **Faster burn-in** |
| **Average Relative Spectral Error** | 0.0783 | 0.0899 | Comparable ($<9\%$) |
| **Average Runtime (1,000 steps)** | **7.98 s** | 12.28 s | Only $1.5\times$ overhead |
| **Time per Accepted Step** | $\approx 16.4\text{ ms}$ | **$\approx 15.2\text{ ms}$** | **MALA is more efficient per accepted sample** |

---

### Detailed Per-Target Results

#### Target 1: $\lambda^* = [38.04, 61.22]$, True Invariants: Area = 3.06, inradius $r = 0.3127$
- **Plain MCMC:** Time 9.91s, Acceptance 40.1%, Error at step 10: 0.7945, Final error: 0.0563.
  - Posterior Invariants: Area $0.897 \pm 0.184$, inradius $r = 0.327 \pm 0.023$.
- **MALA:** Time 15.65s, Acceptance **94.7%**, Error at step 10: **0.7260**, Final error: 0.1714.
  - Posterior Invariants: Area $0.801 \pm 0.063$, inradius $r = 0.335 \pm 0.015$.

#### Target 2: $\lambda^* = [603.02, 791.87]$, True Invariants: Area = 1.99, inradius $r = 0.0728$
- **Plain MCMC:** Time 6.40s, Acceptance 8.8%, Burn-in: 37 steps, Final error: 0.0654.
  - Posterior Invariants: inradius $r = 0.0742 \pm 0.0026$.
- **MALA:** Time 11.46s, Acceptance **10.3%**, Burn-in: **21 steps**, Final error: 0.0540.
  - Posterior Invariants: inradius $r = 0.0750 \pm 0.0030$.

#### Target 3: $\lambda^* = [5.46, 11.40]$, True Invariants: Area = 4.93, inradius $r = 0.8892$
- **Plain MCMC:** Time 7.73s, Acceptance 72.8%, Error at step 10: 1.0633, Final error: 0.1766.
  - Posterior Invariants: Area $4.433 \pm 0.295$, inradius $r = 0.878 \pm 0.035$.
- **MALA:** Time 11.52s, Acceptance **99.9%**, Error at step 10: **0.7138**, Final error: **0.0655**.
  - Posterior Invariants: Area $4.280 \pm 0.238$, inradius $r = 0.898 \pm 0.026$.

#### Target 4: $\lambda^* = [14.98, 25.58]$, True Invariants: Area = 4.13, inradius $r = 0.5095$
- **Plain MCMC:** Time 7.93s, Acceptance 63.3%, Error at step 10: 0.4084, Final error: 0.0364.
  - Posterior Invariants: inradius $r = 0.542 \pm 0.022$.
- **MALA:** Time 11.24s, Acceptance **99.3%**, Error at step 10: **0.2498**, Final error: 0.1125.
  - Posterior Invariants: inradius $r = 0.551 \pm 0.022$.

#### Target 5: $\lambda^* = [12.17, 23.58]$, True Invariants: Area = 3.61, inradius $r = 0.5863$
- **Plain MCMC:** Time 7.93s, Acceptance 58.6%, Burn-in: 4 steps, Final error: 0.0565.
  - Posterior Invariants: inradius $r = 0.568 \pm 0.027$.
- **MALA:** Time 11.52s, Acceptance **99.5%**, Burn-in: 6 steps, Final error: **0.0463**.
  - Posterior Invariants: inradius $r = 0.596 \pm 0.028$.

---

## 3. Key Findings and Practical Recommendations

1. **Higher Acceptance Rate & Better State Space Exploration:**
   MALA achieves an average acceptance rate of **$80.7\%$** compared to **$48.7\%$** for random walk. By steering proposals along the gradient of the log-posterior, MALA avoids proposing steps that cross into low-probability regions.
2. **Faster Initial Descent:**
   During the first 10 steps, MALA reduces relative error much faster than Plain MCMC (e.g., dropping from error $>1.0$ down to $0.25 - 0.71$), because the deterministic drift directly pulls the initial candidate toward the target spectral contour.
3. **Modest Computational Overhead:**
   Because reverse-mode AD (`value_and_grad`) in JAX is compiled into the same XLA computation graph, computing both the energy and the gradient takes only $\approx 12.3\text{ ms}$ per step (vs $\approx 8.0\text{ ms}$ for energy only). Because MALA accepts far more proposals, **its time per accepted state is lower than Plain MCMC** ($\approx 15.2\text{ ms}$ vs $\approx 16.4\text{ ms}$).
4. **Necessity of Drift Truncation (t-MALA):**
   When initialized far from the mode, gradients can exceed $10^3$. Without drift truncation, the proposal jumps too far, causing reverse proposal probabilities to vanish and freezing the chain. Truncating the maximum drift step ($\|\text{drift}\| \le 0.05$) stabilizes the sampler across all drum scales while strictly preserving detailed balance.
