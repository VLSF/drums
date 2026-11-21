# Comparative Analysis of Shape Reconstruction Approaches for Triangular Drums

## 1. Executive Summary

This report presents a systematic empirical comparison of three fundamentally different paradigms for solving the inverse spectral problem on planar triangular drums—reconstructing domain boundary geometry from the lowest 30 Dirichlet Laplacian eigenvalues:

1. **Direct Supervised Regression (`DrumTransformer`):** An amortized model trained end-to-end to map the input spectrum directly to normalized vertex coordinates under an invariant Procrustes loss.
2. **Exact Direct Optimization (Differentiating through the FEM Solver):** A classical inverse optimization approach that evaluates eigenvalues via a Finite Element Method (FEM) mesh solver at each iteration, differentiating the solver end-to-end via automatic differentiation.
3. **Surrogate-Guided Optimization (`RingTransformer`):** An optimization approach where an amortized neural surrogate forward model is trained to predict spectra from vertices, and shape recovery is performed by backpropagating spectral error into the drum coordinates. Two training regimes are evaluated under epoch-matched training (~166.7 epochs):
   - $N = 900$ samples (15,000 updates)
   - $N = 5,000$ samples (83,333 updates)

All methods are evaluated on the exact same 100 held-out test drums (samples 900–999 of `triangle_drums.npz`). For iterative optimization methods, identical starting drum configurations drawn from the training set are used.

### Comparative Benchmark Summary

| Method | Min Error (%) | Median Error (%) | Mean Error (%) | Max Error (%) | Runtime $t$ per Sample |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Regression (`DrumTransformer`)** | **0.100%** | **0.574%** | **0.663%** | **2.477%** | **$\approx 3.4 \times 10^{-6}$ s** |
| **Optimization (Exact FEM Solver)** | 0.340% | 1.626% | 11.426% | 269.291% | 12.561 s |
| **Optim. (Learned Surrogate, $N=900$)** | 0.180% | 3.679% | 5.233% | 77.987% | 0.673 s |
| **Optim. (Learned Surrogate, $N=5000$)** | 0.121% | 0.723% | 2.669% | 23.268% | 0.672 s |

*(All errors are Procrustes all-orderings relative vertex errors in %).*

---

## 2. Problem Formulation and Evaluated Approaches

### 2.1 Domain Geometry and Symmetries
A planar triangular drum is defined by its three boundary vertices $X \in \mathbb{R}^{3 \times 2}$. The Laplacian eigenvalue spectrum $\lambda = (\lambda_1, \dots, \lambda_{30}) \in \mathbb{R}^{30}$ satisfies the Helmholtz Dirichlet boundary value problem $-\Delta u = \lambda u$ with $\left.u\right|_{\partial \Omega} = 0$.

The mapping from geometry to eigenvalues possesses key physical invariances:
- Translation invariance
- Global $O(2)$ rotation and reflection invariance
- Invariance to cyclic permutation of vertex indices
- Invariance to vertex order reversal (dihedral flip)

Because a planar triangle possesses exactly 3 degrees of freedom up to rigid motions, any non-degenerate triangle can be represented by scaling and shearing a canonical reference triangle $R \in \mathbb{R}^{3 \times 2}$ via an upper-triangular matrix:

$$
A(p) = \begin{pmatrix}
e^{p_0} & p_1 \\
0 & e^{p_2}
\end{pmatrix}, \quad p = (p_0, p_1, p_2) \in \mathbb{R}^3
$$

---

### 2.2 Approach 1: Supervised Regression (`DrumTransformer`)
The `DrumTransformer` architecture uses a sequence of transformer encoder layers with multi-head attention to map the raw spectrum $\lambda \in \mathbb{R}^{30}$ directly to predicted vertices $\hat{X} \in \mathbb{R}^{3 \times 2}$.

- **Architecture:** $L = 5$ layers, hidden width $N_f = 128$, 4 attention heads, MLP expansion ratio 4 (996,226 parameters).
- **Target Normalization:** Vertices are centered and scaled by the maximum vertex norm in the training set.
- **Training Protocol:** Trained on the 900 training drums for 5,000 updates with batch size 64 using AdamW ($\eta = 10^{-3}$) with a 500-step linear warmup and cosine decay.
- **Loss Function:** Procrustes all-orderings relative error, directly optimizing the target metric:

$$
\mathcal{L}_{\text{reg}}(\theta) = \frac{1}{B} \sum_{b=1}^B \min_{\pi \in \Pi} \text{rel. error}\left(X^{(b)}, \pi(f_{\theta}(\lambda^{(b)}))\right)
$$

---

### 2.3 Approach 2: Exact Direct FEM Optimization
Rather than learning an inverse map, this approach optimizes the 3 affine parameters $p \in \mathbb{R}^3$ directly for each target spectrum $\lambda^{\ast}$ by differentiating through the forward FEM solver:

1. Mesh coordinates are deformed: $X_{\text{mesh}}(p) = X_{\text{ref}} A(p)$.
2. Element stiffness matrices $K^e(p)$ and mass matrices $M^e(p)$ are assembled via `jax.vmap`.
3. Boundary conditions are applied by slicing free interior nodes: $K_{\text{free}}, M_{\text{free}}$.
4. Generalized eigenvalues are computed via Cholesky decomposition of $M_{\text{free}}$ ($M_{\text{free}} = L L^\top$) followed by symmetric tridiagonalization / divide-and-conquer on $C = L^{-1} K_{\text{free}} L^{-\top}$:

$$
\lambda(p) = \text{eigvalsh}(C(p))
$$

5. The objective is minimized via Adam ($\eta = 0.1$, 500 steps):

$$
\min_{p \in \mathbb{R}^3} \mathcal{L}_{\text{FEM}}(p) = \sum_{k=1}^{10} \left( \ln \lambda_k(p) - \ln \lambda_k^{\ast} \right)^2
$$

---

### 2.4 Approach 3: Neural Surrogate Optimization (`RingTransformer`)
This approach replaces the costly FEM solver with a fast, differentiable neural forward surrogate:

- **Surrogate Architecture:** The `RingTransformer` ($L = 4$, $N_f = 128$, 4 heads) operates on the Gram matrix $G = (X - \bar{X})(X - \bar{X})^\top$ and circulant ring tokens to strictly enforce translation, rotation, and cyclic shift invariance. Dihedral flip invariance is achieved by averaging forward and reversed vertex passes via `forward_flip`.
- **Surrogate Training:** Trained to predict spectra by minimizing log-spectral MSE. To establish an equalized comparison, both models are trained for $\sim 166.7$ effective epochs:
  - **Small Surrogate ($N=900$):** 15,000 updates at batch size 10 (166.7 epochs), final training loss = 0.0564.
  - **Extended Surrogate ($N=5000$):** 83,333 updates at batch size 10 (166.7 epochs), final training loss = 0.0015.
- **Inverse Optimization:** The surrogate weights $\theta^{\ast}$ are frozen, and drum vertex coordinates $X \in \mathbb{R}^{3 \times 2}$ are optimized directly via Adam ($\eta = 0.1$, 1,000 steps):

$$
\min_{X \in \mathbb{R}^{3 \times 2}} \mathcal{L}_{\text{surr}}(X) = \sum_{k=1}^{30} \left( \ln \lambda_k^{\ast} - \ln \hat{\lambda}_k(X; \theta^{\ast}) \right)^2
$$

---

## 3. Experimental Setup & Evaluation Protocol

### 3.1 Evaluation Metric
All methods are scored using the exact Procrustes all-orderings relative error (in percent), evaluating over all 6 dihedral/cyclic vertex permutations $\Pi$:

$$
\text{Error}(X_{\text{pred}}, X_{\text{target}}) = 100\% \times \min_{\pi \in \Pi} \frac{1}{3} \sum_{i=1}^3 \frac{\|X_{\text{target}, i} - \pi(X_{\text{pred}})_i\|}{\|X_{\text{target}, i}\|}
$$

### 3.2 Evaluation Set
- **Test Set:** Samples 900 through 999 (100 triangles) of `triangle_drums.npz`.
- **Initial Starting Configurations:** For the iterative optimization methods (Exact FEM and both Surrogate variants), each test drum is initialized from a deterministically sampled random drum from the 900 training drums (`seed=42`). Both surrogate variants and the exact FEM optimizer share the exact same paired start configurations.

---

## 4. In-Depth Results and Comparison

### 4.1 Statistical Distribution of Errors

```
Relative Shape Error (%) Distribution (Box Plot Representation)
-----------------------------------------------------------------------------------------
Method                  Min     25%     Median    Mean     75%     Max
-----------------------------------------------------------------------------------------
Regression              0.100%  0.372%  0.574%    0.663%   0.844%    2.477%
Optim. (Surr., N=5000)  0.121%  0.421%  0.723%    2.669%   1.614%   23.268%
Optimization (Exact)    0.340%  0.892%  1.626%   11.426%   4.041%  269.291%
Optim. (Surr., N=900)   0.180%  1.455%  3.679%    5.233%   6.210%   77.987%
-----------------------------------------------------------------------------------------
```

### 4.2 Runtime and Computational Efficiency

- **Regression (`DrumTransformer`):** A single vectorized batch evaluation on GPU takes $0.34$ milliseconds for all 100 samples, or **$\approx 3.4\,\mu\text{s}$ per sample**.
- **Surrogate Optimization (`RingTransformer`):** 1,000 gradient updates on GPU take **0.672 seconds per sample** (~$18.7\times$ faster than the exact FEM solver).
- **Exact Direct Optimization (FEM):** 500 gradient updates on GPU take **12.561 seconds per sample** (~20.9 minutes for 100 samples).

---

## 5. Technical Insights & Trade-Off Analysis

### 5.1 Why Regression Achieves the Lowest Overall Error and Extreme Speed
1. **Direct Global Mapping vs. Local Search:** Supervised regression learns a direct global mapping from spectral invariants to coordinates across the training manifold. It does not suffer from local minima, saddle points, or unfavorable initial configurations.
2. **Absence of Optimization Sinks:** While iterative methods can stall in plateaus when starting from distant random drums, regression maps any valid spectrum directly to the neighborhood of the true geometry in a single forward pass.
3. **Latency Advantage:** Requiring only a single feed-forward evaluation, regression is over **195,000$\times$ faster** than surrogate optimization and **3.6 million$\times$ faster** than direct FEM optimization.

### 5.2 Performance of Exact Direct FEM Optimization
1. **Unbiased Physical Gradients:** When the optimization landscape is favorable, exact FEM differentiation achieves high accuracy (minimum error 0.340%, median error 1.626%).
2. **Sensitivity to Initial Starting Drums:** Because the optimization starts from distant random training drums within a fixed budget of 500 steps, a few outlier drums encounter steep non-convex valleys or saddle points, causing the maximum error to reach 269% and pulling the mean to 11.43%.
3. **Heavy Computational Burden:** Computing Cholesky factorizations ($1118 \times 1118$) and symmetric eigenvalue decompositions at each iteration makes full-dataset evaluation computationally demanding (over 20 minutes for 100 drums).

### 5.3 Neural Surrogate Optimization: Data Scaling and Energy Landscape
1. **Dramatic Gains from Dataset Scaling ($N=900 \to N=5000$):**
   When the surrogate's training dataset is expanded from 900 to 5,000 samples under equalized epoch exposure:
   - Median error drops by **80.3%** (from 3.679% down to **0.723%**), closely approaching regression accuracy (0.574%).
   - Mean error is halved (from 5.233% to 2.669%).
   - Maximum error drops from 77.99% to 23.27%.
2. **Smoothing of the Learned Energy Surface:**
   With 5,000 training examples, the surrogate learns an accurate, globally consistent spectral surface. Spurious local minima are suppressed, allowing gradient descent to navigate reliably from arbitrary initial drums to the true geometry.
3. **Speed–Accuracy Trade-Off:**
   Surrogate optimization provides the iterative adaptability of gradient-based optimization while maintaining sub-second execution (0.67s per sample), bridging the gap between slow FEM solvers and purely predictive regression models.

---

## 6. Recommendations and Conclusion

- **For Fast, High-Throughput Inverse Problems:** Supervised regression (`DrumTransformer`) is the clear choice, delivering superior accuracy (median 0.57%) and instantaneous inference ($\mu\text{s}$ scale).
- **For Iterative Inversion and Refinement:** If an iterative optimization formulation is required (e.g. for downstream constrained optimization or domain deformation), a well-trained neural surrogate on an extended dataset ($N=5000$) provides near-regression accuracy ($0.72\%$ median error) with a $19\times$ speedup over full FEM differentiation.
- **For Exact Physical Validation:** Direct FEM optimization remains valuable as a final polishing step or for generating high-precision ground truth, provided it is initialized close to the target (such as initializing from the regression prediction).
