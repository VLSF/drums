# Shape Reconstruction of Triangular Domains via Surrogate Optimization

## 1. Executive Summary

This study investigates the inverse problem of recovering the shape of a planar triangular drum from its Dirichlet Laplacian spectrum using an amortized surrogate forward model. Instead of differentiating through expensive Finite Element Method (FEM) mesh assembly and eigenvalue solvers at every gradient step, we train a symmetry-preserving neural surrogate model (the **RingTransformer**) to predict spectra from polygon vertices, and subsequently invert it by backpropagating target spectral losses directly into the vertex coordinates.

We conduct experiments to evaluate how the size of the surrogate's training dataset influences downstream inverse shape recovery under **epoch-matched training** (~166.7 epochs):
- **Experiment 1 (Extended Dataset):** Surrogate trained on 5,000 samples (`triangle_drums_extended/triangle_drums.npz`) for 83,333 updates with batch size 10 (166.7 epochs).
- **Experiment 2 (Small Dataset):** Surrogate trained on the first 900 samples (`triangle_drums.npz`) for 15,000 updates with batch size 10 (166.7 epochs).

Both models are evaluated on the exact same 10 test drums randomly drawn from the held-out test split (samples 900–999 of `triangle_drums.npz`), starting from the same randomly chosen initial drums from the training set.

### Key Findings
1. **Significant Advantage of the Extended Dataset Under Matched Epochs:**
   When trained for the same number of epochs (~166.7 epochs), the surrogate trained on 5,000 samples dramatically outperforms the 900-sample surrogate in downstream shape reconstruction:
   - **Mean invariant shape error** drops from **0.0523** down to **0.0163** (a **68.8% error reduction**).
   - **Median shape error** drops from **0.0383** down to **0.0146** (a **61.9% error reduction**).
   - **Worst-case error** drops from **0.1128** down to **0.0389**.
   - Out of 10 test drums, the 5,000-sample surrogate achieved superior shape recovery on **9 out of 10 drums**, often reducing error by 70% to 86%.
2. **Surrogate Forward Accuracy:**
   The extended dataset model achieved a relative spectrum prediction error of only **0.7475%** on the 10 test drums (and **0.8035%** across all 100 test drums), compared to **1.8190%** for the 900-sample model.
3. **Data Scaling Dynamics:**
   A surrogate trained on 5,000 samples with only 15,000 updates (30 epochs) underperformed (mean error 0.1318) due to undertraining. Matching the epoch budget (83,333 updates) unlocked the full capacity of the RingTransformer, producing highly accurate gradients and smooth loss surfaces for direct inverse optimization.
4. **Computational Speedup:**
   Evaluating gradients through the neural surrogate allows 1,000 Adam optimization steps to execute in under **0.5 seconds per drum** on GPU, several orders of magnitude faster than repeatedly re-meshing and solving generalized matrix eigenvalue problems.

---

## 2. Methodology and Mathematical Formulation

### 2.1 Surrogate Architecture: RingTransformer
The forward mapping from boundary vertices $X \in \mathbb{R}^{n \times 2}$ ($n=3$ for triangles) to Dirichlet eigenvalues $\lambda \in \mathbb{R}^{30}$ must respect physical symmetries:
- Translation invariance
- Global $O(2)$ rotation and reflection invariance
- Cyclic permutation invariance of vertex ordering
- Vertex order reversal (dihedral flip) invariance

The **RingTransformer** architecture implements these symmetries by constructing a centered Gram matrix $G = (X - \bar{X})(X - \bar{X})^\top$ and extracting ring tokens with circulant multi-head self-attention. Dihedral reflection invariance is enforced via the `forward_flip` method:

$$
f_{\text{flip}}(c, X) = \frac{1}{2}\left(f(c, X) + f(c, X_{\text{reversed}})\right)
$$

The network configuration uses:
- Depth: $L = 4$ transformer blocks
- Hidden width: $N_f = 128$
- Number of heads: $n_{\text{heads}} = 4$
- Context dimension: $m = 1$
- Output dimension: $N_{\text{inv}} = 30$

The predicted spectrum $\hat{\lambda}$ is obtained from the model's invariant output token $\tilde{c}$ as:

$$
\hat{\lambda}(X) = \left[ f_{\text{flip}}(\mathbf{1}, X)_0 \right]^2
$$

### 2.2 Surrogate Training and Epoch Matching
The surrogate models are trained by minimizing the mean squared log-spectral loss:

$$
\mathcal{L}_{\text{train}}(\theta) = \frac{1}{B} \sum_{b=1}^B \sum_{k=1}^{30} \left( \ln \hat{\lambda}_k(X^{(b)}; \theta) - \ln \lambda_k^{(b)} \right)^2
$$

To isolate the effect of dataset size from optimization maturity, the number of updates is chosen so that each sample is drawn an equal number of times on average:

$$
E = \frac{N_{\text{updates}} \times B}{N_{\text{train}}} \approx 166.7 \text{ epochs}
$$

- **Small Dataset (900 samples):** $B = 10$, $N_{\text{updates}} = 15,000$ updates ($E = 166.7$ epochs).
- **Extended Dataset (5,000 samples):** $B = 10$, $N_{\text{updates}} = 83,333$ updates ($E = 166.7$ epochs).
- Optimizer: Adam ($\eta = 10^{-4}$), fully vectorized with `jax.lax.scan` on GPU.

### 2.3 Direct Inverse Shape Optimization
Given target spectrum $\lambda^*$, we reconstruct the unknown triangle vertices $X$ by solving:

$$
\min_{X \in \mathbb{R}^{3 \times 2}} \mathcal{L}_{\text{shape}}(X) = \sum_{k=1}^{30} \left( \ln \lambda^*_k - \ln \hat{\lambda}_k(X; \theta^*) \right)^2
$$

where $\theta^*$ are the frozen weights of the trained surrogate.
- Optimization parameters: Adam ($\eta = 0.1$, 1,000 steps).
- For each test drum, the initial drum $X_0$ is initialized as a randomly chosen drum from the training set.

### 2.4 Evaluation Metric
Geometric reconstruction error is measured using the Procrustes-invariant distance across all cyclic and dihedral vertex permutations:

$$
\text{Error}(X_{\text{pred}}, X_{\text{target}}) = \min_{\pi \in \Pi} \text{Procrustes}(X_{\text{pred}}, \pi(X_{\text{target}}))
$$

---

## 3. Experimental Setup

- **Test Target Set:** 10 random drums drawn from the held-out test split (samples 900–999) of `triangle_drums.npz`:
  - Target sample IDs: `[996, 971, 908, 960, 941, 994, 968, 909, 919, 982]`
- **Initial Starting Drums:** 10 random drums drawn from the training split (indices 0–899):
  - Starting sample IDs: `[700, 361, 331, 577, 739, 163, 401, 446, 490, 829]`
- Both Experiment 1 and Experiment 2 share the exact same paired test and starting configurations.

---

## 4. Experimental Results

### 4.1 Surrogate Forward Model Convergence

| Metric | Exp 1: Extended (5,000 samples) | Exp 2: Small (900 samples) |
| :--- | :---: | :---: |
| **Dataset Size** | 5,000 | 900 |
| **Total Updates** | 83,333 | 15,000 |
| **Effective Epochs** | **166.7** | **166.7** |
| **Training Time (GPU)** | 121.93 s | 26.35 s |
| **Initial Training Loss** | 1579.50 | 1411.10 |
| **Final Training Loss** | **0.0015** | 0.0564 |
| **Mean Relative Spectrum Error (10 Test Drums)** | **0.7475%** | 1.8190% |
| **Mean Relative Spectrum Error (All 100 Test Drums)** | **0.8035%** | 2.6246% |

---

### 4.2 Invariant Shape Optimization Results

The table below presents the initial shape error from the random starting drum and the final shape error achieved by both surrogate models after 1,000 optimization steps.

| # | Target ID | Start ID | Initial Error | Exp 1 (5k, Matched) Final Error | Exp 2 (900) Final Error | Relative Difference | Exp 1 Loss | Exp 2 Loss |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **1** | 996 | 700 | 0.8260 | **0.0157** | 0.0879 | **-82.1%** | $3.06 \times 10^{-4}$ | $1.12 \times 10^{-2}$ |
| **2** | 971 | 361 | 0.3501 | **0.0389** | 0.0856 | **-54.6%** | $3.86 \times 10^{-4}$ | $8.78 \times 10^{-3}$ |
| **3** | 908 | 331 | 0.7061 | **0.0083** | 0.0431 | **-80.8%** | $5.49 \times 10^{-4}$ | $9.94 \times 10^{-3}$ |
| **4** | 960 | 577 | 0.4364 | **0.0057** | 0.0250 | **-77.4%** | $1.84 \times 10^{-3}$ | $9.35 \times 10^{-3}$ |
| **5** | 941 | 739 | 1.5096 | **0.0316** | 0.1128 | **-72.0%** | $1.57 \times 10^{-3}$ | $7.30 \times 10^{-3}$ |
| **6** | 994 | 163 | 0.2134 | **0.0042** | 0.0098 | **-57.1%** | $5.79 \times 10^{-4}$ | $1.07 \times 10^{-2}$ |
| **7** | 968 | 401 | 1.5613 | **0.0189** | 0.0869 | **-78.2%** | $1.78 \times 10^{-3}$ | $9.61 \times 10^{-3}$ |
| **8** | 909 | 446 | 1.5459 | **0.0045** | 0.0335 | **-86.6%** | $3.31 \times 10^{-4}$ | $1.85 \times 10^{-2}$ |
| **9** | 919 | 490 | 0.4303 | **0.0217** | 0.0257 | **-15.6%** | $4.05 \times 10^{-4}$ | $4.35 \times 10^{-3}$ |
| **10** | 982 | 829 | 0.5036 | 0.0135 | **0.0130** | +3.4% | $5.98 \times 10^{-4}$ | $6.99 \times 10^{-3}$ |

---

### 4.3 Summary Statistics

| Metric | Initial Shape Error | Exp 1: 5,000 samples (166.7 epochs) | Exp 2: 900 samples (166.7 epochs) |
| :--- | :---: | :---: | :---: |
| **Mean Error** | 0.7853 | **0.0163** | 0.0523 |
| **Median Error** | 0.6049 | **0.0146** | 0.0383 |
| **Minimum Error** | 0.2134 | **0.0042** | 0.0098 |
| **Maximum Error** | 1.5613 | **0.0389** | 0.1128 |

---

### 4.4 Training Budget Ablation: 30 vs. 166.7 Epochs on the 5,000-Sample Dataset

To highlight the importance of matching the epoch budget, we compare the 5,000-sample surrogate trained for 15,000 updates (30 epochs) versus 83,333 updates (166.7 epochs):

| Configuration | Epochs | Final Training Loss | Test Spec. Error (10 Drums) | Mean Final Shape Error | Median Final Shape Error |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **5k Dataset (Fixed Updates)** | 30.0 | 0.0613 | 5.35% | 0.1318 | 0.0973 |
| **900 Dataset (Fixed Updates)** | 166.7 | 0.0564 | 1.82% | 0.0523 | 0.0383 |
| **5k Dataset (Matched Epochs)** | **166.7** | **0.0015** | **0.75%** | **0.0163** | **0.0146** |

When the 5,000-sample model was only trained for 15,000 updates (30 epochs), it was undertrained compared to the 900-sample model (which completed 166.7 epochs in 15,000 updates). Once the epoch budget was matched, the 5,000-sample surrogate achieved more than **$3\times$ lower shape reconstruction error** than the 900-sample model.

---

## 5. Discussion and Insights

1. **Benefit of Dataset Diversity on Surrogate Inversion:**
   With equalized epoch exposure, expanding the dataset from 900 to 5,000 samples provides the surrogate with a much richer sampling of the triangle parameter space. This improves both the forward spectral approximation (relative error $< 0.8\%$) and the fidelity of its spatial gradients $\nabla_X \mathcal{L}$, avoiding spurious local minima that occasionally trapped the smaller model.
2. **Sharpness of the Inverse Energy Landscape:**
   In the 5,000-sample matched model, final shape losses reached $\sim 10^{-4}$, accompanied by tight invariant shape errors (e.g. $0.0042$, $0.0045$, $0.0057$). The higher density of training data ensures that the learned surrogate spectral surface preserves accurate curvature near the global optimum.
3. **Extreme Distance Invariance:**
   Even when the optimization started from completely disparate geometries (e.g. initial errors $> 1.5$ in Drums #5, #7, #8), the gradient trajectory navigated back to the exact target shape, demonstrating wide and well-behaved basins of attraction in the learned surrogate.
4. **Computational Efficiency:**
   Training the 83,333-update surrogate required only ~2 minutes on a single GPU using `jax.lax.scan`. The subsequent shape inversions took less than 0.5 seconds per drum, providing an efficient alternative to full forward FEM differentiation.
