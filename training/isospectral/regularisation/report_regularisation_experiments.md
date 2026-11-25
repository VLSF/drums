# Geometric Invariant Regularization for Isospectral Drum Reconstruction

This report investigates the inverse problem of recovering planar drum shapes from Dirichlet Laplacian spectra in the presence of binary isospectral ambiguity. Using the Gordon–Webb–Wolpert (GWW) 30k dataset (`isospectral_30k.npz`), we analyze why standard variance penalties fail, develop boundary invariant regularizers (edge multiset matching and turning angle cosine matching), and benchmark them against pure regression and the unambiguous single-branch control ceiling.

---

## Executive Summary

1. **Failure of Naive Variance Regularization:**
   In an ambiguous inverse problem with twin targets $\{D_1, D_2\}$, regression minimizers naturally converge toward the conditional expectation (the blurry Fréchet midpoint $M = \frac{1}{2}(D_1 + D_2)$). Minimizing a penalty proportional to vertex variance $\sum \|r_i - \bar{r}\|^2$ actively accelerates collapse: by the parallelogram inequality, the blurry midpoint already has an average $12.5\%$ variance deficit compared to the individual sharp drums. Penalizing variance drives the model toward shrunken and degenerate shapes rather than sharp drums.

2. **Resolution of the Pure Regression Discrepancy:**
   Previous runs suggested pure regression reached only $13\%$ sharpness. When evaluated under the benchmark protocol of `isospectral_dataset_experiments.md` ($30\text{k}$ updates, batch size $128$, PCA-whitened `ratio_white` features, separation filtering $\text{sep} > 0.1$, and sharpness threshold $e_{\text{set}}/\text{sep} < 0.25$), baseline pure regression achieves **$35.2\%$ sharp predictions** ($e_{\text{set}}/\text{sep} = 0.2667$), closely matching the reference report ($32.8\%$).

3. **Angle Cosines Regularization Doubles Sharp Predictions:**
   Because Gordon–Webb–Wolpert twins are constructed from 7 congruent right triangles, both drums $D_1$ and $D_2$ share the **exact same multiset of sorted corner angle cosines** (identical to $10^{-15}$). Penalizing deviations from the target multiset of corner angles acts as a non-convex symmetry-breaking barrier that destabilizes rounded midpoint corners:
   - **Sharp drum rate ($e_{\text{set}}/\text{sep} < 0.25$)** jumps from **$35.2\%$ (pure regression) to $60.8\%$** (nearly double).
   - **Blurry predictions ($e_{\text{set}}/\text{sep} > 0.40$)** drop from **$14.4\%$ down to $8.6\%$**.
   - **Mean separation ratio ($e_{\text{set}}/\text{sep}$)** drops from **$0.2667$ down to $0.2098$**.
   - **Closer-twin reconstruction error ($e_{\text{set}}$)** drops from **$0.1032$ down to $0.0887$**.

4. **The Single-Branch Ceiling:**
   When trained on an unambiguous canonical single branch (`isospectral_30k_single.npz`), the model achieves **$100.0\%$ sharp predictions** with a relative vertex error of **$0.0017$ ($0.17\%$)**, confirming that spectral inversion is structurally precise once the binary ambiguity is resolved.

---

## Problem Formulation & Geometry of Blurring

Let $\lambda \in \mathbb{R}^{30}$ denote the first 30 Dirichlet Laplacian eigenvalues of a planar drum. In the `isospectral_30k` dataset, each spectrum $\lambda$ corresponds to exactly two distinct, non-isometric polygons:
$$
\text{Spec}(D_1) = \text{Spec}(D_2) = \lambda
$$
When a deterministic neural network $f_\theta(\lambda)$ is trained to predict vertex coordinates $Y \in \{D_1, D_2\}$ under a symmetric loss $\mathcal{D}$, the empirical risk minimizer is the conditional expectation:
$$
f_{\theta}^{\ast}(\lambda) = \arg\min_{\hat{X}} \mathbb{E}_{Y \sim \{D_1, D_2\}} [\mathcal{D}(Y, \hat{X})] \approx \frac{D_1 + D_2}{2}
$$
The resulting prediction is a blurry geometric blend that satisfies neither physical boundary constraint.

### Why Variance Penalties Fail
Let $D_1, D_2$ be two centered configurations. For their midpoint $M = \frac{1}{2}(D_1 + D_2)$:
$$
\|M\|_2^2 = \frac{\|D_1\|_2^2 + \|D_2\|_2^2}{4} + \frac{\langle D_1, D_2 \rangle}{2}
$$
Because $D_1 \ne D_2$, the Cauchy-Schwarz inequality gives $\langle D_1, D_2 \rangle < \|D_1\|_2 \|D_2\|_2$, implying:
$$
\|M\|_2^2 < \frac{\|D_1\|_2^2 + \|D_2\|_2^2}{2}
$$
Across all 30,000 dataset pairs, the midpoint's vertex variance is on average **$12.5\%$ smaller** than the individual drums. Adding a positive variance penalty $+ \gamma \sum \|r_i - \bar{r}\|^2$ to a minimization objective creates positive curvature centered at the origin, rewarding boundary shrinkage and worsening shape collapse.

---

## Geometric Invariant Regularizers

To force the network toward a sharp drum without requiring twin supervision at inference time, we introduce regularizers based on exact shared invariants of the GWW construction:

```
+---------------------------------------------------------------------------------------------------+
|                                   GWW INVARIANT REGULARIZERS                                      |
+--------------------------+--------------------------------------+---------------------------------+
| Regularizer              | Invariant Formulation                | Geometric Role                  |
+--------------------------+--------------------------------------+---------------------------------+
| Perimeter Consistency    | ((Perim(X) - Perim(Y)) / Perim(Y))^2 | Fixes global boundary scaling   |
| Edge Multiset Matching   | Mean((sort(e(X)) - sort(e(Y))) / e)^2| Enforces 3a, 3b, 3c side lengths|
| Angle Cosines Matching   | Mean(sort(cos theta(X)) - sort(cos))^2| Enforces discrete corner angles |
| Combined Edges + Angles  | 0.5 R_edges + 0.5 R_angles           | Differential boundary lock      |
+--------------------------+--------------------------------------+---------------------------------+
```

### 1. Exact Boundary Perimeter Invariance
By Kac's theorem (second heat trace asymptotic coefficient), every pair of isospectral planar domains has identical boundary perimeter:
$$
\text{Perimeter}(D_1) \equiv \text{Perimeter}(D_2) \quad (\text{verified to } 10^{-14} \text{ on all 30k pairs})
$$
The relative perimeter discrepancy regularizer is defined as:
$$
\mathcal{R}_{\text{perim}}(\hat{X}, Y) = \left( \frac{\sum_{i=1}^9 \|x_{i+1} - x_i\|}{\sum_{i=1}^9 \|y_{i+1} - y_i\|} - 1 \right)^2
$$

### 2. Edge-Length Multiset Matching
GWW drums are formed from 7 congruent right triangles with legs $a, b$ and hypotenuse $c = \sqrt{a^2 + b^2}$. Across all pairs, the 9 boundary edges strictly consist of $\{3 \times a, 3 \times b, 3 \times c\}$. Both twins share the identical sorted edge vector:
$$
\text{sort}(\vec{e}(D_1)) \equiv \text{sort}(\vec{e}(D_2))
$$
In the blurry midpoint, interior vertex averaging produces fractional, non-physical edges deviating by up to $36\%$. The edge regularizer penalizes this deviation:
$$
\mathcal{R}_{\text{edges}}(\hat{X}, Y) = \frac{1}{9} \sum_{i=1}^9 \left( \frac{\text{sort}(\vec{e}(\hat{X}))_i - \text{sort}(\vec{e}(Y))_i}{\text{sort}(\vec{e}(Y))_i} \right)^2
$$

### 3. Turning Angle Cosines Matching
Because the right triangles meet in congruent arrangements along the boundary, the interior corner turning angles between adjacent edges satisfy:
$$
\text{sort}(\cos \vec{\theta}(D_1)) \equiv \text{sort}(\cos \vec{\theta}(D_2))
$$
Where for adjacent edge vectors $w_1 = p_{i-1} - p_i$ and $w_2 = p_{i+1} - p_i$:
$$
\cos \theta_i = \frac{w_1 \cdot w_2}{\|w_1\|_2 \|w_2\|_2 + \epsilon}
$$
The blurry midpoint rounds corners (e.g. true $\cos \theta \in [-0.71, 0.84]$ vs midpoint $\cos \theta \in [-0.98, 0.97]$). The angle regularizer penalizes non-physical corner rounding:
$$
\mathcal{R}_{\text{angles}}(\hat{X}, Y) = \frac{1}{9} \sum_{i=1}^9 \left( \text{sort}(\cos \vec{\theta}(\hat{X}))_i - \text{sort}(\cos \vec{\theta}(Y))_i \right)^2
$$

---

## Experimental Setup

- **Dataset:** `iso30k/isospectral_30k.npz` (30,000 drums).
  - Train: 27,000 drums.
  - Validation: 1,500 drums.
  - Test: 1,500 drums.
- **Input Features (`ratio_white`):**
  Log-frequency ratios $z = [\ln(\lambda_k/\lambda_1)_{k \ge 2}, \ln \lambda_1] \in \mathbb{R}^{30}$, standardized and PCA-whitened on the training set:
  - Empirical training covariance: $\max | \text{Cov}(F_{\text{train}}) - I | = 3.96 \times 10^{-5}$, condition number $= 1.000062$.
- **Model Architecture:** `DrumTransformer` (5 layers, embedding dimension 256, 4 heads, MLP ratio 4).
- **Optimization:** AdamW with weight decay $10^{-4}$, gradient clipping at norm $1.0$.
  - Peak learning rate: $5 \times 10^{-4}$.
  - Warmup: 500 steps, cosine decay to $10^{-6}$ over 30,000 updates.
  - Batch size: 128 (total $3.84 \times 10^6$ samples seen).
- **Evaluation Protocol:**
  Following `iso30k/metrics.py`, metrics are computed on the 1,308 test drums with twin separation $\text{sep} > 0.1$:
  - $e_{\text{set}} = \min(d(D_1, \hat{X}), d(D_2, \hat{X}))$ (Procrustes distance to closer twin).
  - $\text{sharp}$: fraction with $e_{\text{set}}/\text{sep} < 0.25$.
  - $\text{blurry}$: fraction with $e_{\text{set}}/\text{sep} > 0.40$.
  - $\text{mid}/\text{sep}$: Procrustes distance to midpoint relative to separation.
  - $\text{pick}$: fraction of sharp predictions closer to the sample's target drum than its twin.

---

## Empirical Results

| Experiment | Regularizer $\mathcal{R}$ | Target Data Arm | Sharp ($< 0.25$) | Blurry ($> 0.40$) | $e_{\text{set}}/\text{sep}$ | Closer Error $e_{\text{set}}$ | Midpoint Ratio $\text{mid}/\text{sep}$ | Pick Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control (Upper Bound)** | None | Single Branch | **$100.0\%$** | **$0.0\%$** | **$0.0050$** | **$0.0017$** | $0.5020$ | $49.6\%$ |
| **Run 0: Baseline Regression** | None | Ambiguous | $35.2\%$ | $14.4\%$ | $0.2667$ | $0.1032$ | $0.4135$ | $54.4\%$ |
| **Run 1: Perimeter** | $\mathcal{R}_{\text{perim}}$ ($\gamma=1.0$) | Ambiguous | $34.4\%$ | $15.2\%$ | $0.2777$ | $0.1074$ | $0.4097$ | $53.1\%$ |
| **Run 2: Edge Multiset** | $\mathcal{R}_{\text{edges}}$ ($\gamma=1.0$) | Ambiguous | $48.2\%$ | $10.2\%$ | $0.2341$ | $0.0956$ | $0.4352$ | $52.1\%$ |
| **Run 3: Angle Cosines** | $\mathcal{R}_{\text{angles}}$ ($\gamma=1.0$) | Ambiguous | **$60.8\%$** | **$8.6\%$** | **$0.2098$** | **$0.0887$** | $0.4390$ | $51.1\%$ |
| **Run 4: Combined** | $0.5\mathcal{R}_e + 0.5\mathcal{R}_a$ | Ambiguous | $47.3\%$ | $13.3\%$ | $0.2490$ | $0.0976$ | $0.4188$ | $53.8\%$ |

---

## Analysis & Discussion

### 1. Perimeter Consistency is Insufficient
Perimeter regularization successfully cuts the boundary length deficit in half (from $5.2\%$ to $2.9\%$), but leaves sharpness virtually unchanged ($34.4\%$ vs $35.2\%$). A 1D scalar perimeter constraint can be satisfied by isotropically expanding the blurry midpoint without resolving the differential angle and edge distortions.

### 2. Micro-Geometric Invariants Break Symmetry
Both Edge Multiset Matching and Angle Cosines Matching provide multi-dimensional constraints on boundary geometry:
- **Edge multiset matching** forces the segments to take on discrete lengths $\{a, b, c\}$, raising the sharp rate from $35.2\%$ to $48.2\%$.
- **Angle cosines matching** is the top-performing regularizer, raising sharpness to **$60.8\%$** and cutting blurriness to **$8.6\%$**. Because angle cosines are inherently scale-invariant, they provide clean, well-conditioned gradient signals that directly penalize the curvature smoothing typical of midpoint averaging.
- The pick rate remains near $50\%$ ($51.1\%$), confirming that the model breaks symmetry spontaneously without leaking the training target label.

### 3. Gap to Single-Branch Upper Bound
While angle cosines regularization nearly doubles sharpness on ambiguous data ($60.8\%$), it does not reach the $100\%$ sharpness of the single-branch control arm. This remaining gap reflects the fundamental topological barrier of learning one-to-many continuous mappings with a single-valued deterministic network. Complete resolution requires either explicit branch conditioning (canonical ordering) or stochastic generative modeling (flow matching / diffusion).

---

## Reproduction Commands

From `training/isospectral/`:

```bash
# Run baseline pure regression (Run 0)
python regularisation/regularisation_experiments.py --run run0

# Run Angle Cosines Matching (Run 3)
python regularisation/regularisation_experiments.py --run run3 --gamma 1.0

# Run Single-Branch Control Ceiling
python regularisation/regularisation_experiments.py --run control

# Run all ablation experiments
python regularisation/regularisation_experiments.py --run all --gamma 1.0
```
