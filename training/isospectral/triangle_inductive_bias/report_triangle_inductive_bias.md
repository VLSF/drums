# GWW Triangle-Gluing Inductive Bias for Isospectral Drum Inversion

This report investigates a structural inductive bias for recovering planar drums from Dirichlet Laplacian spectra under binary isospectral ambiguity. Rather than predicting 9 unconstrained Cartesian vertices, the network predicts a single 3-vertex base triangle, which is then analytically and differentially expanded into a complete 9-vertex Gordon–Webb–Wolpert (GWW) drum via reflection symmetries. We evaluate this architecture with and without variance regularization, and introduce non-variance geometric regularizers (exact area consistency, perimeter coupling, and canonical orientation).

---

## Executive Summary

1. **Structural Elimination of the Blurry Midpoint:**
   In standard unconstrained regression, training on ambiguous isospectral pairs $\{D_1, D_2\}$ causes the network to collapse toward the unphysical, blurred conditional expectation $M = \frac{1}{2}(D_1 + D_2)$ ($35.2\%$ sharp predictions, $e_{\text{set}}/\text{sep} = 0.2667$). Constraining the model to predict a 3-vertex base triangle that is analytically assembled into a GWW drum physically prevents the model from generating intermediate non-GWW geometries:
   - **GWW-I Model (no regularization):** Sharp predictions ($r < 0.25$) jump to **$86.0\%$** (up from $35.2\%$). Closer-twin reconstruction error ($e_{\text{set}}$) drops to **$0.0389$** (a **$62.3\%$ error reduction**).
   - **GWW-II Model (no regularization):** Sharp predictions reach **$80.0\%$**, with $e_{\text{set}} = 0.0480$.

2. **Empirical Confirmation of Variance Shrinkage Collapse:**
   Applying relative variance regularization $+ \gamma \frac{\text{Var}(\text{pred})}{\text{Var}(\text{target})}$ degrades performance in both models:
   - Sharpness drops from **$86.0\%$ down to $66.8\%$** on GWW-I, and from **$80.0\%$ down to $69.0\%$** on GWW-II.
   - Closer-twin error increases from **$0.0389$ up to $0.0758$**, and boundary perimeter deficit increases from $1.48\%$ to $3.98\%$.
   - **Mechanism:** Minimizing variance rewards smaller base triangles, driving the optimizer to scale down the triangle and artificially shrinking the drum.

3. **Area Consistency Achieves a New SOTA ($90.9\%$ Sharp):**
   Unlike variance (which differs between $D_1$ and $D_2$), the total domain area $\text{Area}(\text{Drum}) \equiv 7 \times \text{Area}(T)$ is an exact invariant shared by both twins down to $10^{-14}$. Enforcing exact area consistency ($\mathcal{R}_{\text{area}}$):
   - Sharp predictions reach an all-time high of **$90.9\%$**, with blurry predictions dropping to just **$3.3\%$**.
   - Separation ratio ($e_{\text{set}}/\text{sep}$) falls to **$0.0803$**.
   - Closer-twin error ($e_{\text{set}}$) drops to **$0.0247$** (a **$76.1\%$ reduction** relative to unconstrained regression).
   - Boundary perimeter deficit drops to an almost negligible **$0.64\%$**.

---

## Geometric Formulation: Gordon–Webb–Wolpert Construction

Gordon, Webb, and Wolpert (1992) and Buser et al. (1994) showed that planar isospectral pairs can be generated from an arbitrary base triangle $T = [t_0, t_1, t_2] \in \mathbb{R}^{3 \times 2}$. Seven copies of $T$ are glued along matching edges through reflections across boundary lines:

```
+---------------------------------------------------------------------------------------------------+
|                                 GWW REFLECTION ASSEMBLY PROTOCOL                                  |
+--------------------------+--------------------------------------+---------------------------------+
| Domain                   | Sequence of Edge Reflections         | Boundary Vertex Assembly        |
+--------------------------+--------------------------------------+---------------------------------+
| GWW Drum I               | T0 -> refl(1,2) -> refl(1,0) ->      | Stack [t0[0], t2[1], t2[2],     |
|                          | refl(0,2) -> refl(0,1) -> refl(2,1)  |        t2[0], t4[2], t6[1],     |
|                          | -> refl(2,0)                         |        t5[0], t5[1], t3[2]]     |
+--------------------------+--------------------------------------+---------------------------------+
| GWW Drum II              | T0 -> refl(0,1) -> refl(1,2) ->      | Stack [t1[0], t1[2], t1[1],     |
|                          | refl(0,2) -> refl(0,1) -> refl(0,2)  |        t2[0], t5[1], t5[2],     |
|                          | -> refl(2,1)                         |        t6[0], t6[1], t3[2]]     |
+--------------------------+--------------------------------------+---------------------------------+
```

### Differentiable Reflection Operator
Let $T = [t_0, t_1, t_2]$. The reflection of vertex $v$ across the line segment connecting edge $(s_0, s_1)$ is given by:

$$
e_1 = t_{s_1} - t_{s_0}, \quad u = \frac{e_1}{\|e_1\|_2}
$$

$$
p = t_v - t_{s_0}, \quad t_v^{\text{reflected}} = 2 (u \cdot p) u - p + t_{s_0}
$$

Because this mapping is composed exclusively of linear projections and inner products, the entire boundary assembly is infinitely differentiable with respect to the initial vertices of $T$.

---

## Model Architectures

Instead of predicting all 9 vertices directly, the network predicts only the 3 vertices of the base triangle:

1. **`GWW_I_Model`:**
   - Base Network: `DrumTransformer(N_V=3, N_f=256, dim=30, L=5)`
   - Input: Whitened spectrum $\lambda \in \mathbb{R}^{30}$
   - Output: Base triangle $T \in \mathbb{R}^{3 \times 2}$
   - Final Prediction: $\hat{X} = \text{Assembly}_{\text{I}}(T) \in \mathbb{R}^{9 \times 2}$

2. **`GWW_II_Model`:**
   - Base Network: `DrumTransformer(N_V=3, N_f=256, dim=30, L=5)`
   - Input: Whitened spectrum $\lambda \in \mathbb{R}^{30}$
   - Output: Base triangle $T \in \mathbb{R}^{3 \times 2}$
   - Final Prediction: $\hat{X} = \text{Assembly}_{\text{II}}(T) \in \mathbb{R}^{9 \times 2}$

---

## Non-Variance Geometric Regularizers

To complement the triangle-gluing inductive bias without inducing the shrinkage caused by variance penalties, we introduce regularizers based on strict geometric invariants:

### 1. Strategy 1: Exact Area Consistency (Anti-Collapse)
Both twins consist of 7 copies of the base triangle. By Weyl's law, the domain area is uniquely determined by the spectrum and is identical between $D_1$ and $D_2$ down to $10^{-14}$:

$$
\text{Area}(\text{Drum}) \equiv 7 \times \text{Area}(T) = \text{Area}(Y)
$$

$$
\mathcal{R}_{\text{area}}(T, Y) = \left( \frac{7 \times \text{Area}(T)}{\text{Area}(Y)} - 1 \right)^2
$$

where $\text{Area}(T) = \frac{1}{2} |(t_1 - t_0)_x (t_2 - t_0)_y - (t_1 - t_0)_y (t_2 - t_0)_x|$.

### 2. Strategy 2: Exact Perimeter Consistency (3-to-1 Coupling)
The 9 boundary edges of any GWW drum consist of exactly 3 copies of each of the 3 sides of $T$:

$$
\text{Perimeter}(\text{Drum}) \equiv 3 \times \text{Perimeter}(T) = \text{Perimeter}(Y)
$$

$$
\mathcal{R}_{\text{perim}}(T, Y) = \left( \frac{3 \times \text{Perimeter}(T)}{\text{Perimeter}(Y)} - 1 \right)^2
$$

### 3. Strategy 4: Canonical Triangle Orientation
In the dataset generator, the canonical domain ordering is defined by $\theta_0 \ge \theta_2$. We penalize non-canonical triangle orientations:

$$
\mathcal{R}_{\text{canonical}}(T) = \max(0, \, \theta_2 - \theta_0)^2
$$

where $\theta_i$ is the interior angle at corner $i$.

---

## Experimental Benchmark

All models were trained on 27,000 ambiguous training drums for **30,000 updates** (batch size 128, AdamW, peak learning rate $5 \times 10^{-4}$ with warmup and cosine decay to $10^{-6}$, `ratio_white` features). Evaluated on the 1,500 test drums using the standard protocol ($\text{sep} > 0.1$, $N = 1,308$):

| Experiment | Inductive Bias | Regularizer $\mathcal{R}$ | Sharp ($r < 0.25$) | Blurry ($r > 0.40$) | $e_{\text{set}}/\text{sep}$ | Closer Error $e_{\text{set}}$ | Midpoint Ratio $\text{mid}/\text{sep}$ | Pick Rate | Var Ratio $\frac{\text{Var}(\text{pred})}{\text{Var}(\text{target})}$ | Perimeter Deficit |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GWW-I + Area Consist.** | Triangle Gluing I | $\mathcal{R}_{\text{area}}$ ($\gamma=1.0$) | **$90.9\%$** | **$3.3\%$** | **$0.0803$** | **$0.0247$** | $0.4455$ | $48.9\%$ | $0.9667$ | **$0.64\%$** |
| **GWW-I + Perim Consist.** | Triangle Gluing I | $\mathcal{R}_{\text{perim}}$ ($\gamma=1.0$) | **$87.0\%$** | **$5.0\%$** | **$0.1020$** | **$0.0347$** | $0.4340$ | $49.6\%$ | $0.9616$ | **$1.02\%$** |
| **GWW-I Model (Pure)** | Triangle Gluing I | None | **$86.0\%$** | **$5.2\%$** | **$0.1096$** | **$0.0389$** | $0.4318$ | $50.7\%$ | $0.9579$ | **$1.48\%$** |
| **GWW-I + Area & Perim** | Triangle Gluing I | $\mathcal{R}_a + \mathcal{R}_p$ ($\gamma=1.0$) | **$86.3\%$** | **$4.0\%$** | **$0.0958$** | **$0.0306$** | $0.4387$ | $50.6\%$ | $0.9670$ | **$0.67\%$** |
| **GWW-II Model (Pure)** | Triangle Gluing II | None | **$80.0\%$** | **$7.9\%$** | **$0.1355$** | **$0.0480$** | $0.4176$ | $50.4\%$ | $0.9594$ | **$1.52\%$** |
| **GWW-I + Canonical** | Triangle Gluing I | $\mathcal{R}_{\text{canon}}$ ($\gamma=0.2$) | **$81.1\%$** | **$7.2\%$** | **$0.1295$** | **$0.0485$** | $0.4329$ | $52.2\%$ | **$1.0073$** | **$1.74\%$** |
| **GWW-II + Variance Reg** | Triangle Gluing II | $\text{Var}_{\text{rel}}$ ($\gamma=0.1$) | $69.0\%$ | $11.6\%$ | $0.1903$ | $0.0757$ | $0.4084$ | $50.8\%$ | $0.9205$ | $3.86\%$ |
| **GWW-I + Variance Reg** | Triangle Gluing I | $\text{Var}_{\text{rel}}$ ($\gamma=0.1$) | $66.8\%$ | $12.2\%$ | $0.1950$ | $0.0758$ | $0.4007$ | $49.4\%$ | $0.9190$ | $3.98\%$ |
| *Unconstrained Baseline* | None (9 Vertices) | None | $35.2\%$ | $14.4\%$ | $0.2667$ | $0.1032$ | $0.4135$ | $54.4\%$ | $0.8745$ | $5.18\%$ |

---

## Detailed Findings

1. **Architectural Inductive Bias is Superior to Soft Loss Regularization:**
   Unconstrained 9-vertex regression reaches only $35.2\%$ sharpness because the loss landscape allows the network to find a compromise midpoint. Constraining the network to output a 3-vertex base triangle structurally forbids the non-GWW blurry midpoint, instantly elevating sharpness to **$86.0\%$** without any extra loss penalty.

2. **Area Consistency Anchors Physical Scale ($90.9\%$ Sharp):**
   When unconstrained base triangles are trained on ambiguous pairs, the optimizer occasionally shrinks the base triangle slightly to reduce the average Procrustes distance across alternating targets. Because $\text{Area}(\text{Drum}) \equiv 7 \times \text{Area}(T)$ is an invariant shared by both twins, penalizing area discrepancy ($\mathcal{R}_{\text{area}}$) provides a stiff anchor that eliminates shrinkage, pushing sharpness to **$90.9\%$** and cutting closer-twin error to **$0.0247$**.

3. **Why Variance Regularization Fails Universally:**
   Across all experiments, adding a positive variance penalty $\gamma \frac{\text{Var}(\text{pred})}{\text{Var}(\text{target})}$ degrades performance. Even when the geometry is restricted to GWW drums, the optimizer satisfies the variance penalty by shrinking the base triangle toward $(0, 0)$, as evidenced by the drop in variance ratio ($0.958 \to 0.919$) and the rise in perimeter deficit ($1.48\% \to 3.98\%$).

---

## Reproduction Commands

From `training/isospectral/`:

```bash
# Run pure GWW-I Model (no regularization)
python triangle_inductive_bias/task_4_experiments.py --run gww1_noreg

# Run GWW-I with Area Consistency (Strategy 1, SOTA 90.9% sharp)
python triangle_inductive_bias/task_4_experiments.py --run gww1_area --gamma 1.0

# Run GWW-I with Perimeter Consistency (Strategy 2)
python triangle_inductive_bias/task_4_experiments.py --run gww1_perim --gamma 1.0

# Run GWW-I with Variance Regularization (gamma=0.1)
python triangle_inductive_bias/task_4_experiments.py --run gww1_var --gamma 0.1

# Run pure GWW-II Model
python triangle_inductive_bias/task_4_experiments.py --run gww2_noreg
```
