# Hearing the shape of a triangular drum

**Speaker notes · 7–10 minutes**  
Use the figures to show the admissible shapes, the time-window sweep, and the reconstruction of dataset row 900. Keep the questions at the end in reserve. [Figure 1](isosceles_boundary.png) · [Figure 2](window_sweep.png) · [Figure 3](row_900_reconstruction.png)

## 0:00–1:00 · The question and the result

Suppose I give you the spectrum of a vibrating triangular membrane and tell you that its edge is fixed. Can you recover the shape? With the full, exact Dirichlet spectrum, the answer is yes. Grieser and Maronna proved that the heat trace reveals three quantities that specify a triangle up to congruence. We wanted to see how this works with a finite spectrum and a numerical fit. [Grieser and Maronna, *Hearing the Shape of a Triangle*](https://arxiv.org/abs/1208.3163)

We tested the 100 original held-out triangles in the dataset, with 60,000 numerical eigenvalues per triangle. At our initial time window, strict inversion reached the two-percent error target on 36 triangles. A geometry-constrained fallback brought that to 41. Then we moved the time window and ran the same strict inverse. The pass count reached 97.

## 1:00–2:30 · From a spectrum to a triangle

Each eigenvalue $\lambda_j$ contributes $e^{-t\lambda_j}$ to the heat trace at time $t$. At short times, the trace has three leading terms.

$$
Z(t)=\frac{A}{4\pi t}-\frac{P}{8\sqrt{\pi t}}+\gamma+O(e^{-c/t}),
\qquad
24\gamma+1=\pi\sum_{i=1}^3\frac{1}{\theta_i}.
$$

Here $A$ is the area, $P$ is the perimeter, and the constant term depends on the angles. The exact spectrum gives the whole function $Z(t)$, so the exact coefficients can be recovered in the short-time limit. In the experiment we kept the first $K=60{,}000$ terms and fitted the three coefficients to $Z_K(t)$ on 100 logarithmically spaced times. Each fit used the same eigenvalue list throughout its window.

To reconstruct the triangle, we use $R=24\gamma+1$ and $F=P^2/(4A)$. The incircle geometry gives $F=\sum_i\cot(\theta_i/2)$. With the angles ranked from smallest to largest, fixing $R$ restricts the possible shapes to an arc. $F$ varies monotonically along it. We solve for one angle on the arc, recover the other two, and use the law of sines and the perimeter to set the physical scale. We score the mean relative vertex error after aligning and relabeling the vertices, with scale preserved.

## 2:30–4:15 · Compatibility and the isosceles fallback

*Show Figure 1. Point along the arc and then across the narrow band on the right.*

At $\gamma=3/8$, this arc starts at the isosceles triangle with angles $36,72,72$ degrees and ends at the right isosceles triangle with angles $45,45,90$ degrees. The admissible values of $F$ run from about 5.82843 to 5.83045. Even a small fit error can move the estimate above the top of that band. There is also an absolute floor of $1/3$ for $\gamma$, attained by the equilateral triangle.

Our strict inverse checks compatibility before solving for the angles. In the original window, 49 fits failed that check. Of those, 29 fell below the corner-term floor, 18 gave incompatible invariants, and two gave a nonpositive area or perimeter.

For these cases, we look for the valid triangle with the smallest heat-trace residual. Why does the search go to the isosceles branches? A scalene triangle is an interior point of the feasible coefficient set. Around that point we can vary all three coefficients independently. So an interior minimum would have to satisfy the ordinary least-squares equations. That solution is already outside the feasible set. A finite positive-scale minimizer therefore lies on one of the two isosceles branches. The closure also has a zero-scale limit, which counts as a failed reconstruction. The fallback produced 49 additional triangles in this run. Five met the error target.

## 4:15–6:10 · Why the spectrum and the time window matter

At fixed mode count, taking $t$ very small makes each retained exponential approach one. The finite sum approaches $K$ while the full trace grows like $A/(4\pi t)$. We want the window early enough for the three-term expansion and late enough to suppress the omitted modes. The quantity to watch is $t_0\lambda_K$, where $t_0$ is the start of the window. Weyl’s leading law gives $\lambda_K\approx4\pi K/A$, so a fixed time window leaves a larger tail for a larger triangle.

Over a narrow window, the three fit functions are similar enough that even small trace errors can shift their coefficients noticeably. The geometry itself can amplify those shifts, especially near symmetric triangles. In the original window, 80 and 100 samples both gave 51 admissible fits and 36 strict passes. Adding time samples describes the same truncated trace more densely. The tail error at a given time stays the same.

We also checked the frequency solver. We verified numerical eigenvalue counts against exact spectra for the equilateral and 30–60–90 triangles. On a separate unit-area audit of the 100 dataset triangles, the median absolute corner-term error was about $5.4\times10^{-7}$. Some cases have larger errors, so numerical precision can still affect a decision close to two percent.

## 6:10–8:15 · Move the window and look at the contour

*Show Figure 2. Point out the strict pass counts at the original, intermediate, and selected windows.*

We kept all 60,000 eigenvalues and 100 time samples per fit. Moving the window from $[0.0001,0.0002]$ to $[0.00025,0.0005]$ raised strict passes from 36 to 77. At $[0.005,0.01]$, 97 triangles gave admissible fits and all 97 reached the two-percent target. This recovered 47 of the 49 triangles rejected by the original strict fit. At the next window, $[0.01,0.02]$, 94 passed. The sweep measures the effect on this dataset and gives a range to test on future shapes.

*Show Figure 3. Start with the original-window contour for row 900.*

This is one of the larger triangles. Its area is 6.81 and its true corner term is 0.416. In the original window $t_0\lambda_K$ was 11.13. The truncated trace was about 0.079 below the three-term expression at the first sample. The fit assigned $\gamma=-0.573$, so strict inversion rejected it. The isosceles fallback gave a triangle with 24.86% vertex error.

*Point to the shifted-window contour.*

At $t_0=0.00025$, the window shown on the right, the cutoff product reaches 27.83 and the strict error falls below 0.007%. The unit-area audit estimated its corner term within $8.2\times10^{-8}$, which supports truncation as the dominant source of the original-window failure. We changed the weight assigned to the same spectrum and recovered the shape.

## 8:15–8:45 · Close

The theorem explains why the inverse exists. Our experiment shows how much the finite-spectrum result depends on where we measure the heat trace. The two basic checks are whether the fitted coefficients describe a valid triangle and whether the trace is measured on a scale where the omitted modes can be controlled. For these 100 shapes, shifting the window improved strict reconstruction far more than enforcing geometry after the original fit failed.

## Questions to keep in reserve

**Why does the fallback include a zero-scale line?**  
In the coordinates $(\alpha,\beta,\gamma)$ of the heat fit, the closure of the positive-scale coefficient set adds $\{(0,0,\gamma):\gamma\ge1/3\}$. With bounded $\gamma$, the reciprocal-angle sum keeps every angle away from zero. A sequence of triangles can therefore leave the positive-scale set at finite coefficients only as its perimeter and area vanish together. Every $\gamma\ge1/3$ can occur in that limit by shrinking a suitable isosceles shape. Least squares has a minimizer on the closure because the design has full rank. The code evaluates zero perimeter as a limiting candidate and records failure if it wins.

**What do you minimize on the two isosceles branches?**  
Write the angles as $(\phi,\phi,\pi-2\phi)$ and let the perimeter vary. At fixed $\phi$, both $F$ and $\gamma$ are known, and $A=P^2/(4F)$. The predicted trace is quadratic in $P$, so the sum of squared residuals is quartic. The implementation checks positive roots of its cubic derivative and the zero-scale limit, then refines a numerical search over $\phi$.

**How can a small trace residual coexist with a large vertex error?**  
If $D$ is the three-column design matrix, the coefficient error induced by a trace perturbation $\delta Z$ is $D^+\delta Z$. The fitted residual only measures the component of $\delta Z$ orthogonal to the columns. The column-normalized design has a condition number of about 470 for the original window. A perturbation can move the fitted point outside the narrow admissible region even when the curve still has a small residual.

**Can we select a window for a new triangle from its spectrum?**  
The last available eigenvalue gives a scale-aware quantity, $t_0\lambda_K$, to monitor before fitting. For these planar triangles, Pólya's bound $\lambda_j\ge4\pi j/A$ also gives $T_K(t)\le e^{-b(K+1)}/(1-e^{-b})$, where $b=4\pi t/A$ and $T_K$ is the omitted tail. Area estimates and stability across neighboring windows could guide a rule. The window with 97 passes came from a sweep on this set, so its performance on new shapes still needs to be measured. [Pólya, *On the eigenvalues of vibrating membranes*](https://doi.org/10.1112/plms/s3-11.1.419)

**Where did the 60,000 eigenvalues come from?**  
The dataset stores 30 FEM eigenvalues per triangle. For this study, a corner-adapted Fourier–Bessel solver computed numerical Dirichlet frequencies at each original geometry in overlapping frequency windows. We stitched the modes with their multiplicities, resolved observed count disagreements, and restored each triangle’s physical scale. The reconstruction then took only the resulting eigenvalues as input. [Lozej, Casati and Prosen, *Quantum chaos in triangular billiards*](https://doi.org/10.1103/PhysRevResearch.4.013138)

**What happens as the mode count grows?**  
At fixed positive times the omitted tail vanishes and the fit approaches the least-squares fit of the full trace on that window. A finite-time model error can remain. If the fitted coefficients converge to those of a scalene triangle, they eventually enter the feasible interior at a finite mode count. A sequence of isosceles shapes has two consecutive ordered angles equal throughout, so any nondegenerate limit remains isosceles.
