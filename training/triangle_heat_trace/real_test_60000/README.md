# Heat-trace reconstruction on the real triangle test split

This run uses the original vertex coordinates at zero-based indices 900–999 of `triangle_drums.npz`. At each geometry, a boundary solver computes numerical Dirichlet frequencies $k_j$ with multiplicity. The reconstruction takes the lowest 60,000 numerical eigenvalues $\lambda_j=k_j^2$, in the physical units of that triangle, and fits the truncated heat trace on 100 logarithmically spaced times in $[10^{-4}, 2\cdot10^{-4}]$. Vertex errors use the same alignment and scale convention as the analytic benchmark. The dataset's stored 30 FEM modes are used for a low-frequency comparison.

The solver rotates or reflects a triangle and rescales it to unit area. Centering a corner-adapted Fourier–Bessel basis at the largest angle satisfies the zero boundary condition on the two adjacent sides. On the remaining side the scaling method estimates roots in overlapping frequency windows. A projected Taylor expansion refines the estimate in each window. Residual checks and overlap disagreements trigger local recentering, then the windows are stitched with multiplicities retained. After stitching, $\lambda_j=k_{j,\mathrm{unit}}^2/A$ restores the original physical scale. [Lozej, Casati and Prosen, *Quantum chaos in triangular billiards*](https://doi.org/10.1103/PhysRevResearch.4.013138)

Because the spectra are numerical, the metadata record overlap counts and a separate unit-area audit with $t\in[4\cdot10^{-5},8\cdot10^{-5}]$. The Pólya bound gives a tail below $1.6\cdot10^{-10}$ at the start of this auxiliary window for 60,000 modes. Comparing the numerical trace to the three-term expression at the known invariants checks the solver's numerical error where the omitted tail is small. The actual reconstruction uses the fitted invariants from the physical window. [Analytic benchmark and derivation](../README.md)

## Results

| Method | Feasible reconstructions | Vertex error at most 2% | Median vertex error among feasible cases |
|---|---:|---:|---:|
| Strict inversion of the fitted coefficients | 51/100 | 36/100 | 0.194% |
| With geometry-constrained fallback | 100/100 | 41/100 | 3.54% |

The 49 strict failures include 29 corner coefficients below the equilateral minimum, 18 sets of incompatible invariants and 2 nonpositive area or perimeter estimates. The geometry-constrained fallback produced a triangle in each of those cases and added five passes. The 90th percentile vertex error with fallback was 22.1%.

| Area quartile | Area range | Strict passes | Passes with fallback |
|---|---:|---:|---:|
| Smallest 25 | 0.230–1.061 | 19/25 | 20/25 |
| Second 25 | 1.144–3.085 | 16/25 | 17/25 |
| Third 25 | 3.270–5.906 | 1/25 | 3/25 |
| Largest 25 | 6.006–11.086 | 0/25 | 1/25 |

At fixed time and mode count, a larger domain has a smaller $t_0\lambda_{60000}$ and a larger omitted tail. The results also show sensitivity to numerical eigenvalue error for some near-symmetric triangles. A small heat-trace residual alone gives limited information about vertex error.

The numerical frequency counts agreed exactly with the analytic counts through the solver cutoff on equilateral and $30^\circ$–$60^\circ$–$90^\circ$ controls. At the dataset median area, the fitted corner term differed from the corresponding exact-spectrum fit by $8.95\cdot10^{-8}$ and $4.84\cdot10^{-8}$, respectively. Seven dataset rows needed targeted spectral repairs. The final spectra show zero count disagreements on the checked overlaps. On the 100 unit-area audits, the median absolute corner coefficient error was $5.43\cdot10^{-7}$ and the largest was $1.71\cdot10^{-4}$. Frequency approximations can still affect the 2% decision near the threshold.

The main spectral batch ran for 46 minutes and 6 seconds on five local worker processes. Stitching, reconstruction, the unit-area audit and archive compression took 19 seconds. The six million stored float64 eigenvalues occupy 48 MB before compression and 37.83 MB in the compressed archive. [Per-case results](results.json) · [Compressed spectra](spectra.npz) · [Example reconstructions](example_reconstructions.png)

## Example of truncation error

Dataset row 900 has area 6.8115 and true angles of about $33.17^\circ$, $48.27^\circ$, and $98.56^\circ$. In the main window, $t_0\lambda_{60000}=11.13$. At $t_0$ the truncated trace falls about 0.079 below the three-term expression evaluated at the exact invariants. The fitted corner coefficient is $-0.573$ and the true value is $0.416$. Strict inversion rejects this estimate. Geometry-constrained fitting gives an isosceles triangle with 24.86% vertex error. The unit-area audit estimates the corner term within $8.2\cdot10^{-8}$, supporting truncation as the dominant error for this case in the physical window.

Keeping all 60,000 numerical eigenvalues and using 100 points in $[t_0,2t_0]$ gives the following sensitivity check on row 900.

| $t_0$ | $t_0 \lambda_{60000}$ | Fitted $\gamma$ | Vertex error |
|---:|---:|---:|---:|
| $1.0\cdot10^{-4}$ | 11.13 | $-0.572588$ | 24.86% |
| $2.0\cdot10^{-4}$ | 22.27 | $0.415918$ | 0.529% |
| $2.5\cdot10^{-4}$ | 27.83 | $0.415924$ | 0.00683% |

## Reproduction

Run from the repository root with Python, NumPy, SciPy and a C compiler. The first command compiles local Bessel interpolation. The spectral run writes one checkpoint per triangle and skips completed checkpoints for matching vertices.

```bash
solver=training/triangle_heat_trace/real_test_60000
scratch=results/triangle_heat_trace_real60000
mkdir -p "$scratch"
cc -O3 -shared -fPIC "$solver/interpolate.c" -o "$scratch/interpolate.dylib"
export DRUMS_TRIANGLE_INTERPOLATION_LIB="$PWD/$scratch/interpolate.dylib"
export PYTHONPATH="$PWD/training/triangle_heat_trace"
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python "$solver/solve_dataset.py" triangle_drums.npz "$scratch/checkpoints" --workers 5

python "$solver/repair_checkpoint.py" "$scratch/checkpoints/909.npz" "$scratch/909_repaired.npz" \
  --bands 808.5:811.5 --provenance "$scratch/909_repair_provenance.json"
python "$solver/scan_robust.py" triangle_drums.npz "$scratch/913_narrow.npz" \
  --index 913 --stop 445 --step .23 --phase .31 --pad 14
python "$solver/repair_checkpoint.py" "$scratch/checkpoints/913.npz" "$scratch/913_repaired.npz" \
  --low-scan "$scratch/913_narrow.npz" --bands 485:491 545:567 581:597 786:798 836:849 \
  --provenance "$scratch/913_repair_provenance.json"
python "$solver/repair_checkpoint.py" "$scratch/checkpoints/944.npz" "$scratch/944_repaired.npz" \
  --bands 281:286 494:499 554:575 799:809 834:841 \
  --provenance "$scratch/944_repair_provenance.json"
TRIANGLE_SPECTRUM_PYTHON="$(command -v python)" sh "$solver/repair_945.sh" \
  triangle_drums.npz "$scratch/checkpoints/945.npz" "$scratch/repair945"
python "$solver/repair_checkpoint.py" "$scratch/checkpoints/970.npz" "$scratch/970_repaired.npz" \
  --bands 687:690.2 --provenance "$scratch/970_repair_provenance.json"
python "$solver/repair_checkpoint.py" "$scratch/checkpoints/996.npz" "$scratch/996_repaired.npz" \
  --bands 71.8:75 735.7:739.1 818.7:822.4 --provenance "$scratch/996_repair_provenance.json"
python "$solver/repair_checkpoint.py" "$scratch/checkpoints/997.npz" "$scratch/997_repaired.npz" \
  --bands 479:484 --step .13 --phase .41 --provenance "$scratch/997_repair_provenance.json"

python "$solver/aggregate.py" triangle_drums.npz "$scratch/checkpoints" \
  --override "909=$scratch/909_repaired.npz" \
  --override "913=$scratch/913_repaired.npz" \
  --override "944=$scratch/944_repaired.npz" \
  --override "945=$scratch/repair945/945_final.npz" \
  --override "970=$scratch/970_repaired.npz" \
  --override "996=$scratch/996_repaired.npz" \
  --override "997=$scratch/997_repaired.npz" \
  --results "$scratch/results.json" --spectra "$scratch/spectra.npz"
```

The JSON contains per-case fit coefficients, vertex errors, eligibility for strict inversion, and numerical solver diagnostics. The compressed spectra archive contains `indices` and an array `spectra` with one row per completed triangle.

The targeted repairs independently recompute and restitch problematic frequency bands. The original checkpoints and their overlap diagnostics are retained. The bands were selected to resolve spectral discrepancies detected in the solver audit.

To reproduce the analytic control with the numerical solver:

```bash
python "$solver/scan_robust.py" triangle_drums.npz "$scratch/equilateral.npz" \
  --control equilateral
python "$solver/check_control.py" "$scratch/equilateral.npz" equilateral
```
