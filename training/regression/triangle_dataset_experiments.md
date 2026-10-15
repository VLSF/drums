# Regression of triangle drums from their spectra: experiments

This report documents the training runs on the triangle drums dataset and what was tried
to improve them. All numbers are relative errors of the predicted vertices (defined below),
averaged over the training or the test drums.

## Setup

**Dataset** (`drums.data.datasets_collection`, name `triangle_drums`): 1000 random
triangles from `generate_triangle_drums`, with seed 0, all angles at least 30 degrees,
circumradius in [0.5, 3], and the lowest 30 Dirichlet eigenvalues of each (finite elements,
a mesh with 4000 elements per drum). The first 900 drums are used for training and the last 100
for testing.

**Model** (`drums.models.transformer.DrumTransformer`): $N\_v = 3$ vertices, $N\_f = 128$, $N\_{eig} = 30$,
$L = 5$ layers, 4 heads, MLP ratio 4, 996,226 parameters.

**Training**: 5000 updates, batch size 64 drawn from the training set, learning rate $10^{-3}$,
model initialised and batches drawn from a key seeded with 17 (and 18 for the second seed). The
targets are the vertices divided by the largest vertex norm in the training set. The input is the
raw spectrum unless stated otherwise. One run takes about one minute on the GPU.

**Errors.** For a predicted drum $p$ and the true one $t$, after centring both, the prediction is
rotated (or reflected) onto the target by the optimal orthogonal map (orthogonal Procrustes).
The error is the mean over vertices of $\lVert t\_i - p\_i \rVert / \lVert t\_i \rVert$. The errors differ in the vertex
orderings that are allowed, since the ordering of the vertices is not determined by the spectrum.
A model is trained with one of these errors as its loss, but it can be scored with any of them:

- **all-orderings error**: the smallest error over all $2 N\_v = 6$ orderings (3 cyclic shifts of
  the order and of its reverse). This is the error that corresponds to the actual problem, and it
  is the one used to compare the Procrustes runs.
- **two-orderings error**: the smallest of the given and the reversed order only. This is
  the loss of the first Procrustes script, `procrustes_loss.py`.
- **align error**: the prediction is aligned (centred, longest vertex up) and compared with
  all aligned and augmented targets. This is the loss of `align_loss.py`. Because the
  alignment is not optimal, this error is not directly comparable to the Procrustes ones.

A random untrained model has an error of about 0.8 for all losses.

## What was tried

1. The two original losses from the notebook (`align_loss.py`, `procrustes_loss.py`), with
   constant Adam and raw features.
2. **Input normalisation**: the logarithm of the eigenvalues, standardised for each
   eigenvalue index with the training mean and standard deviation.
3. **Optimiser**: AdamW with weight decay $10^{-4}$, 250 warm-up steps up to $10^{-3}$, cosine decay to
   $10^{-5}$, and gradient clipping at global norm 1.0 (against constant Adam with $10^{-3}$).
4. **All vertex orderings in the Procrustes loss** (instead of the given and reversed order).

Each setting was run with two seeds (17 and 18).

## Results

The run with seed 17, constant Adam and raw features reproduces the notebook exactly: align loss
train/test 0.1604 / 0.1534, Procrustes (two orderings) train/test 0.1726 / 0.1965.

### Trained with the align loss

The models are scored with the align error, which is also the loss they were trained on.

| Features | Optimiser | Train (seed 17 / 18) | Test (seed 17 / 18) | Test mean |
|---|---|---|---|---|
| raw | constant Adam | 0.1604 / 0.1657 | 0.1534 / 0.1630 | 0.1582 |
| raw | warm-up + cosine, AdamW, clip | 0.1428 / 0.1439 | 0.1513 / 0.1650 | 0.1582 |
| log + standardise | constant Adam | 0.1461 / 0.1550 | 0.1598 / 0.1605 | 0.1602 |
| log + standardise | warm-up + cosine, AdamW, clip | 0.1299 / 0.1317 | 0.1493 / 0.1601 | 0.1547 |

### Trained with the Procrustes loss on two orderings

The two tables below show the **same trained models**: four settings (features and optimiser),
two seeds each, all trained with the two-orderings Procrustes loss. They are not different runs
and they do not differ in any training parameter. The tables differ only in the error measure
used to score the models after training.

**Evaluated with the all-orderings error.** The best match over all 6 orderings is used,
including the cyclic shifts. This measures how good the predicted shape is regardless of how
the vertices are labelled.

| Features | Optimiser | Train (seed 17 / 18) | Test (seed 17 / 18) | Test mean |
|---|---|---|---|---|
| raw | constant Adam | 0.0532 / 0.0720 | 0.0509 / 0.0718 | 0.0614 |
| raw | warm-up + cosine, AdamW, clip | 0.0463 / 0.0441 | 0.0460 / 0.0407 | 0.0434 |
| log + standardise | constant Adam | 0.0709 / 0.0518 | 0.0677 / 0.0513 | 0.0595 |
| log + standardise | warm-up + cosine, AdamW, clip | 0.0432 / 0.0357 | 0.0431 / 0.0386 | 0.0409 |

**Evaluated with the two-orderings error.** Only the given and the reversed order are used,
which is what these models were trained on. The seed 17 values of the first setting (raw features,
constant Adam) are the numbers of the original notebook.

| Features | Optimiser | Train (seed 17 / 18) | Test (seed 17 / 18) | Test mean |
|---|---|---|---|---|
| raw | constant Adam | 0.1726 / 0.1751 | 0.1965 / 0.1912 | 0.1938 |
| raw | warm-up + cosine, AdamW, clip | 0.1573 / 0.1580 | 0.1880 / 0.1925 | 0.1902 |
| log + standardise | constant Adam | 0.1713 / 0.1713 | 0.1971 / 0.1910 | 0.1941 |
| log + standardise | warm-up + cosine, AdamW, clip | 0.1546 / 0.1482 | 0.1842 / 0.1883 | 0.1863 |

### Trained with the Procrustes loss on all orderings

Again the two tables show the **same trained models** (four settings, two seeds each, all trained
with the all-orderings Procrustes loss), scored with two different error measures.

**Evaluated with the all-orderings error**, which is the loss they were trained on:

| Features | Optimiser | Train (seed 17 / 18) | Test (seed 17 / 18) | Test mean |
|---|---|---|---|---|
| raw | constant Adam | 0.0263 / 0.0299 | 0.0266 / 0.0291 | 0.0278 |
| raw | warm-up + cosine, AdamW, clip | 0.0049 / 0.0063 | 0.0054 / 0.0067 | 0.0061 |
| log + standardise | constant Adam | 0.0198 / 0.0201 | 0.0206 / 0.0224 | 0.0215 |
| log + standardise | warm-up + cosine, AdamW, clip | 0.0054 / 0.0054 | 0.0058 / 0.0065 | 0.0061 |

**Evaluated with the two-orderings error.** The values are much larger than in the previous table. The loss ignores
the vertex labelling, so the network is free to choose it, and the prediction is most likely a
cyclic shift of the target in many cases (we did not check this directly).

| Features | Optimiser | Train (seed 17 / 18) | Test (seed 17 / 18) | Test mean |
|---|---|---|---|---|
| raw | constant Adam | 0.1786 / 0.1787 | 0.1947 / 0.1934 | 0.1941 |
| raw | warm-up + cosine, AdamW, clip | 0.2531 / 0.2547 | 0.2520 / 0.2538 | 0.2529 |
| log + standardise | constant Adam | 0.2534 / 0.2643 | 0.2510 / 0.2641 | 0.2576 |
| log + standardise | warm-up + cosine, AdamW, clip | 0.2395 / 0.2124 | 0.2495 / 0.2075 | 0.2285 |

## Observations

- **Allowing all orderings in the Procrustes loss is by far the most important change.** The
  all-orderings test error drops from about 0.05 - 0.07 (trained on two orderings, constant
  Adam) to 0.005 - 0.007 (all orderings, cosine schedule), roughly a factor of ten. The test
  error of 0.19 reported originally for the two-orderings Procrustes loss was mostly a penalty for
  the vertex labelling: for a triangle the starting vertex is arbitrary and the two-orderings
  loss cannot absorb cyclic shifts.
- **The optimiser schedule matters next.** With all orderings, the cosine schedule with AdamW and
  clipping reduces the test error from 0.027 to 0.006 for raw features. The training loss is still
  decreasing at the end of training (see the loss curve of
  `procrustes_all_orderings.py`), so longer runs may help further.
- **Log-normalising the input helps only a little.** With constant Adam it gives 0.021 instead of
  0.028 (all orderings), and with the cosine schedule the two choices are equal within the
  difference between seeds (0.0061 on average for both). It helps the early training (the loss
  after 100 steps is much lower), but not the final result.
- **There is no overfitting for the Procrustes runs**: train and test errors are almost equal.
- **The align loss is much worse (about 0.15 in its own error)** and improves only slightly
  with the schedule and the normalisation (0.153 / 0.163 to 0.149 / 0.160, which is within the
  difference between seeds). Its training error goes down to 0.13 while the test error stays,
  so it begins to overfit. A possible reason is that the alignment in this loss (the longest
  vertex up) is not a smooth function of the prediction, so the gradients are poor. We did not
  test this.
- **The predicted spectra match the true ones.** For the model of
  `procrustes_all_orderings.py` the spectrum of the predicted triangle, meshed with the
  same resolution as the dataset (4000 elements), coincides with the true spectrum of the
  drum for all 30 eigenvalues (sample 7 in the last panel of the figure). With the default
  mesh of `get_drum_spectrum` (`a = 0.001`) the predicted eigenvalues came out up to about 10%
  too high for the highest modes. This is most likely a discretisation error of the coarser
  mesh and not an error of the model. The older scripts use the default mesh in the spectrum
  plot, as the notebook did.

## Caveats

- Only one dataset (1000 triangles) and two seeds. Differences of about 0.01 in the align
  errors and about 0.001 in the all-orderings Procrustes errors are within the seed noise.
- The ablation runs were made with a throwaway script that is not part of the repository.
  The script `training/regression/procrustes_all_orderings.py` reproduces the best setting
  (all orderings, cosine schedule, raw features): train/test error 0.0049 / 0.0054 for seed 17.
- The errors are relative errors of the vertices, not of the spectrum. The spectrum of the
  predicted drums was only inspected on one sample.

## Ideas not tried yet

- Predicting a size-free shape and the size separately (the eigenvalues scale as $1 / \text{size}^2$).
- More training data, a longer schedule, or a larger model.
- Refining the predicted drum by gradient descent on the spectrum, using a differentiable
  eigenvalue solver.
- Fewer input eigenvalues (the highest modes are the least accurate).
- The same losses for polygons with more than three vertices, where the number of orderings
  grows as $2 N\_v$.
