# Classification of matching drum and spectrum pairs: triangle dataset experiments

This report compares three approaches that classify a pair (drum shape, spectrum) as matching or
non-matching on the triangle drums dataset. All the numbers below were produced by the scripts
`ring_transformer_classifier.py`, `invariant_features_classifier.py` and
`predicted_drums_classifier.py` in this folder.

## Setup

**Data.** The dataset `triangle_drums` (see `drums.data.datasets_collection`): 1000 triangles,
each with its vertices and the lowest 30 Dirichlet eigenvalues. The first 900 drums are used for
training and the last 100 for testing. The vertices are divided by the largest vertex norm of the
training drums. The spectra are used as they are, without normalisation.

**Pairs.** Every drum is paired with its own spectrum (label 1, matching) and with the spectrum
of another random drum of the same split (label 0, non-matching), so the classes are balanced.
The other drum is drawn with a random permutation without fixed points, so that a drum is never
paired with its own spectrum by chance. This gives 1800 training pairs and 200 test pairs.

**Models.** All models return one number $r = N\_2 - N\_1$, and the probability that the pair
matches is $p(\text{same}) = \sigma(-r)$, where $\sigma$ is the sigmoid function. The loss is the
negative log likelihood of the labels, which is computed from $\log \sigma(\pm r)$ for numerical
stability.

- **Ring transformer** (`ring_transformer_classifier.py`): the `RingTransformer` of
  `drums.models.ring_transformer` with the spectrum as the context vector and the vertices of
  the drum as the polygon, $N\_f = 128$, $L = 4$ layers, 4 heads, $N\_{\text{inv}} = 1$. The scalar
  $r$ is its invariant output of the pass with the flip. The model is invariant to rotations,
  reflections, translations, cyclic shifts and the reversion of the vertices by construction.
- **Invariant features** (`invariant_features_classifier.py`): the drum is described only by its
  Procrustes distances to 100 learnable reference drums. The distance is minimised over all
  $2 N\_v = 6$ vertex orderings, so it is invariant to all the symmetries. The 100 distances are
  concatenated with the spectrum and processed by an MLP with $L = 6$ residual blocks of width
  $N\_f = 128$.
- **Predicted drums, end-to-end** (`predicted_drums_classifier.py`): a `DrumTransformer` predicts
  $K$ drums at once from the spectrum (its decoder gives $2K$ numbers for every vertex). The
  relative Procrustes distances between the given drum and the $K$ predicted drums are computed
  (minimised over all vertex orderings, relative to the size of the given drum), and a small
  scorer acts on their logarithms: either a linear layer, or one hidden layer with 16 units. The
  whole model is trained once on the pair labels with the loss above. The predicted drums get no
  direct supervision from the true drums, only the gradient of the classification loss.
  $N\_f = 128$, $L = 5$ layers.

**Training.** Batch size 64 drawn from the training pairs, AdamW with weight decay $10^{-4}$,
warm-up for 250 steps to the learning rate $10^{-3}$, cosine decay to $10^{-5}$, gradient clipping
at global norm 1.0, single precision. The ring transformer is trained for 5000 updates and the
invariant features model for 3000 updates, as in the notebooks. Because the budgets differ, the
invariant features model was also trained for 5000 updates. The predicted drums model is trained in the same way,
for 5000 updates and, in one configuration, for 10000 updates. The ring transformer and the invariant
features configurations were run with 11 seeds (0 to 9 and the default seed 176 of the scripts), and
the predicted drums configurations with 5 seeds (0 to 4). The seed changes the initialisation of
the model, the batches and the non-matching pairs. The split into training and test drums is
fixed.

**Metrics.**

- *Train acc.* and *Test acc.*: accuracy on the balanced training and test pairs defined above.
  The test set has only 200 pairs, so a single test accuracy has a resolution of 0.5% and is
  noisy.
- *All test pairs*: the spectrum of each of the 100 test drums is paired with every test drum.
  This gives 100 matching and 9900 non-matching pairs. We report the accuracy on the matching
  pairs, the accuracy on the non-matching pairs, their mean (balanced accuracy) and the area
  under the ROC curve (ROC AUC, the probability that a matching pair gets a higher score than a
  non-matching pair). These numbers use all pairs and are much less noisy than the accuracy on
  200 pairs.

## Results

Mean ± standard deviation over the seeds.

| Model | Seeds | Params | Updates | Train acc. | Test acc. (200 pairs) | Acc. matching | Acc. non-matching | Balanced acc. | ROC AUC |
|---|---|---|---|---|---|---|---|---|---|
| Ring transformer | 11 | 798,436 | 5000 | 0.9883 ± 0.0024 | 0.9814 ± 0.0087 | 0.9991 ± 0.0030 | 0.9658 ± 0.0040 | 0.9824 ± 0.0029 | 0.9912 ± 0.0018 |
| Invariant features, 3000 updates (notebook) | 11 | 809,561 | 3000 | 0.9524 ± 0.0024 | 0.9418 ± 0.0182 | 0.9855 ± 0.0052 | 0.8928 ± 0.0056 | 0.9391 ± 0.0039 | 0.9675 ± 0.0031 |
| Invariant features, 5000 updates (matched) | 11 | 809,561 | 5000 | 0.9613 ± 0.0051 | 0.9455 ± 0.0182 | 0.9836 ± 0.0067 | 0.9118 ± 0.0107 | 0.9477 ± 0.0064 | 0.9742 ± 0.0044 |
| Predicted drums, K = 1, linear scorer | 5 | 996,228 | 5000 | 0.5418 ± 0.0945 | 0.5580 ± 0.1192 | 0.5040 ± 0.3914 | 0.5991 ± 0.2460 | 0.5516 ± 0.1144 | 0.6618 ± 0.1675 |
| Predicted drums, K = 4, linear scorer | 5 | 997,005 | 5000 | 0.7553 ± 0.1791 | 0.7720 ± 0.1862 | 0.8120 ± 0.3926 | 0.6938 ± 0.2415 | 0.7529 ± 0.1783 | 0.8331 ± 0.1939 |
| Predicted drums, K = 4, MLP scorer | 5 | 997,097 | 5000 | 0.8924 ± 0.0140 | 0.9010 ± 0.0282 | 0.9680 ± 0.0342 | 0.8007 ± 0.0412 | 0.8843 ± 0.0207 | 0.9452 ± 0.0158 |
| Predicted drums, K = 4, MLP scorer, 10000 updates | 5 | 997,097 | 10000 | 0.9173 ± 0.0991 | 0.9300 ± 0.0880 | 1.0000 ± 0.0000 | 0.8376 ± 0.1860 | 0.9188 ± 0.0930 | 0.9579 ± 0.0635 |

The ring transformer is better than the invariant features model for every seed, both in the
balanced accuracy over all test pairs (minimum of the ring transformer 0.9757 against the maximum
0.9562 of the invariant features model trained for 5000 updates) and in the ROC AUC (0.9880 against
0.9814). The accuracy on the 200 test pairs of one run is the least reliable number: its standard
deviation over seeds is 0.9% for the ring transformer and 1.8% for the invariant features model.

### Predicted drums, end to end

The end-to-end predicted drums model is clearly worse than the other two, and its results depend
strongly on the seed. Balanced accuracy / ROC AUC over all test pairs for every seed:

| Seed | K = 1, linear | K = 4, linear | K = 4, MLP | K = 4, MLP, 10000 updates |
|---|---|---|---|---|
| 0 | 0.7273 / 0.7846 | 0.4987 / 0.5002 | 0.8704 / 0.9377 | 0.7571 / 0.8451 |
| 1 | 0.5995 / 0.8950 | 0.8396 / 0.9068 | 0.8849 / 0.9464 | 0.9958 / 0.9994 |
| 2 | 0.4972 / 0.5570 | 0.9708 / 0.9996 | 0.8953 / 0.9624 | 0.9378 / 0.9809 |
| 3 | 0.4358 / 0.5261 | 0.7836 / 0.8466 | 0.8591 / 0.9227 | 0.9502 / 0.9814 |
| 4 | 0.4980 / 0.5460 | 0.6717 / 0.9124 | 0.9120 / 0.9568 | 0.9531 / 0.9827 |

### Which non-matching pairs are difficult

The non-matching pairs are ordered by the relative distance of the two spectra,
$\lVert \lambda^{(i)} - \lambda^{(j)} \rVert / \lVert \lambda^{(i)} \rVert$, and split into three equal
groups. The table gives the accuracy on the non-matching pairs of each group (mean over the seeds).

| Model | Closest third | Middle third | Farthest third |
|---|---|---|---|
| Ring transformer | 0.8998 | 0.9979 | 0.9997 |
| Invariant features, 3000 updates (notebook) | 0.7151 | 0.9702 | 0.9930 |
| Invariant features, 5000 updates (matched) | 0.7582 | 0.9809 | 0.9963 |
| Predicted drums, K = 1, linear scorer | 0.5633 | 0.3972 | 0.8368 |
| Predicted drums, K = 4, linear scorer | 0.4307 | 0.7702 | 0.8803 |
| Predicted drums, K = 4, MLP scorer | 0.4290 | 0.9852 | 0.9879 |
| Predicted drums, K = 4, MLP scorer, 10000 updates | 0.6196 | 0.8959 | 0.9973 |
| mean relative spectral distance | 0.27 | 0.77 | 6.12 |

For the ring transformer and the invariant features, almost all errors are made on the third of
the non-matching pairs with the most similar spectra, and in the other two thirds they are nearly
perfect. The predicted drums models with a linear scorer also fail in the other thirds. Even in the closest third the mean
relative distance between the spectra of the two drums is large (0.27), so these negative
examples are easy compared with a drum with almost the same spectrum.

### Per seed

Balanced accuracy / ROC AUC over all test pairs.

| Seed | Ring transformer | Invariant, 3000 updates | Invariant, 5000 updates |
|---|---|---|---|
| 0 | 0.9827 / 0.9907 | 0.9419 / 0.9661 | 0.9431 / 0.9710 |
| 1 | 0.9830 / 0.9915 | 0.9438 / 0.9686 | 0.9469 / 0.9745 |
| 2 | 0.9757 / 0.9906 | 0.9390 / 0.9608 | 0.9393 / 0.9667 |
| 3 | 0.9858 / 0.9911 | 0.9387 / 0.9698 | 0.9550 / 0.9814 |
| 4 | 0.9814 / 0.9901 | 0.9419 / 0.9650 | 0.9562 / 0.9799 |
| 5 | 0.9836 / 0.9922 | 0.9359 / 0.9682 | 0.9559 / 0.9763 |
| 6 | 0.9841 / 0.9933 | 0.9324 / 0.9657 | 0.9440 / 0.9766 |
| 7 | 0.9852 / 0.9940 | 0.9409 / 0.9692 | 0.9403 / 0.9707 |
| 8 | 0.9841 / 0.9925 | 0.9326 / 0.9675 | 0.9489 / 0.9761 |
| 9 | 0.9789 / 0.9880 | 0.9407 / 0.9724 | 0.9529 / 0.9731 |
| 176 | 0.9825 / 0.9894 | 0.9423 / 0.9692 | 0.9425 / 0.9697 |

## Comparison with the notebooks

The notebooks report one accuracy for all 2000 pairs of the training and the test set together.
Running the code of the notebooks gave 0.9855 (ring transformer, notebook: 0.9885) and 0.949
(invariant features, notebook: 0.9425), so the notebooks are reproduced to within 0.5%.
The scripts differ from the notebooks in the following way:

- The accuracy is computed separately for the training pairs, the test pairs and all test pairs.
- The non-matching pairs use a random permutation without fixed points. In the notebooks a
  random permutation can leave a drum in place, which then gives a matching pair with the label 0
  (about one such pair is expected in each split).
- The scale of the drums is computed from the raw vertices. The notebooks centre the drums over the
  wrong axis before computing the scale, so their scale is slightly different. The models
  centre the drums themselves.
- Training is in single precision. The notebooks ran in double precision only as a side effect of
  importing `drums.data.spectrum`.

## Observations

- **The ring transformer is clearly better.** The balanced accuracy over all test pairs is about
  0.98 against 0.94 to 0.95, and the ROC AUC is 0.991 against 0.967 to 0.974, with no overlap
  between the seeds. Giving the invariant features model the same number of updates helps it
  a little (from 0.939 to 0.948 in the balanced accuracy) but does not close the gap. The
  numbers of parameters are about the same (0.80 and 0.81 million).
- **Both models separate non-matching pairs with clearly different spectra almost perfectly.**
  The remaining errors are on non-matching pairs with the most similar spectra, and there the
  ring transformer is much better (0.90 against 0.72 to 0.76).
- **The matching pairs are almost always recognised.** The accuracy on them is 0.999 for the
  ring transformer and 0.984 to 0.986 for the invariant features model, so most of the errors
  are false matches.
- **The training accuracy is close to the test accuracy** (0.988 against 0.981 for the ring
  transformer), so there is no sign of overfitting. With only 100 test drums the test
  accuracy on 200 pairs should not be used to rank the models.
- **The end-to-end predicted drums model is worse and much less stable.** With a scorer with
  one hidden layer and K = 4 the balanced accuracy over all test pairs is 0.8843 ± 0.0207, and 0.9188 ± 0.0930 with
  10000 updates, against 0.982 ± 0.003 for the ring transformer. One run out of 20 (K = 4, MLP
  scorer, 10000 updates, seed 1: 0.996, ROC AUC 0.999) is better than the ring transformer. The
  spread between seeds is large: with 10000 updates the balanced accuracy is between 0.757 to 0.996, and
  with the linear scorer and K = 4 between 0.499 to 0.971.
- **A linear scorer is not enough in practice.** With K = 1 the balanced accuracy is 0.5516 ± 0.1144, close
  to chance in four of the five seeds: the scorer sees one distance, and the only thing that
  moves the predicted drum towards the true drum is the classification loss. With K = 4 and a
  linear scorer it is better on average (0.7529 ± 0.1783), but unstable. A scorer with one hidden layer is
  clearly better and much more stable (0.8843 ± 0.0207). It can combine the K distances nonlinearly, for
  example it can act on the smallest of them. More updates helped (0.9188 ± 0.0930 for 10000 updates), and
  the loss was still decreasing, but we did not run longer.
- **The K predicted drums are only loosely related to the true drum.** The relative Procrustes
  distance from the closest of the K predicted drums to the true drum on the test drums is
  typically between 0.14 and 0.37 (0.025 in the best run), and the mean over the K drums is
  often above 1. A regression trained directly on the true drums reaches about 0.005 (see the
  regression report). The predicted drums act as reference shapes conditioned on the spectrum,
  as the learned reference drums of the invariant features model, rather than as
  reconstructions of the drum. Among the runs with the MLP scorer we did not find a relation
  between this error and the accuracy.

## Caveats

- One dataset of 1000 triangles with a fixed split of 900 and 100 drums.
- The predicted drums model has only 5 seeds per configuration, and its hyperparameters
  (the number of drums K, the size of the scorer, the learning rate, the number of updates) were
  not tuned. Its high variance means that these numbers are rough.
- The non-matching examples are random pairs of different drums. They are mostly easy, since the
  eigenvalues scale with the size of the drum, and they do not test whether the model can tell
  apart drums with almost the same spectrum.
- The hyperparameters were taken from the notebooks and not tuned. The number of reference drums
  of the invariant features model (100) was not varied. A larger model or a longer training could
  improve both models.
- The experiments over seeds were run with a throwaway script, which is not part of the
  repository. The scripts `ring_transformer_classifier.py` and
  `invariant_features_classifier.py` have a function `run` that returns the same numbers for one
  seed.

## Ideas not tried yet

- Harder non-matching pairs: for each drum use the spectrum of the drum with the most similar
  spectrum.
- Vary the number of the reference drums of the invariant features model.
- For the predicted drums model: longer training and a tuned learning rate, a scorer that acts
  on the smallest of the K distances, and initialising the `DrumTransformer` from a regression
  on the true drums before the end-to-end training.
- More data, and polygons with more than three vertices, where the number of vertex orderings grows
  as $2 N\_v$.
