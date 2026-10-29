Can One Hear the Shape of a Drum?

A tutorial on inverse problems: restoring the shape of a 2D drum from its spectrum.

## Installation

Clone the repository and install it from the clone in editable mode, so that
changes to the source are picked up without reinstalling:

```bash
git clone https://github.com/VLSF/drums.git
cd drums
pip install -e .
```

Use `pip install .` instead for a regular (non-editable) install. In a running
Python session or notebook, restart the kernel (or reload the module) after
editing the source.

## Dependencies

Python >= 3.9 and the following packages (installed automatically):

- `numpy`
- `scipy`
- `jax`
- `triangle` (Python wrapper for the Triangle mesh generator)
- `equinox` (neural networks in JAX)
- `optax` (optimisers, used in the model tests)

## Project structure

```
.
├── pyproject.toml            # package metadata and dependencies
├── README.md
├── training/
│   ├── classification/
│   │   ├── ring_transformer_classifier.py      # classify (drum, spectrum) pairs with the RingTransformer
│   │   ├── invariant_features_classifier.py    # the same with invariant distances to learned drums
│   │   ├── predicted_drums_classifier.py       # end-to-end: distances to drums predicted from the spectrum
│   │   └── triangle_dataset_experiments.md     # comparison of the two classifiers
│   └── regression/
│       ├── align_loss.py       # example training run, loss with aligned and augmented targets
│       ├── procrustes_loss.py  # example training run, Procrustes loss
│       ├── procrustes_all_orderings.py  # Procrustes loss over all vertex orderings, best schedule
│       └── triangle_dataset_experiments.md  # report on the triangle dataset experiments
└── src/
    └── drums/
        ├── data/
        │   ├── spectrum.py       # FEM solver: drum shape -> Laplacian eigenvalues
        │   ├── preprocessing.py  # align (standardise shape) and augment (vertex orderings)
        │   ├── discretization.md # how the finite-element discretization works
        │   ├── triangle_drums.py # random triangular drums dataset (perimeters and spectra)
        │   ├── quadrilateral_drums.py  # random convex / non-convex quadrilateral drums datasets
        │   ├── gww_drums.py      # perturbed isospectral Gordon-Webb-Wolpert drums dataset
        │   └── datasets_collection.py  # script that collects the named datasets and saves them
        └── models/
            ├── transformer.py    # DrumTransformer: eigenvalues -> vertices of a drum
            ├── ring_transformer.py  # RingTransformer: invariant/equivariant transformer on closed polygons
            └── ring_transformer.md  # description of the RingTransformer architecture
```

`get_drum_spectrum(drum_perimeter, N_eig, a)` meshes the polygon given by its
vertices, assembles the P1 finite-element stiffness and mass matrices, and returns
the lowest Dirichlet eigenvalues and eigenfunctions together with the mesh.

`align` shifts and rotates a drum so that the mean vertex is at the origin and the
first longest vertex points up; `augment` returns all cyclic permutations of the
vertex sequence and of its reverse.

Running `python src/drums/data/spectrum.py` checks the solver against the exact
spectra of a unit square and a 2x1 rectangle. Running
`python src/drums/data/preprocessing.py` tests `align` and `augment` and checks that
the spectrum is unchanged by shifts, rotations and cyclic reorderings.

The named datasets below are collected with `python src/drums/data/datasets_collection.py <name> <folder>`,
which saves `<name>.npz` in `<folder>`. The number of drums is set with `--N_samples` (1000 by default).

`generate_triangle_drums(n_samples, seed, N_eig, ...)` returns a dictionary with the
vertices (`perimeters`) and the lowest `N_eig` Dirichlet eigenvalues (`spectra`) of random
triangles whose angles are all at least 30 degrees. The mesh is scaled with the area of each
triangle. With `return_all=True` the dictionary also holds, as lists with one entry per drum, the
`eigenfunctions`, `triangulations` and `coordinates` of the meshes. Running `python src/drums/data/triangle_drums.py` tests the generator and the
solver on a right isosceles triangle with known spectrum.

`generate_quadrilateral_drums(n_samples, seed, N_eig, nonconvex_fraction, ...)` is the analogue
for quadrilaterals, with `perimeters` of shape `(n_samples, 4, 2)` and an extra boolean `is_convex`.
With `nonconvex_fraction=0` all drums are convex, with `0.5` each drum is non-convex with
probability one half. Four random points give either a convex quadrilateral (their hull) or a
non-convex one with one reflex vertex, and samples with an angle below 30 degrees or above 330 degrees,
a very short edge, or a vertex within 0.1 rad of straight are rejected, so the two classes are separated.
The named datasets `convex_quadrilateral_drums` and `mixed_quadrilateral_drums` are in
`drums.data.datasets_collection`. Running `python src/drums/data/quadrilateral_drums.py` tests the
samplers, the symmetries of the spectrum and the mesh convergence on non-convex drums.

`generate_gww_drums(n_samples, seed, N_eig, delta, ...)` returns a dataset based on the two isospectral
drums of Gordon, Webb and Wolpert (9 vertices each). Every drum is one of the two with probability one half
(`domain` is 0 or 1), and each coordinate of its vertices is shifted by independent uniform noise from
`[-delta, delta]`, `0 < delta < 0.5`; the noise is drawn again if the boundary would intersect itself. The drum is
then centred, rotated at random and its first vertex is chosen at random. `perimeters` has shape `(n_samples, 9, 2)`. The named
dataset is `gww_drums` (`delta=0.1`) in `drums.data.datasets_collection`. Running `python src/drums/data/gww_drums.py`
checks that the two noise-free drums are isospectral, the samplers and the mesh convergence.

`drums.models.transformer.DrumTransformer(N_v, N_f, N_eig, key, L, n_heads=4, mlp_ratio=4, n_drums=1)`
maps the (normalised) eigenvalues of one drum to its `N_v` vertices. The spectrum is
projected to one token, joined with `N_v` learnable vertex tokens, and processed by `L`
pre-norm transformer blocks; a linear decoder turns the vertex tokens into coordinates.
With `n_drums=K` the decoder gives `2 K` numbers per vertex, i.e. `K` drums at once, as an array
of shape `(N_v, 2 K)`.
Use `jax.vmap` for batches. `python src/drums/models/transformer.py` runs its tests.

The scripts in `training/regression/` are examples of training the transformer on the
triangle drums dataset (see `drums.data.datasets_collection`), e.g.
`python training/regression/align_loss.py --data triangle_drums.npz --out results`.
They additionally need `matplotlib`.
The results of the experiments, including the comparison of the losses, are in
`training/regression/triangle_dataset_experiments.md`.

`notebooks/flow_matching.ipynb` compares conditional flow matching with that baseline.
To train with a RingTransformer velocity predictor, call
`drums.models.flow_matching.main(seed=17, canonical=False, predictor="ring")`.
It returns `(model, history, spectra, vertices)`. Ring training randomly rotates,
reflects and reorders targets, preserving their shape and scale without canonicalisation.

`training/regression/aligned_flow_experiment.py` defaults to 20k aligned FM updates
(checkpoints: 5k/10k/20k) and full-dataset inference (Euler/50, 32 draws per drum).
`--features log_ratios` uses `log(lambda_1)` and `log(lambda_k/lambda_1)` (k=2..30),
standardized on train. Results and feature comparisons are in
`notebooks/FM_more_epochs_stats.ipynb`.

`drums.models.ring_transformer.RingTransformer(n, m, N_f, N_inv, key, L, n_heads=4, mlp_ratio=4)`
takes a closed polygon `X` of `n` vertices in the plane and a context vector `c` of length `m`.
Its forward pass `model(c, X)` returns an invariant vector (invariant to rotations, reflections,
translations and cyclic shifts of the vertices) and `n` corrections that are equivariant to the
same transformations. `model.forward_flip(c, X)` makes both symmetric also under reversion of
the vertex order. `python src/drums/models/ring_transformer.py` runs the symmetry tests.
The architecture is described in `src/drums/models/ring_transformer.md`.

`training/classification/` contains two examples that classify pairs (drum shape, spectrum) as
matching or non-matching, with the ring transformer, with invariant features of the drum, and with
distances to drums predicted from the spectrum (trained end-to-end).
They are run in the same way, e.g.
`python training/classification/ring_transformer_classifier.py --data triangle_drums.npz`.
The comparison of the two approaches is in
`training/classification/triangle_dataset_experiments.md`.
