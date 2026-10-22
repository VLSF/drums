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

`generate_triangle_drums(n_samples, seed, N_eig, ...)` returns a dictionary with the
vertices (`perimeters`) and the lowest `N_eig` Dirichlet eigenvalues (`spectra`) of random
triangles whose angles are all at least 30 degrees. The mesh is scaled with the area of each
triangle. With `return_all=True` the dictionary also holds, as lists with one entry per drum, the
`eigenfunctions`, `triangulations` and `coordinates` of the meshes. Running `python src/drums/data/triangle_drums.py` tests the generator and the
solver on a right isosceles triangle with known spectrum.

`drums.models.transformer.DrumTransformer(N_v, N_f, N_eig, key, L, n_heads=4, mlp_ratio=4)`
maps the (normalised) eigenvalues of one drum to its `N_v` vertices. The spectrum is
projected to one token, joined with `N_v` learnable vertex tokens, and processed by `L`
pre-norm transformer blocks; a linear decoder turns the vertex tokens into coordinates.
Use `jax.vmap` for batches. `python src/drums/models/transformer.py` runs its tests.

The scripts in `training/regression/` are examples of training the transformer on the
triangle drums dataset (see `drums.data.datasets_collection`), e.g.
`python training/regression/align_loss.py --data triangle_drums.npz --out results`.
They additionally need `matplotlib`.
The results of the experiments, including the comparison of the losses, are in
`training/regression/triangle_dataset_experiments.md`.

`drums.models.ring_transformer.RingTransformer(n, m, N_f, N_inv, key, L, n_heads=4, mlp_ratio=4)`
takes a closed polygon `X` of `n` vertices in the plane and a context vector `c` of length `m`.
Its forward pass `model(c, X)` returns an invariant vector (invariant to rotations, reflections,
translations and cyclic shifts of the vertices) and `n` corrections that are equivariant to the
same transformations. `model.forward_flip(c, X)` makes both symmetric also under reversion of
the vertex order. `python src/drums/models/ring_transformer.py` runs the symmetry tests.
The architecture is described in `src/drums/models/ring_transformer.md`.

`training/classification/` contains two examples that classify pairs (drum shape, spectrum) as
matching or non-matching, with the ring transformer and with invariant features of the drum.
They are run in the same way, e.g.
`python training/classification/ring_transformer_classifier.py --data triangle_drums.npz`.
The comparison of the two approaches is in
`training/classification/triangle_dataset_experiments.md`.
