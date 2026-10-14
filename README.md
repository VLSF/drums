Can One Hear the Shape of a Drum?

A tutorial on inverse problems: restoring the shape of a 2D drum from its spectrum.

## Installation

Clone the repository and install it from the clone in editable mode, so that
changes to the source are picked up without reinstalling:

```bash
git clone <repository-url>
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

## Project structure

```
.
├── pyproject.toml            # package metadata and dependencies
├── README.md
└── src/
    └── drums/
        └── data/
            └── spectrum.py   # FEM solver: drum shape -> Laplacian eigenvalues
```

`get_drum_spectrum(drum_perimeter, N_eig, a)` meshes the polygon given by its
vertices, assembles the P1 finite-element stiffness and mass matrices, and returns
the lowest Dirichlet eigenvalues and eigenfunctions together with the mesh.

Running `python src/drums/data/spectrum.py` checks the solver against the exact
spectra of a unit square and a 2x1 rectangle.
