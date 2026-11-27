# Can One Hear the Shape of a Drum?

Inverse problems: recovering a planar drum's shape from its Laplacian spectrum.
The project compares regression, flow matching, diffusion and classifier-free
guidance (CFG), including Conway pairs with two known isospectral solutions.

## Installation

From the repository root:

```bash
pip install -e .
```

Dependencies are listed in [pyproject.toml](pyproject.toml).
For the local Apple GPU environment, see [requirements-metal.txt](requirements-metal.txt).

## Reports

- [Regression and flow matching on isospectral drums](training/isospectral/isospectral_dataset_experiments.md)
- [Geometric invariant regularization on isospectral drums](training/isospectral/regularisation/report_regularisation_experiments.md)
- [GWW triangle-gluing inductive bias on isospectral drums](training/isospectral/regularisation/report_triangle_inductive_bias.md)
- [Conditional diffusion](training/isospectral/diffusion_dataset_experiments.md)
- [Diffusion with CFG](training/isospectral/cfg_diffusion_experiments.md)
- [Triangle regression](training/regression/triangle_dataset_experiments.md)
- [Triangle classification](training/classification/triangle_dataset_experiments.md)
- [Triangle shape optimization](training/optimisation/report_optimisation_triangle.md)
- [Triangle shape surrogate optimization](training/optimisation/report_optimisation_surrogate.md)
- [Comparison of shape optimization approaches](training/optimisation/report_comparative_optimisation.md)
- [MCMC and MALA for incomplete spectra on triangular domains](training/triangles_incomplete/report_mcmc_vs_mala.md)

Reports contain the results, figures, experiment settings and reproduction commands.

## Code

| Location | Purpose |
|---|---|
| `src/drums/data/` | Dataset generation and the forward FEM solver |
| `src/drums/models/` | Model definitions, losses and samplers |
| `training/isospectral/` | Conway training, inference and evaluation |
| `training/regression/`, `training/classification/` | Other training experiments |
| `training/optimisation/` | Direct optimization through the spectrum solver |
| `training/triangles_incomplete/` | MCMC and MALA sampling from incomplete Dirichlet spectra |
| `notebooks/` | Exploratory analysis |

[src/drums/models/diffusion.py](src/drums/models/diffusion.py) implements diffusion;
[training/isospectral/diffusion.py](training/isospectral/diffusion.py) runs it on Conway data.
See also the [RingTransformer architecture](src/drums/models/ring_transformer.md)
and [FEM discretization](src/drums/data/discretization.md).

## Conway diffusion / CFG

Generate the dataset:

```bash
python training/isospectral/generate_data.py \
  --n 30000 --seed 2026 --workers 6 \
  --out training/isospectral/isospectral_30k.npz
```

Train a joint conditional/unconditional model in a new output directory:

```bash
python training/isospectral/diffusion.py \
  --data training/isospectral/isospectral_30k.npz --out results/cfg_main_ratio \
  --features ratio --coordinate-scale rms --time-features 32 --ema-decay 0.999 \
  --condition-presence --cond-drop-prob 0.1 --train-only
```

Defaults are 40,000 updates, width 256, three layers and batch size 128.
Omit the two conditioning flags for conditional-only training.
Use `--stop-after` and `--resume` for staged training, as shown in the CFG report.

Sample and evaluate on validation data before choosing guidance:

```bash
python training/isospectral/diffusion.py --out results/cfg_main_ratio \
  --eval-only --eval-split validation --n-eval 200 --weights ema \
  --sampler ddim --sampling-steps 200 --eta 1 --guidance-scale 1.25
```

For the final test evaluation, use `--eval-split test --n-eval 1500`.
With a jointly trained checkpoint, `--guidance-scale 0` samples the prior,
`1` uses ordinary conditioning, and values above `1` apply CFG with two predictor
calls per step. Saved outputs include samples, metrics and sampler settings.
