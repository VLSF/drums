"""Metrics based on the distances to the isospectral twin.

For every test spectrum there are two drums, the truth y and its twin y', and the data do not tell which
one is the label. For a prediction p let
    e_truth = d(y, p), e_twin = d(y', p),  e_set = min(e_truth, e_twin),  sep = d(y, y'),
with d the relative vertex error (minimised over the 18 labelings and rotations/reflections).
Reference values (see `references`): a predictor that is exact but picks the twin at random has
e_truth = sep / 2 on average and e_set = 0 (sharp); the midpoint of the two drums has e_truth ~ sep / 2
and e_set ~ sep / 2 (blurry). So e_truth alone cannot tell them apart, e_set / sep does.

  sharp    fraction of drums with e_set / sep < 0.25: p is close to one of the two drums
  blurry   fraction of drums with e_set / sep > 0.40: p is far from both drums, about midway or worse
  mid      e_mid / sep, e_mid = d(midpoint of y and y', p): ~0 if p is the midpoint
  pick     fraction of the sharp predictions that are nearer to the truth than to the twin
           (0.5 means that the label cannot be inferred from the spectrum, as it should be)

Drums with sep < sep_min are excluded from the ratios (the twins are almost congruent there).
"""
import jax.numpy as jnp
import numpy as np
from jax import vmap

from common import relative_error, procrustes_mean


def per_drum(pred, truth, twin):
    e_truth = vmap(relative_error)(truth, pred)
    e_twin = vmap(relative_error)(twin, pred)
    sep = vmap(relative_error)(truth, twin)
    mid = vmap(lambda a, b: procrustes_mean(jnp.stack([a, b]), n_iter=3))(truth, twin)
    e_mid = vmap(relative_error)(mid, pred)
    return dict(e_truth=np.asarray(e_truth), e_twin=np.asarray(e_twin), sep=np.asarray(sep), e_mid=np.asarray(e_mid))


def summarize(m, sep_min=0.1):
    e_set = np.minimum(m["e_truth"], m["e_twin"])
    sel = m["sep"] > sep_min
    r = e_set[sel] / m["sep"][sel]
    sharp = r < 0.25
    return dict(
        n_separated=int(sel.sum()), mean_sep=float(m["sep"][sel].mean()),
        error=float(m["e_truth"].mean()), error_separated=float(m["e_truth"][sel].mean()),
        error_set=float(e_set[sel].mean()),
        sharp=float(sharp.mean()), blurry=float((r > 0.40).mean()),
        mean_set_over_sep=float(r.mean()), mid_over_sep=float((m["e_mid"][sel] / m["sep"][sel]).mean()),
        pick=float((m["e_truth"][sel][sharp] < m["e_twin"][sel][sharp]).mean()) if sharp.any() else float("nan"),
    )


def references(truth, twin, sep_min=0.1):
    """Numbers for the ideal sharp predictor that picks a twin at random, and for the midpoint."""
    sep = np.asarray(vmap(relative_error)(truth, twin))
    sel = sep > sep_min
    return dict(sharp_random_pick_error_separated=float(0.5 * sep[sel].mean()), mean_sep=float(sep[sel].mean()),
                fraction_separated=float(sel.mean()))


def evaluate(pred, truth, twin, sep_min=0.1):
    m = per_drum(pred, truth, twin)
    return summarize(m, sep_min), m
