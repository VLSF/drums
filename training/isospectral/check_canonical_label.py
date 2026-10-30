"""Checks that the three ways to get the canonical drum of a pair agree, on a small dataset made with the current code.

  (a) recovery  : the labelled triangle is recovered from the stored polygon (data generated without `triangle`),
  (b) triangle  : the angles are read from the stored `triangle`,
  (c) canonical : the generator draws the triangle with canonical_labelling=True, the canonical drum is domain 0.

The same seed gives the same random draws with and without canonical labelling, so for every sample the canonical drums obtained by
(a), (b) on the default data and by (c) must be the same polygon (up to the pose), and their spectra must agree up to the mesh noise.

    python check_canonical_label.py --n 120
"""
import argparse
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np

from drums.data.isospectral_drums import generate_isospectral_drums, corner_angle, _congruence_distance
from make_single_domain import keep_mask, recover_triangle

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=120)
p.add_argument("--seed", type=int, default=5)
a = p.parse_args()

plain = generate_isospectral_drums(a.n, seed=a.seed)
canon = generate_isospectral_drums(a.n, seed=a.seed, canonical_labelling=True)
ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f"{name}: {bool(cond)}")


label = lambda d, keep: np.where(keep[:, None, None], d["perimeters"], d["twin_perimeters"])

# default data: the rule with the recovered triangles and with the stored ones
keep_recovered = keep_mask(plain["perimeters"], plain["domain"])
keep_stored = keep_mask(plain["perimeters"], plain["domain"], plain["triangle"])
check("default data: recovery and the stored triangle give the same decision for all samples", np.array_equal(keep_recovered, keep_stored))
print(f"  the stored drum is canonical for {keep_stored.sum()} of {a.n} samples")
angles = lambda T: (corner_angle(T, 0), corner_angle(T, 2))
rec = [recover_triangle(p, d) for p, d in zip(plain["perimeters"], plain["domain"])]
check("default data: the recovered triangles have the angles of the stored ones",
      all(np.allclose(angles(r), angles(t), atol=1e-8) for r, t in zip(rec, plain["triangle"])))

# canonical data: the rule gives "the stored drum is canonical iff domain == 0" with any of the two ways
for name, kc in [("recovery", keep_mask(canon["perimeters"], canon["domain"])),
                 ("stored triangle", keep_mask(canon["perimeters"], canon["domain"], canon["triangle"]))]:
    check(f"canonical data, {name}: the stored drum is canonical iff domain == 0", np.array_equal(kc, canon["domain"] == 0))
check("canonical data: angle(T0) >= angle(T2) for all stored triangles",
      all(corner_angle(t, 0) >= corner_angle(t, 2) - 1e-12 for t in canon["triangle"]))
check("same seed: the domain draws and the triangles agree up to the relabelling",
      np.array_equal(plain["domain"], canon["domain"])
      and all(np.allclose(t, u) or np.allclose(t, u[[2, 1, 0]]) for t, u in zip(plain["triangle"], canon["triangle"])))

# the canonical drums themselves and their spectra
label_plain = label(plain, keep_stored)
label_canon = label(canon, canon["domain"] == 0)
d = np.array([_congruence_distance(x, y) for x, y in zip(label_plain, label_canon)])
check(f"the canonical drum is the same polygon in the three ways (max distance {d.max():.1e})", d.max() < 1e-8)
gap = np.max(np.abs(plain["spectra"] / canon["spectra"] - 1))
check(f"the spectra of the two datasets agree up to the mesh noise (max relative difference {gap:.1e})", gap < 3e-3)
assert ok, "canonical label check failed"
