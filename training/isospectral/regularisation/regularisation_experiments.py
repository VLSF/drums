import argparse
import json
import os
import sys
import time

# Ensure parent directory (training/isospectral) and iso30k are on sys.path
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(_cur_dir, '..')))
sys.path.append(os.path.join(_cur_dir, 'iso30k'))
sys.path.append('iso30k')
sys.path.append('training/isospectral')
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import vmap
from common import (
    load_data,
    split,
    Features,
    relative_error,
    PERMS,
    N_V,
    N_TRAIN,
    N_VAL,
    N_TEST,
)
from metrics import evaluate
from drums.models.transformer import DrumTransformer


# ---------------------------------------------------------------------------
# Regularizer Definitions
# ---------------------------------------------------------------------------

def polygon_perimeter(p):
    """Total boundary perimeter of a polygon p (n, 2)."""
    edges = jnp.linalg.norm(p - jnp.roll(p, -1, axis=0), axis=-1)
    return jnp.sum(edges)


def polygon_edges_sorted(p):
    """Sorted edge lengths of a polygon p (n, 2)."""
    edges = jnp.linalg.norm(p - jnp.roll(p, -1, axis=0), axis=-1)
    return jnp.sort(edges)


def polygon_cos_angles_sorted(p, eps=1e-7):
    """Sorted cosines of interior angles between adjacent edges."""
    w1 = jnp.roll(p, 1, axis=0) - p
    w2 = jnp.roll(p, -1, axis=0) - p
    u1 = w1 / (jnp.linalg.norm(w1, axis=-1, keepdims=True) + eps)
    u2 = w2 / (jnp.linalg.norm(w2, axis=-1, keepdims=True) + eps)
    cosines = jnp.sum(u1 * u2, axis=-1)
    return jnp.sort(cosines)


def reg_perimeter_consistency(pred, targ):
    """Normalized perimeter discrepancy: ((Perim(pred) - Perim(targ)) / Perim(targ))^2."""
    p_pred = polygon_perimeter(pred)
    p_targ = polygon_perimeter(targ)
    return ((p_pred - p_targ) / p_targ) ** 2


def reg_edge_multiset_matching(pred, targ):
    """Normalized sorted edge length discrepancy."""
    e_pred = polygon_edges_sorted(pred)
    e_targ = polygon_edges_sorted(targ)
    return jnp.mean(((e_pred - e_targ) / e_targ) ** 2)


def reg_angle_cosines_matching(pred, targ):
    """Discrepancy in sorted cosines of interior corner angles."""
    c_pred = polygon_cos_angles_sorted(pred)
    c_targ = polygon_cos_angles_sorted(targ)
    return jnp.mean((c_pred - c_targ) ** 2)


# ---------------------------------------------------------------------------
# Training and Evaluation Harness
# ---------------------------------------------------------------------------

def train_and_eval(
    exp_name,
    reg_type,
    reg_weight,
    F_tr,
    V_tr,
    F_va,
    V_va,
    F_te,
    V_te,
    W_te,
    n_updates=30000,
    batch_size=128,
    lr=5e-4,
    seed=0,
):
    print("=" * 75)
    print(f"RUNNING: {exp_name} (reg_type={reg_type}, gamma={reg_weight})")
    print(f"  Setup: {n_updates} updates, batch {batch_size}, lr {lr}, seed {seed}")
    print("=" * 75)

    key = jax.random.PRNGKey(seed)
    k_model, k_train = jax.random.split(key)

    dim_in = F_tr.shape[1]
    model = DrumTransformer(N_V, 256, dim_in, k_model, L=5)

    schedule = optax.warmup_cosine_decay_schedule(0.0, lr, 500, n_updates, 1e-6)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=1e-4),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def loss_single(m, f, v):
        pred = m(f)
        data_loss = relative_error(v, pred)

        if reg_type == "none" or reg_weight == 0.0:
            penalty = 0.0
        elif reg_type == "perimeter":
            penalty = reg_perimeter_consistency(pred, v)
        elif reg_type == "edges":
            penalty = reg_edge_multiset_matching(pred, v)
        elif reg_type == "angles":
            penalty = reg_angle_cosines_matching(pred, v)
        elif reg_type == "both":
            penalty = 0.5 * reg_edge_multiset_matching(pred, v) + 0.5 * reg_angle_cosines_matching(pred, v)
        else:
            raise ValueError(f"Unknown reg_type: {reg_type}")

        return data_loss + reg_weight * penalty

    def batch_loss(m, f_b, v_b):
        return jnp.mean(vmap(loss_single, in_axes=(None, 0, 0))(m, f_b, v_b))

    compute_loss_and_grads = eqx.filter_value_and_grad(batch_loss)

    inds = np.random.RandomState(seed).randint(0, len(F_tr), (n_updates, batch_size))

    @eqx.filter_jit
    def step_fn(carry, b_inds):
        m_curr, opt_curr = carry
        loss_val, grads = compute_loss_and_grads(m_curr, F_tr[b_inds], V_tr[b_inds])
        updates, opt_curr = optimizer.update(grads, opt_curr, eqx.filter(m_curr, eqx.is_array))
        m_curr = eqx.apply_updates(m_curr, updates)
        return [m_curr, opt_curr], loss_val

    # Warmup compilation
    carry = [model, opt_state]
    carry, loss_init = step_fn(carry, inds[0])
    loss_init.block_until_ready()

    print(f"  Training on {len(F_tr)} samples...")
    t0 = time.time()
    for i, b_inds in enumerate(inds):
        carry, loss_val = step_fn(carry, b_inds)
        if (i + 1) % 5000 == 0:
            print(f"    Step {i + 1}/{n_updates}: loss = {float(loss_val):.4f}")
    model = carry[0]
    loss_val.block_until_ready()
    t1 = time.time()
    print(f"  Training completed in {t1 - t0:.2f}s ({n_updates / (t1 - t0):.1f} steps/s).")

    # Evaluation on 1,500 test drums
    print("  Evaluating on 1,500 test drums with iso30k protocol (sep_min=0.1)...")
    predict_all = eqx.filter_jit(lambda m, f: vmap(m)(f))
    preds_te = predict_all(model, F_te)
    preds_te.block_until_ready()

    summary, per_drum = evaluate(preds_te, V_te, W_te, sep_min=0.1)

    print(f"  RESULTS for {exp_name}:")
    print(f"    sharp (e_set/sep < 0.25):   {summary['sharp']:.1%}")
    print(f"    blurry (e_set/sep > 0.40):  {summary['blurry']:.1%}")
    print(f"    mean_set_over_sep:          {summary['mean_set_over_sep']:.4f}")
    print(f"    mid_over_sep:               {summary['mid_over_sep']:.4f}")
    print(f"    error_set (closer twin):    {summary['error_set']:.4f}")
    print(f"    error (truth):              {summary['error']:.4f}")
    print(f"    pick:                       {summary['pick']:.1%}")

    return {
        "exp_name": exp_name,
        "reg_type": reg_type,
        "reg_weight": float(reg_weight),
        "summary": summary,
        "training_time": float(t1 - t0),
        "final_loss": float(loss_val),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        choices=["all", "control", "run0", "run1", "run2", "run3", "run4"],
        default="all",
    )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--out", default="iso30k_matched_results.json")
    args = parser.parse_args()

    # Load ambiguous data
    
def find_data_file(filename):
    candidates = [
        filename,
        os.path.join(_cur_dir, '..', filename),
        os.path.join('iso30k', filename),
        os.path.join(_cur_dir, filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return filename

    ambiguous_path = find_data_file("isospectral_30k.npz")
    L, V, W, scale = load_data(ambiguous_path)
    L_tr, L_va, L_te = split(L)
    V_tr, V_va, V_te = split(V)
    W_tr, W_va, W_te = split(W)

    # Load single-branch control data
    single_path = find_data_file("isospectral_30k_single.npz")
    L_c, V_c, W_c, scale_c = load_data(single_path)
    L_tr_c, _, _ = split(L_c)
    V_tr_c, _, _ = split(V_c)

    # ratio_white features fitted on training set
    feats = Features("ratio_white", L_tr)
    F_tr = feats(L_tr)
    F_va = feats(L_va)
    F_te = feats(L_te)

    results = {}
    if os.path.exists(args.out):
        try:
            with open(args.out, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}

    # Control Baseline: Single-Branch (Unambiguous)
    if args.run in ["all", "control"]:
        res_ctrl = train_and_eval(
            "Control: Single-Branch (Perfect Regularization)",
            reg_type="none",
            reg_weight=0.0,
            F_tr=F_tr,
            V_tr=V_tr_c,
            F_va=F_va,
            V_va=V_va,
            F_te=F_te,
            V_te=V_te,
            W_te=W_te,
        )
        results["control_single_branch"] = res_ctrl
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # Run 0: Baseline (No Regularization, Ambiguous Arm)
    if args.run in ["all", "run0"]:
        res0 = train_and_eval(
            "Run 0: Baseline Pure Regression (No Reg)",
            reg_type="none",
            reg_weight=0.0,
            F_tr=F_tr,
            V_tr=V_tr,
            F_va=F_va,
            V_va=V_va,
            F_te=F_te,
            V_te=V_te,
            W_te=W_te,
        )
        results["run0_baseline"] = res0
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # Run 1: Perimeter Consistency
    if args.run in ["all", "run1"]:
        res1 = train_and_eval(
            f"Run 1: Perimeter Consistency (gamma={args.gamma})",
            reg_type="perimeter",
            reg_weight=args.gamma,
            F_tr=F_tr,
            V_tr=V_tr,
            F_va=F_va,
            V_va=V_va,
            F_te=F_te,
            V_te=V_te,
            W_te=W_te,
        )
        results["run1_perimeter"] = res1
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # Run 2: Edge Multiset Matching
    if args.run in ["all", "run2"]:
        res2 = train_and_eval(
            f"Run 2: Edge Multiset Matching (gamma={args.gamma})",
            reg_type="edges",
            reg_weight=args.gamma,
            F_tr=F_tr,
            V_tr=V_tr,
            F_va=F_va,
            V_va=V_va,
            F_te=F_te,
            V_te=V_te,
            W_te=W_te,
        )
        results["run2_edges"] = res2
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # Run 3: Angle Cosines Matching
    if args.run in ["all", "run3"]:
        res3 = train_and_eval(
            f"Run 3: Angle Cosines Matching (gamma={args.gamma})",
            reg_type="angles",
            reg_weight=args.gamma,
            F_tr=F_tr,
            V_tr=V_tr,
            F_va=F_va,
            V_va=V_va,
            F_te=F_te,
            V_te=V_te,
            W_te=W_te,
        )
        results["run3_angles"] = res3
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # Run 4: Combined Edges + Angles
    if args.run in ["all", "run4"]:
        res4 = train_and_eval(
            f"Run 4: Combined Edges + Angles (gamma={args.gamma})",
            reg_type="both",
            reg_weight=args.gamma,
            F_tr=F_tr,
            V_tr=V_tr,
            F_va=F_va,
            V_va=V_va,
            F_te=F_te,
            V_te=V_te,
            W_te=W_te,
        )
        results["run4_combined"] = res4
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 95)
    print("MATCHED ISO30K BENCHMARK SUMMARY (1,500 TEST DRUMS, sep > 0.1)")
    print("=" * 95)
    header = f"{'Experiment':<42} | {'sharp (<0.25)':<14} | {'blurry (>0.40)':<15} | {'e_set/sep':<10} | {'error_set':<10}"
    print(header)
    print("-" * len(header))
    for k, v in results.items():
        s = v["summary"]
        print(f"{v['exp_name']:<42} | {s['sharp']:<14.1%} | {s['blurry']:<15.1%} | {s['mean_set_over_sep']:<10.4f} | {s['error_set']:<10.4f}")


if __name__ == "__main__":
    main()
