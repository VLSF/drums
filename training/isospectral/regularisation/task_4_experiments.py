import argparse
import json
import os
import sys
import time

# Ensure parent directories (training/isospectral, drums root, iso30k) are on sys.path
_cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(_cur_dir, '..')))
sys.path.append(os.path.abspath(os.path.join(_cur_dir, '..', '..', '..')))
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
# JAX Differentiable GWW Gluing Functions
# ---------------------------------------------------------------------------

def safe_norm(v, axis=-1, eps=1e-12):
    """Numerically stable vector 2-norm differentiable everywhere."""
    return jnp.sqrt(jnp.sum(v ** 2, axis=axis) + eps)


def reflect_jax(triangle, s0, s1):
    """Reflects triangle across the edge connecting vertex s0 and s1."""
    e1 = triangle[s1] - triangle[s0]
    v = e1 / safe_norm(e1)
    vertex = 3 - s0 - s1
    p = triangle[vertex] - triangle[s0]
    reflected = 2 * jnp.dot(v, p) * v - p + triangle[s0]
    return triangle.at[vertex].set(reflected)


def get_GWW_I_boundary_jax(triangle):
    """Assembles GWW Drum I (9 vertices) from base triangle."""
    t0 = triangle
    t1 = reflect_jax(t0, 1, 2)
    t2 = reflect_jax(t1, 1, 0)
    t3 = reflect_jax(t1, 0, 2)  # triangles[-2] before append was t1
    t4 = reflect_jax(t3, 0, 1)
    t5 = reflect_jax(t4, 2, 1)
    t6 = reflect_jax(t5, 2, 0)
    return jnp.stack([
        t0[0], t2[1], t2[2], t2[0],
        t4[2], t6[1], t5[0], t5[1], t3[2]
    ])


def get_GWW_II_boundary_jax(triangle):
    """Assembles GWW Drum II (9 vertices) from base triangle."""
    t0 = triangle
    t1 = reflect_jax(t0, 0, 1)
    t2 = reflect_jax(t0, 1, 2)  # triangles[-2] before append was t0
    t3 = reflect_jax(t2, 0, 2)
    t4 = reflect_jax(t3, 0, 1)
    t5 = reflect_jax(t4, 0, 2)
    t6 = reflect_jax(t4, 2, 1)  # triangles[-2] before append was t4
    return jnp.stack([
        t1[0], t1[2], t1[1], t2[0],
        t5[1], t5[2], t6[0], t6[1], t3[2]
    ])


# ---------------------------------------------------------------------------
# Model Architectures
# ---------------------------------------------------------------------------

class GWW_I_Model(eqx.Module):
    transformer: DrumTransformer

    def __init__(self, key, N_f=256, dim=30, L=5):
        self.transformer = DrumTransformer(3, N_f, dim, key, L=L)

    def __call__(self, f):
        triangle = self.transformer(f)
        return get_GWW_I_boundary_jax(triangle)

    def get_triangle(self, f):
        return self.transformer(f)


class GWW_II_Model(eqx.Module):
    transformer: DrumTransformer

    def __init__(self, key, N_f=256, dim=30, L=5):
        self.transformer = DrumTransformer(3, N_f, dim, key, L=L)

    def __call__(self, f):
        triangle = self.transformer(f)
        return get_GWW_II_boundary_jax(triangle)

    def get_triangle(self, f):
        return self.transformer(f)


# ---------------------------------------------------------------------------
# Regularization Definitions (Variance and Non-Variance)
# ---------------------------------------------------------------------------

def drum_variance(p):
    """Vertex variance around centroid."""
    p_centered = p - jnp.mean(p, axis=0, keepdims=True)
    return jnp.sum(p_centered ** 2)


def reg_variance_relative(pred, targ):
    """Relative variance penalty as in regression_with_regularisation.ipynb."""
    return drum_variance(pred) / drum_variance(targ)


def triangle_area(t):
    v1 = t[1] - t[0]
    v2 = t[2] - t[0]
    return 0.5 * jnp.abs(v1[0] * v2[1] - v1[1] * v2[0])


def polygon_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * jnp.abs(jnp.sum(x * jnp.roll(y, -1) - y * jnp.roll(x, -1)))


def triangle_perim(t):
    return jnp.sum(safe_norm(t - jnp.roll(t, -1, axis=0)))


def polygon_perim(p):
    return jnp.sum(safe_norm(p - jnp.roll(p, -1, axis=0)))


def corner_angle_jax(t, i):
    a, b, c = t[i], t[(i + 1) % 3], t[(i + 2) % 3]
    u, v = b - a, c - a
    norm_u = safe_norm(u)
    norm_v = safe_norm(v)
    cos_val = jnp.clip(jnp.dot(u, v) / (norm_u * norm_v + 1e-12), -0.9999, 0.9999)
    return jnp.arccos(cos_val)


def reg_area(t, v):
    """Strategy 1: Exact Area Consistency 7 * Area(T) == Area(Y)."""
    a_pred = 7.0 * triangle_area(t)
    a_targ = polygon_area(v)
    return ((a_pred - a_targ) / (a_targ + 1e-8)) ** 2


def reg_perim(t, v):
    """Strategy 2: Exact Perimeter Consistency 3 * Perim(T) == Perim(Y)."""
    p_pred = 3.0 * triangle_perim(t)
    p_targ = polygon_perim(v)
    return ((p_pred - p_targ) / (p_targ + 1e-8)) ** 2


def reg_canonical(t):
    """Strategy 4: Canonical Orientation Penalty (theta0 >= theta2)."""
    theta0 = corner_angle_jax(t, 0)
    theta2 = corner_angle_jax(t, 2)
    return jnp.maximum(0.0, theta2 - theta0) ** 2


# ---------------------------------------------------------------------------
# Training and Evaluation
# ---------------------------------------------------------------------------

def train_and_eval(
    exp_name,
    model_type,
    reg_type,
    reg_weight,
    F_tr,
    V_tr,
    F_va,
    V_va,
    F_te,
    V_te,
    W_te,
    scale,
    n_updates=30000,
    batch_size=128,
    lr=5e-4,
    seed=0,
):
    print("=" * 80)
    print(f"RUNNING: {exp_name}")
    print(f"  Model: {model_type}, Reg: {reg_type} (gamma={reg_weight})")
    print(f"  Updates: {n_updates}, Batch: {batch_size}, LR: {lr}, Seed: {seed}")
    print("=" * 80)

    key = jax.random.PRNGKey(seed)
    k_model, k_train = jax.random.split(key)

    dim_in = F_tr.shape[1]
    if model_type == "gww_1":
        model = GWW_I_Model(k_model, N_f=256, dim=dim_in, L=5)
    elif model_type == "gww_2":
        model = GWW_II_Model(k_model, N_f=256, dim=dim_in, L=5)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    schedule = optax.warmup_cosine_decay_schedule(0.0, lr, 500, n_updates, 1e-6)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=1e-4),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def loss_single(m, f, v):
        triangle = m.get_triangle(f)
        if model_type == "gww_1":
            pred = get_GWW_I_boundary_jax(triangle)
        else:
            pred = get_GWW_II_boundary_jax(triangle)

        data_loss = relative_error(v, pred)

        if reg_type == "none" or reg_weight == 0.0:
            penalty = 0.0
        elif reg_type == "var_rel":
            penalty = reg_variance_relative(pred, v)
        elif reg_type == "area":
            penalty = reg_area(triangle, v)
        elif reg_type == "perimeter":
            penalty = reg_perim(triangle, v)
        elif reg_type == "canonical":
            penalty = reg_canonical(triangle)
        elif reg_type == "area_perim":
            penalty = reg_area(triangle, v) + reg_perim(triangle, v)
        elif reg_type == "combined_nonvar":
            penalty = reg_area(triangle, v) + reg_perim(triangle, v) + reg_canonical(triangle)
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

    print(f"  Training {model_type} on {len(F_tr)} samples...")
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

    # Compute variance ratio on test set
    var_preds = vmap(drum_variance)(preds_te)
    var_targets = vmap(drum_variance)(V_te)
    mean_var_ratio = float(jnp.mean(var_preds / var_targets))

    # Compute perimeter deficit
    def polygon_perimeter(p):
        return jnp.sum(safe_norm(p - jnp.roll(p, -1, axis=0)))
    p_true = vmap(polygon_perimeter)(V_te)
    p_pred = vmap(polygon_perimeter)(preds_te)
    mean_perim_deficit = float(jnp.mean((p_true - p_pred) / p_true))

    print(f"  RESULTS for {exp_name}:")
    print(f"    sharp (e_set/sep < 0.25):   {summary['sharp']:.1%}")
    print(f"    blurry (e_set/sep > 0.40):  {summary['blurry']:.1%}")
    print(f"    mean_set_over_sep:          {summary['mean_set_over_sep']:.4f}")
    print(f"    mid_over_sep:               {summary['mid_over_sep']:.4f}")
    print(f"    error_set (closer twin):    {summary['error_set']:.4f}")
    print(f"    error (truth):              {summary['error']:.4f}")
    print(f"    pick:                       {summary['pick']:.1%}")
    print(f"    mean Var(pred)/Var(target): {mean_var_ratio:.4f}")
    print(f"    mean perimeter deficit:     {mean_perim_deficit:.2%}")

    return {
        "exp_name": exp_name,
        "model_type": model_type,
        "reg_type": reg_type,
        "reg_weight": float(reg_weight),
        "summary": summary,
        "mean_var_ratio": mean_var_ratio,
        "mean_perim_deficit": mean_perim_deficit,
        "training_time": float(t1 - t0),
        "final_loss": float(loss_val),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        choices=[
            "all",
            "gww1_noreg",
            "gww1_var",
            "gww2_noreg",
            "gww2_var",
            "gww1_area",
            "gww1_perim",
            "gww1_canonical",
            "gww1_area_perim",
            "gww1_combined",
        ],
        default="all",
    )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--out", default="task_4_results.json")
    args = parser.parse_args()

    # Load ambiguous data
    
def find_data_file(filename):
    candidates = [
        filename,
        os.path.join(_cur_dir, '..', filename),
        os.path.join('iso30k', filename),
        os.path.join(_cur_dir, filename),
        os.path.join(_cur_dir, '..', '..', '..', '..', 'drums_experiments', 'iso30k', filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return filename

    data_path = find_data_file("isospectral_30k.npz")
    L, V, W, scale = load_data(data_path)
    L_tr, L_va, L_te = split(L)
    V_tr, V_va, V_te = split(V)
    W_tr, W_va, W_te = split(W)

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

    # Run 1: GWW I Model, No Regularization
    if args.run in ["all", "gww1_noreg"]:
        res1 = train_and_eval(
            "GWW-I Model (No Regularization)",
            model_type="gww_1",
            reg_type="none",
            reg_weight=0.0,
            F_tr=F_tr,
            V_tr=V_tr,
            F_va=F_va,
            V_va=V_va,
            F_te=F_te,
            V_te=V_te,
            W_te=W_te,
            scale=scale,
        )
        results["gww1_noreg"] = res1
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # Strategy 1: GWW I Model, Area Consistency
    if args.run in ["all", "gww1_area"]:
        res_area = train_and_eval(
            f"GWW-I Model (Area Consistency, gamma={args.gamma})",
            model_type="gww_1",
            reg_type="area",
            reg_weight=args.gamma,
            F_tr=F_tr,
            V_tr=V_tr,
            F_va=F_va,
            V_va=V_va,
            F_te=F_te,
            V_te=V_te,
            W_te=W_te,
            scale=scale,
        )
        results["gww1_area"] = res_area
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # Strategy 2: GWW I Model, Perimeter Consistency
    if args.run in ["all", "gww1_perim"]:
        res_perim = train_and_eval(
            f"GWW-I Model (Perimeter Consistency, gamma={args.gamma})",
            model_type="gww_1",
            reg_type="perimeter",
            reg_weight=args.gamma,
            F_tr=F_tr,
            V_tr=V_tr,
            F_va=F_va,
            V_va=V_va,
            F_te=F_te,
            V_te=V_te,
            W_te=W_te,
            scale=scale,
        )
        results["gww1_perim"] = res_perim
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # Strategy 4: GWW I Model, Canonical Orientation
    if args.run in ["all", "gww1_canonical"]:
        gamma_canon = 0.2 if args.gamma == 1.0 else args.gamma
        res_canon = train_and_eval(
            f"GWW-I Model (Canonical Orientation, gamma={gamma_canon})",
            model_type="gww_1",
            reg_type="canonical",
            reg_weight=gamma_canon,
            F_tr=F_tr,
            V_tr=V_tr,
            F_va=F_va,
            V_va=V_va,
            F_te=F_te,
            V_te=V_te,
            W_te=W_te,
            scale=scale,
        )
        results["gww1_canonical"] = res_canon
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # Area + Perimeter Consistency
    if args.run in ["all", "gww1_area_perim"]:
        res_ap = train_and_eval(
            f"GWW-I Model (Area + Perimeter, gamma={args.gamma})",
            model_type="gww_1",
            reg_type="area_perim",
            reg_weight=args.gamma,
            F_tr=F_tr,
            V_tr=V_tr,
            F_va=F_va,
            V_va=V_va,
            F_te=F_te,
            V_te=V_te,
            W_te=W_te,
            scale=scale,
        )
        results["gww1_area_perim"] = res_ap
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # Combined Non-Variance: Area + Perimeter + Canonical
    if args.run in ["all", "gww1_combined"]:
        res_comb = train_and_eval(
            f"GWW-I Model (Combined Non-Variance, gamma={args.gamma})",
            model_type="gww_1",
            reg_type="combined_nonvar",
            reg_weight=args.gamma,
            F_tr=F_tr,
            V_tr=V_tr,
            F_va=F_va,
            V_va=V_va,
            F_te=F_te,
            V_te=V_te,
            W_te=W_te,
            scale=scale,
        )
        results["gww1_combined"] = res_comb
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 110)
    print("TASK 4 SUMMARY: NON-VARIANCE REGULARIZATIONS ON GWW-I MODEL")
    print("=" * 110)
    header = f"{'Experiment':<44} | {'sharp (<0.25)':<14} | {'blurry (>0.40)':<15} | {'e_set/sep':<10} | {'error_set':<10} | {'Var Ratio':<10}"
    print(header)
    print("-" * len(header))
    for k, v in results.items():
        s = v["summary"]
        print(f"{v['exp_name']:<44} | {s['sharp']:<14.1%} | {s['blurry']:<15.1%} | {s['mean_set_over_sep']:<10.4f} | {s['error_set']:<10.4f} | {v.get('mean_var_ratio', 0.0):<10.4f}")


if __name__ == "__main__":
    main()
