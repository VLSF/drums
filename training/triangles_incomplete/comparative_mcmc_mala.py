"""
Comparative benchmark between Plain MCMC (Random Walk Metropolis)
and MALA (Metropolis-Adjusted Langevin Algorithm) for Dirichlet Laplace
spectrum inversion on triangular domains.
"""

import time
import jax
import jax.numpy as jnp
from jax import random, vmap

import MCMC_search_for_spectrum as rw_mcmc
import MALA_search_for_spectrum as mala_mcmc


def run_comparison():
    master_key = random.PRNGKey(42)
    k_targets, k_inits = random.split(master_key, 2)

    num_samples = 5
    target_keys = random.split(k_targets, num_samples)
    init_keys = random.split(k_inits, num_samples)

    print("=" * 80)
    print("BENCHMARK: PLAIN MCMC (RANDOM WALK) vs. MALA (LANGEVIN DYNAMICS)")
    print("=" * 80)
    print(f"Generating {num_samples} reference targets from invariant prior...")

    target_triangles = jax.jit(vmap(lambda k: rw_mcmc.sample_prior_mcmc(k, N_steps=400, sigma_p=0.4, sigma_t=0.08)))(target_keys)
    target_spectra = jax.jit(vmap(lambda t: rw_mcmc.get_triangle_spectrum(t, N=35)[:2]))(target_triangles)

    results = []

    for i in range(num_samples):
        target_spec = target_spectra[i]
        log_target = jnp.log(target_spec)
        init_key = init_keys[i]

        t_area, t_s, t_r, t_R = rw_mcmc.get_invariants(target_triangles[i])

        print(f"\n--- Target {i+1}/{num_samples} ---")
        print(f"Target Spectrum: [{target_spec[0]:.4f}, {target_spec[1]:.4f}]")
        print(f"Target Invariants: Area = {t_area:.4f}, s = {t_s:.4f}, inradius r = {t_r:.4f}")

        # ----------------------------------------------------------------------
        # 1. Plain Random Walk MCMC
        # ----------------------------------------------------------------------
        t0 = time.time()
        T_rw, S_rw = rw_mcmc.test_run(
            log_target,
            init_key,
            sigma_s=0.1,
            sigma_t=0.08,
            sigma_p=0.4,
            N_steps=1000
        )
        t_rw = time.time() - t0

        rw_errors = jnp.linalg.norm(jnp.exp(S_rw) / target_spec.reshape(1, -1) - 1.0, axis=1)
        # Compute acceptance rate by counting state changes
        diffs_rw = jnp.linalg.norm(T_rw[1:] - T_rw[:-1], axis=(1, 2))
        rw_acc_rate = float(jnp.mean(diffs_rw > 1e-6))

        # Find steps to reach relative error < 0.20
        below_20_rw = jnp.where(rw_errors < 0.20)[0]
        step_burnin_rw = int(below_20_rw[0]) if len(below_20_rw) > 0 else 1000

        rw_post_s, rw_post_A, rw_post_r, rw_post_R = vmap(rw_mcmc.get_invariants)(T_rw[500:])

        # ----------------------------------------------------------------------
        # 2. MALA
        # ----------------------------------------------------------------------
        t0 = time.time()
        T_mala, acc_mala, U_mala = mala_mcmc.test_run(
            log_target,
            init_key,
            tau=4e-4,
            max_drift=0.05,
            sigma_s=0.1,
            sigma_p=0.4,
            N_steps=1000
        )
        t_mala = time.time() - t0

        mala_spectra = jax.jit(vmap(lambda t: mala_mcmc.get_triangle_spectrum(t, N=35)[:2]))(T_mala)
        mala_errors = jnp.linalg.norm(mala_spectra / target_spec.reshape(1, -1) - 1.0, axis=1)
        mala_acc_rate = float(jnp.mean(acc_mala))

        below_20_mala = jnp.where(mala_errors < 0.20)[0]
        step_burnin_mala = int(below_20_mala[0]) if len(below_20_mala) > 0 else 1000

        m_post_s, m_post_A, m_post_r, m_post_R = vmap(mala_mcmc.get_invariants)(T_mala[500:])

        print(f"  [Random Walk] Time: {t_rw:.2f}s | Acc: {rw_acc_rate:.1%} | Burn-in steps (<20% err): {step_burnin_rw:4d} | Final err: {rw_errors[-1]:.4f}")
        print(f"  [MALA       ] Time: {t_mala:.2f}s | Acc: {mala_acc_rate:.1%} | Burn-in steps (<20% err): {step_burnin_mala:4d} | Final err: {mala_errors[-1]:.4f}")
        print(f"  [Error at step 10] RW: {rw_errors[10]:.4f}  vs  MALA: {mala_errors[10]:.4f}")
        print(f"  [Area Recovery   ] True: {t_area:.4f} | RW: {jnp.mean(rw_post_A):.4f} +/- {jnp.std(rw_post_A):.4f} | MALA: {jnp.mean(m_post_A):.4f} +/- {jnp.std(m_post_A):.4f}")
        print(f"  [Inradius Recovery]True: {t_r:.4f} | RW: {jnp.mean(rw_post_r):.4f} +/- {jnp.std(rw_post_r):.4f} | MALA: {jnp.mean(m_post_r):.4f} +/- {jnp.std(m_post_r):.4f}")

        results.append({
            "sample": i,
            "t_rw": t_rw,
            "t_mala": t_mala,
            "acc_rw": rw_acc_rate,
            "acc_mala": mala_acc_rate,
            "burnin_rw": step_burnin_rw,
            "burnin_mala": step_burnin_mala,
            "final_err_rw": float(rw_errors[-1]),
            "final_err_mala": float(mala_errors[-1]),
            "err10_rw": float(rw_errors[10]),
            "err10_mala": float(mala_errors[10]),
        })

    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    mean_t_rw = sum(r["t_rw"] for r in results) / len(results)
    mean_t_mala = sum(r["t_mala"] for r in results) / len(results)
    mean_acc_rw = sum(r["acc_rw"] for r in results) / len(results)
    mean_acc_mala = sum(r["acc_mala"] for r in results) / len(results)
    mean_burnin_rw = sum(r["burnin_rw"] for r in results) / len(results)
    mean_burnin_mala = sum(r["burnin_mala"] for r in results) / len(results)
    mean_err_rw = sum(r["final_err_rw"] for r in results) / len(results)
    mean_err_mala = sum(r["final_err_mala"] for r in results) / len(results)

    print(f"Average Runtime (1000 steps):     Plain MCMC: {mean_t_rw:.2f}s  |  MALA: {mean_t_mala:.2f}s")
    print(f"Average Acceptance Rate:          Plain MCMC: {mean_acc_rw:.1%}  |  MALA: {mean_acc_mala:.1%}")
    print(f"Average Steps to <20% Error:      Plain MCMC: {mean_burnin_rw:.1f} steps |  MALA: {mean_burnin_mala:.1f} steps")
    print(f"Average Final Relative Error:     Plain MCMC: {mean_err_rw:.4f}  |  MALA: {mean_err_mala:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    run_comparison()
