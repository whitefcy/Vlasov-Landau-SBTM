# Self-contained Vlasov–Landau solver with CLI + wandb logging
# Example:
# python experiments/vlasov-landau-damping.py --n 1000_000 --M 100 --dt 0.02 --gpu 0 --dv 2 --final_time 15.0 --C 0.05 --alpha 0.1 --score_method sbtm --wandb_run_name "n1e6_M100_dt0.02_C0.05_sbtm"

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import time

import jax
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import numpy as np
from scipy.signal import argrelextrema

from flax import nnx
import optax

from src import path, utils, score_model

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="1D-dvV Vlasov–Landau PIC solver")
    p.add_argument("--n", type=int, default=10**6, help="Number of particles")
    p.add_argument("--M", type=int, default=100, help="Number of spatial cells")
    p.add_argument("--dt", type=float, default=0.02, help="Time step")
    p.add_argument(
        "--time_integrator",
        type=str,
        default="forward_euler",
        choices=["forward_euler", "energy_conserving"],
        help="Particle/field/collision time integrator",
    )
    p.add_argument("--gpu", type=int, default=0, help="CUDA_VISIBLE_DEVICES index")
    p.add_argument("--fp32", action="store_true", help="Use float32 instead of float64")
    p.add_argument("--dv", type=int, default=2, help="Velocity dimension")
    p.add_argument("--final_time", type=float, default=15.0, help="Final simulation time")
    p.add_argument("--C", type=float, default=0.1, help="Collision strength")
    p.add_argument("--alpha", type=float, default=0.1, help="Amplitude of initial density perturbation")
    p.add_argument("--score_method", type=str, default="blob", choices=["blob", "scaled_blob", "sbtm"])
    p.add_argument("--seed", type=int, default=42, help="Random seed for initialization")

    # sbtm-specific args (used only if score_method == "sbtm")
    p.add_argument("--sbtm_batch_size", type=int, default=20_000)
    p.add_argument("--sbtm_num_epochs", type=int, default=10_000)
    p.add_argument("--sbtm_abs_tol", type=float, default=1e-4)
    p.add_argument("--sbtm_lr", type=float, default=2e-4)
    p.add_argument("--sbtm_num_batch_steps", type=int, default=100)

    p.add_argument("--score_quiver_scale", type=float, default=1.0, help="Scale for score quiver plots")
    p.add_argument("--flow_quiver_scale", type=float, default=0.1, help="Scale for flow quiver plots")
    p.add_argument("--mse_every", type=int, default=20, help="Compute and log score/flow MSE every k steps")

    p.add_argument("--wandb_project", type=str, default="vlasov_landau_damping", help="wandb project name")
    p.add_argument("--wandb_run_name", type=str, default="landau_damping", help="wandb run name")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--log_every", type=int, default=1, help="Log every k steps")
    return p.parse_args()


def main():
    args = parse_args()

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    jax.config.update("jax_enable_x64", not args.fp32)

    gpu_name = jax.devices()[0].device_kind
    wandb_run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config={**vars(args), "gpu_name": gpu_name},
    )
    print(f"Args: {args}")
    print(f"GPU = {gpu_name}")

    # Parameters
    seed = args.seed
    q = 1
    dv = args.dv
    alpha = args.alpha
    k = 0.5
    L = 2 * jnp.pi / k
    n = args.n
    M = args.M
    dt = args.dt
    eta = L / M
    cells = (jnp.arange(M) + 0.5) * eta
    w = q * L / n
    C = args.C
    gamma = -dv
    dx = 1  # 1D in x

    key_v, key_x = jr.split(jr.PRNGKey(seed), 2)
    v = jr.normal(key_v, (n, dv))
    v = v - jnp.mean(v, axis=0)

    def spatial_density(x):
        return (1 + alpha * jnp.cos(k * x)) / (2 * jnp.pi / k)

    v_target = lambda vv: jax.scipy.stats.norm.pdf(vv, 0, 1)

    max_value = jnp.max(spatial_density(cells))
    domain = (0.0, float(L))
    x = utils.rejection_sample(key_x, spatial_density, domain, max_value=max_value, num_samples=n)

    # choose score method / sbtm setup
    score_method = args.score_method
    model = None
    optimizer = None
    training_config = None

    if score_method == "blob":
        score_fn = utils.score_blob
    elif score_method == "scaled_blob":
        score_fn = utils.scaled_score_blob
    elif score_method == "sbtm":
        hidden_dims = (256, 256)
        training_config = {
            "batch_size": args.sbtm_batch_size,
            "num_epochs": args.sbtm_num_epochs,
            "abs_tol": args.sbtm_abs_tol,
            "lr": args.sbtm_lr,
            "num_batch_steps": args.sbtm_num_batch_steps,
        }
        model = score_model.MLPScoreModel(dx, dv, hidden_dims=hidden_dims)
        example_name = "landau_damping"
        model_path = os.path.abspath(os.path.join(
            "data/score_models",
            f"{example_name}_dx{dx}_dv{dv}_alpha{alpha}_k{k}/hidden_{str(hidden_dims)}/epochs_{training_config['num_epochs']}",
        ))
        if os.path.exists(model_path):
            model.load(model_path)
        else:
            utils.train_initial_model(model, x, v, -v, batch_size=training_config["batch_size"], num_epochs=training_config["num_epochs"], abs_tol=training_config["abs_tol"], lr=training_config["lr"], verbose=True)
            try:
                model.save(model_path)
            except Exception as e:
                print(f"Warning: could not save model to {model_path}: {e}")
            time.sleep(1)
        optimizer = nnx.Optimizer(model, optax.adamw(training_config["lr"]))

        def score_fn(x_in, v_in, cells_in, eta_in):
            return model(x_in, v_in)
    else:
        raise ValueError(f"Unknown score method: {score_method}")

    rho = utils.evaluate_charge_density(x, cells, eta, w)
    E = jnp.cumsum(rho - 1) * eta

    fig_init = utils.visualize_initial(x, v[:, 0], cells, E, rho, eta, L, spatial_density, v_target)
    wandb.log({"initial_state": wandb.Image(fig_init)}, step=0)
    plt.show()
    plt.close(fig_init)

    final_time = args.final_time
    steps_float = final_time / dt
    num_steps = int(round(steps_float))
    if not np.isclose(steps_float, num_steps, rtol=1e-10, atol=1e-12):
        raise ValueError(
            f"final_time ({final_time}) must be an integer multiple of dt ({dt})"
        )
    E_L2 = [jnp.sqrt(jnp.sum(E ** 2) * eta)]
    problematic_particle_counts = [0]

    print(
        f"Landau Damping with n={n:.0e}, M={M}, dt={dt}, eta={float(eta):.4f}, "
        f"score_method={score_method}, time_integrator={args.time_integrator}, "
        f"dv={dv}, C={C}, alpha={alpha}, gpu={args.gpu}, fp32={args.fp32}"
    )

    # Quiver of scores before time stepping
    if score_method == "sbtm":
        s_plot = model(x, v)
    else:
        s_plot = score_fn(x, v, cells, eta)
    s_true = -v

    fig_quiver = utils.plot_score_quiver(v, s_plot, s_true, label=score_method, scale=args.score_quiver_scale)
    wandb.log({"score_quiver": wandb.Image(fig_quiver)}, step=0)
    plt.show()
    plt.close(fig_quiver)

    Q0_pred = utils.collision(x, v, s_plot, eta, gamma, L, w)
    Q0_true = utils.collision(x, v, s_true, eta, gamma, L, w)

    fig_flow0 = utils.plot_U_quiver_pred(v, -Q0_pred, label=f"{score_method}, t=0.0", U_true=-Q0_true, scale=args.flow_quiver_scale)
    wandb.log({"flow_quiver": wandb.Image(fig_flow0)}, step=0)
    plt.show()
    plt.close(fig_flow0)

    # Main time loop with steps/sec logging
    snapshot_times = np.linspace(0.0, final_time, 6)
    snapshot_steps = set(int(round(T / dt)) for T in snapshot_times)

    x_traj, v_traj, t_traj = [], [], []
    if 0 in snapshot_steps:
        x_traj.append(np.asarray(x.block_until_ready()))
        v_traj.append(np.asarray(v.block_until_ready()))
        t_traj.append(0.0)
        snapshot_steps.remove(0)

    start_time = time.perf_counter()
    for istep in tqdm(range(num_steps)):
        completed_step = istep + 1
        gamma_squared_min = None
        problematic_particle_count = 0

        if args.time_integrator == "energy_conserving":
            def collision_evaluator(x_stage, v_stage, stage):
                if score_method == "sbtm":
                    key = jr.fold_in(jr.PRNGKey(seed), 3 * istep + stage)
                    utils.train_score_model(
                        model,
                        optimizer,
                        x_stage,
                        v_stage,
                        key,
                        batch_size=training_config["batch_size"],
                        num_batch_steps=training_config["num_batch_steps"],
                    )
                    s_stage = model(x_stage, v_stage)
                else:
                    s_stage = score_fn(x_stage, v_stage, cells, eta)
                Q_stage = utils.collision(
                    x_stage, v_stage, s_stage, eta, gamma, L, w
                )
                return s_stage, Q_stage

            (
                x,
                v,
                E,
                stage_results,
                raw_gamma_squared,
                problematic_particles,
                x_collision,
                v_collision,
            ) = utils.energy_conserving_step(
                x,
                v,
                E,
                cells,
                eta,
                dt,
                L,
                w,
                C,
                collision_evaluator=collision_evaluator if C > 0 else None,
            )
            gamma_squared_min = float(jnp.min(raw_gamma_squared))
            problematic_particle_count = int(jnp.sum(problematic_particles))

            if C > 0:
                s, Q = stage_results[-1]
                score_mse = float(
                    jnp.mean(jnp.sum((s - (-v_collision)) ** 2, axis=1))
                )
                if completed_step % args.mse_every == 0:
                    Q_gaussian = utils.collision(
                        x_collision, v_collision, -v_collision, eta, gamma, L, w
                    )
                    flow_mse = float(
                        jnp.mean(jnp.sum((Q - Q_gaussian) ** 2, axis=1))
                    )
                entropy_production = jnp.mean(jnp.sum(s * C * Q, axis=1))
            else:
                entropy_production = 0.0
                score_mse = 0.0
        else:
            x, v, E = utils.vlasov_step(x, v, E, cells, eta, dt, L, w)

        if args.time_integrator == "forward_euler" and C > 0:
            if score_method == "sbtm":
                s = model(x, v)
                key = jr.PRNGKey(istep)
                utils.train_score_model(
                    model,
                    optimizer,
                    x,
                    v,
                    key,
                    batch_size=training_config["batch_size"],
                    num_batch_steps=training_config["num_batch_steps"],
                )
            else:
                s = score_fn(x, v, cells, eta)
            Q = utils.collision(x, v, s, eta, gamma, L, w)
            score_mse = float(jnp.mean(jnp.sum((s - (-v)) ** 2, axis=1)))
            if completed_step % args.mse_every == 0:
                Q_gaussian = utils.collision(x, v, -v, eta, gamma, L, w)
                flow_mse = float(jnp.mean(jnp.sum((Q - Q_gaussian) ** 2, axis=1)))
            v = v - dt * C * Q
            entropy_production = jnp.mean(jnp.sum(s * C * Q, axis=1))
        elif args.time_integrator == "forward_euler":
            entropy_production = 0.0
            score_mse = 0.0

        electric_energy = 0.5 * jnp.sum(E ** 2) * eta # electric energy
        momentum = jnp.mean(v, axis=0)
        kinetic_energy = 0.5 * jnp.mean(jnp.sum(v ** 2, axis=1)) * L
        total_energy = kinetic_energy + electric_energy
        E_norm = jnp.sqrt(electric_energy)

        E_L2.append(E_norm)
        problematic_particle_counts.append(problematic_particle_count)
        if completed_step % args.log_every == 0:
            elapsed = time.perf_counter() - start_time
            steps_per_sec = completed_step / elapsed
            mom_dict = {f"momentum/{i+1}": float(m) for i, m in enumerate(momentum)}
            log_dict = {
                "step": completed_step,
                "time": float(completed_step * dt),
                "steps_per_sec": steps_per_sec,
                "E_L2": float(E_norm),
                "electric_energy": float(electric_energy),
                "kinetic_energy": float(kinetic_energy),
                "total_energy": float(total_energy),
                "problematic_particle_count": problematic_particle_count,
                "entropy_production": float(entropy_production),
                "score_mse": score_mse,
                **mom_dict,
            }
            if gamma_squared_min is not None:
                log_dict["gamma_squared_min"] = gamma_squared_min
            if C > 0 and completed_step % args.mse_every == 0:
                log_dict["flow_mse"] = flow_mse
            wandb.log(
                log_dict,
                step=completed_step,
            )
        
        # snapshots
        if completed_step in snapshot_steps:
            x_host = np.asarray(x.block_until_ready())
            v_host = np.asarray(v.block_until_ready())
            x_traj.append(x_host)
            v_traj.append(v_host)
            t_traj.append(completed_step * dt)

            if score_method == "sbtm":
                s = model(x, v)
            else:
                s = score_fn(x, v, cells, eta)
            Q = utils.collision(x, v, s, eta, gamma, L, w)

            fig_quiver_score_snap = utils.plot_score_quiver_pred(
                v, s, label=f"{score_method}, t={completed_step * dt:.2f}", scale=args.score_quiver_scale
            )
            wandb.log({"score_quiver": wandb.Image(fig_quiver_score_snap)}, step=completed_step)
            plt.close(fig_quiver_score_snap)
            
            fig_quiver_flow_snap = utils.plot_U_quiver_pred(v, -Q, label=f"{score_method}, t={completed_step * dt:.2f}", scale=args.flow_quiver_scale)
            wandb.log({"flow_quiver": wandb.Image(fig_quiver_flow_snap)}, step=completed_step)
            plt.close(fig_quiver_flow_snap)



    # Phase-space snapshots
    title = fr"Landau damping α={alpha}, k={k}, C={C}, n={n:.0e}, M={M}, Δt={dt}, {score_method}"
    outdir_ps = f"data/plots/phase_space/landau_damping_1d_{dv}v/"
    fname_ps = f"landau_damping_phase_space_n{n:.0e}_M{M}_dt{dt}_dv{dv}_C{C}_alpha{alpha}_k{k}.png"

    fig_ps, path_ps = utils.plot_phase_space_snapshots(
        x_traj, v_traj, t_traj, L, title, outdir_ps, fname_ps
    )

    wandb.log({"phase_space_snapshots": wandb.Image(fig_ps)}, step=num_steps + 1)
    plt.show()
    plt.close(fig_ps)

    snap_art = wandb.Artifact(
        name="landau_damping_snapshots",
        type="snapshot_data",
    )
    snap_art.add_file(os.path.join(outdir_ps, fname_ps))
    snap_art.metadata = dict(
        n=n,
        M=M,
        dt=dt,
        dv=dv,
        C=C,
        alpha=alpha,
        k=k,
        score_method=score_method,
    )
    snapshots_dir = os.path.join(
        "data",
        "snapshots",
        f"landau_damping_n{n:.0e}_M{M}_dt{dt}_{score_method}_dv{dv}_C{C}_alpha{alpha}_k{k}",
    )
    os.makedirs(snapshots_dir, exist_ok=True)
    snapshots_raw_path = os.path.join(snapshots_dir, "snapshots_raw.npz")
    np.savez_compressed(
        snapshots_raw_path,
        x_traj=np.array(x_traj, dtype=object),
        v_traj=np.array(v_traj, dtype=object),
        t_traj=np.array(t_traj),
    )
    snap_art.add_file(snapshots_raw_path)
    wandb.log_artifact(snap_art)

    # Problematic Gamma correction count
    problematic_time_grid = np.arange(num_steps + 1) * dt
    fig_problematic = plt.figure(figsize=(6, 4))
    plt.plot(
        problematic_time_grid,
        problematic_particle_counts,
        marker="o",
        ms=2,
    )
    plt.xlabel("Time")
    plt.ylabel("Number of problematic particles")
    plt.title(f"Problematic Gamma corrections, {args.time_integrator}")
    plt.grid(True)
    plt.tight_layout()

    outdir_problematic = "data/plots/problematic_particles/"
    os.makedirs(outdir_problematic, exist_ok=True)
    fname_problematic = (
        f"problematic_particles_n{n:.0e}_M{M}_dt{dt}_{score_method}_"
        f"dv{dv}_C{C}_{args.time_integrator}.png"
    )
    path_problematic = os.path.join(outdir_problematic, fname_problematic)
    plt.savefig(path_problematic)
    wandb.log(
        {"problematic_particles": wandb.Image(fig_problematic)},
        step=num_steps + 1,
    )
    wandb.save(path_problematic)
    plt.show()
    plt.close(fig_problematic)

    # Post-processing: Landau damping fit
    t_grid = jnp.arange(num_steps + 1) * dt
    
    # Limit theoretical predictions to t=15
    t_theory_max = 15.0
    theory_num_steps = int(np.floor(min(t_theory_max, final_time) / dt + 1e-12))
    t_grid_theory = jnp.arange(theory_num_steps + 1) * dt

    fig_final = plt.figure(figsize=(6, 4))
    plt.plot(t_grid, E_L2, marker="o", ms=1, label=f"Simulation (C={C})")

    prefactor = -1 / (k ** 3) * jnp.sqrt(jnp.pi / 8) * jnp.exp(
        -1 / (2 * k**2) - 1.5
    )
    pred = jnp.exp(t_grid_theory * prefactor)
    pred *= E_L2[0] / pred[0]
    plt.plot(
        t_grid_theory,
        pred,
        "k-.",
        label=fr"collisionless: $e^{{\beta t}}, \beta={float(prefactor):.3f}$",
    )

    prefactor_collisional = prefactor - C * jnp.sqrt(2 / (9 * jnp.pi))
    predicted_collisional = jnp.exp(t_grid_theory * prefactor_collisional)
    predicted_collisional *= E_L2[0] / predicted_collisional[0]
    if C > 0:
        plt.plot(
            t_grid_theory,
            predicted_collisional,
            "r--",
            label=fr"collisional: $e^{{\beta t}},\ \beta = {float(prefactor_collisional):.3f}$",
        )

    t_np = np.asarray(t_grid)
    E_np = np.asarray(E_L2)
    mask = (t_np > 0.2) & (t_np < min(t_theory_max, final_time))
    t_mask = t_np[mask]
    E_mask = E_np[mask]
    max_idx = argrelextrema(E_mask, np.greater, order=20)[0]
    mt, mv = t_mask[max_idx], E_mask[max_idx]
    plt.scatter(mt, mv, c="g", zorder=5)
    if len(mt) > 1:
        coeffs = np.polyfit(mt, np.log(mv), 1)
        fit = np.exp(coeffs[1] + coeffs[0] * t_mask)
        plt.plot(
            t_mask,
            fit,
            "g--",
            label=fr"fitted: $e^{{\beta t}}, \beta={coeffs[0]:.3f}$",
        )
        wandb.log({"beta_fit": coeffs[0]})

    plt.xlabel("Time")
    plt.ylabel(r"$||E||_{L^2}$")
    plt.title(f"C={C}, n={n:.0e}, Δt={dt}, dv={dv}, α={alpha}, M={M}")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    outdir = f"data/plots/electric_field_norm/collision_1d_{dv}v/"
    os.makedirs(outdir, exist_ok=True)
    fname = (
        f"landau_damping_n{n:.0e}_M{M}_dt{dt}_{score_method}_dv{dv}_C{C}_alpha{alpha}_{score_method}.png"
    )
    p = os.path.join(outdir, fname)
    plt.savefig(p)

    wandb.log({"landau_damping": wandb.Image(fig_final)}, step=num_steps + 2)
    wandb.save(p)
    plt.show()
    plt.close(fig_final)

    wandb_run.finish()


if __name__ == "__main__":
    main()
