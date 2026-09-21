"""IHW25 Example 5.2: anisotropic homogeneous Landau, Maxwell kernel.

Initial distribution N(0, diag(1.8, 0.2, 1, ...)), dimensions 3 or 10 in
the paper. Default t=0..4, dt=0.01, B=1/24. The exact covariance is known;
the evolving density and score are NOT assumed Gaussian. Figure 5.3's four
panels are written to fig5_3.png. Use homo_benchmark.py for an n sweep and
combined SBTM/blob curves.

The authors' code uses exp(-4*d*B*t); the printed example omits B:
https://github.com/Vilin97/GradientFlows.jl/blob/master/src/problems/landau.jl
"""

import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from src.homogeneous import (
    gaussian_kde, initialize_model, make_parser, maxwell_collision,
    scott_bandwidth, validate_args,
)


def initial_variances(d, dtype=jnp.float64):
    if d < 2:
        raise ValueError("At least two velocity dimensions are required")
    return jnp.ones(d, dtype=dtype).at[0].set(1.8).at[1].set(0.2)


def sample_anisotropic(key, n, d=3, dtype=jnp.float64):
    """Unmodified iid Gaussian samples; 1.8 and 0.2 are variances, not stds."""
    return jr.normal(key, (n, d), dtype=dtype) * jnp.sqrt(initial_variances(d, dtype))


def covariance_exact(t, d=3, B=1 / 24):
    sigma0 = initial_variances(d)
    equilibrium = jnp.mean(sigma0)
    return jnp.diag(equilibrium + (sigma0 - equilibrium) * jnp.exp(-4 * d * B * t))


def initial_score(v):
    """The Gaussian score is exact only at initialization, t=0."""
    return -v / initial_variances(v.shape[-1], v.dtype)


def parse_args(argv=None):
    p = make_parser(__doc__, score_methods=("sbtm", "blob"))
    p.set_defaults(t0=0.0, final_time=4.0, sbtm_hidden_dims=[100],
                   wandb_project="homo_anisotropic")
    args = validate_args(p.parse_args(argv), p)
    if args.t0 != 0:
        p.error("Example 5.2 initializes at t0=0; change final_time to set the duration")
    return args


def benchmark_metrics(v, s, collision, t, initial_mean, initial_energy, B):
    d = v.shape[1]
    mean = jnp.mean(v, axis=0)
    second = v.T @ v / len(v)
    centered_covariance = second - jnp.outer(mean, mean)
    reference = covariance_exact(t, d, B)
    flow = -B * collision
    energy = 0.5 * jnp.trace(second)
    rate = jnp.mean(jnp.sum(s * flow, axis=1))
    values = dict(
        mass=1.0, kinetic_energy=energy, kinetic_energy_exact=d / 2,
        relative_energy_drift=jnp.abs(energy - initial_energy) / initial_energy,
        second_moment_trace_drift=2 * jnp.abs(energy - initial_energy),
        momentum_norm=jnp.linalg.norm(mean), momentum_drift=jnp.linalg.norm(mean - initial_mean),
        second_moment_error_fro=jnp.linalg.norm(second - reference),
        second_moment_error_fro_squared=jnp.sum((second - reference)**2),
        centered_covariance_error_fro=jnp.linalg.norm(centered_covariance - reference),
        estimated_entropy_rate=rate, estimated_entropy_dissipation=-rate,
        collision_momentum_residual=jnp.linalg.norm(jnp.mean(collision, axis=0)),
        collision_energy_residual=jnp.abs(jnp.mean(jnp.sum(v * collision, axis=1))),
    )
    for i in range(d):
        values[f"momentum_{i+1}"] = mean[i]
        values[f"second_moment_exact_{i+1}{i+1}"] = reference[i, i]
        for j in range(d):
            values[f"second_moment_{i+1}{j+1}"] = second[i, j]
    result = {name: float(value) for name, value in values.items()}
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError(f"Nonfinite diagnostic at t={t}")
    return result


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    jax.config.update("jax_enable_x64", not args.fp32)
    dtype = jnp.float32 if args.fp32 else jnp.float64
    name = args.wandb_run_name or f"anisotropic_d{args.dv}_n{args.n}_dt{args.dt}_{args.score_method}_seed{args.seed}"
    outdir = args.output_dir or Path("data/homo_anisotropic") / f"{name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    outdir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(outdir.resolve() / ".matplotlib"))
    config = {**vars(args), "output_dir": str(outdir), "device": jax.devices()[0].device_kind,
              "example": "anisotropic", "time_integrator": "forward_euler", "gamma": 0,
              "particle_weight": 1 / args.n, "initial_variances": [1.8, 0.2] + [1.0] * (args.dv - 2),
              "reference": "IHW25.pdf, Example 5.2, Figure 5.3",
              "covariance_normalization": "exp(-4*d*B*t), matching the authors' code and Figure 5.3",
              "density_bandwidth": "diagonal Scott: h_i=std(v_i)*n^(-1/(d+4)); covariance=diag(h_i^2)"}
    (outdir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Output: {outdir}\nDevice: {config['device']}\n{name}", flush=True)
    run = None
    if args.wandb_mode != "disabled":
        import wandb
        run = wandb.init(project=args.wandb_project, name=name, mode=args.wandb_mode, config=config)

    key, sample_key, init_key = jr.split(jr.PRNGKey(args.seed), 3)
    v = sample_anisotropic(sample_key, args.n, args.dv, dtype)
    initial_mean = jnp.mean(v, axis=0)
    initial_energy = 0.5 * jnp.mean(jnp.sum(v**2, axis=1))
    x = jnp.empty((args.n, 0), dtype=dtype)
    initialization = {}
    if args.score_method == "sbtm":
        from src.adaptive_score import fit_score
        model, optimizer, initialization = initialize_model(args, v, init_key, initial_score(v))
        (outdir / "initialization.json").write_text(json.dumps(initialization, indent=2) + "\n")

    num_steps = math.ceil(max(0, args.final_time / args.dt - 1e-12))
    records, v_traj, t_traj = [], [], []
    started = time.perf_counter()
    with (outdir / "metrics.csv").open("w", newline="") as metrics_file, (outdir / "optimization.csv").open("w", newline="") as optimization_file:
        metric_writer = optimization_writer = None
        for step in range(num_steps + 1):
            t = min(step * args.dt, args.final_time)
            optimization_steps = 0
            if args.score_method == "sbtm":
                if step > 0:
                    key, train_key = jr.split(key)
                    fit = fit_score(model, optimizer, x, v, train_key,
                                    batch_size=args.sbtm_batch_size, max_steps=args.sbtm_num_batch_steps,
                                    adaptive=args.sbtm_adaptive, min_steps=args.sbtm_min_steps,
                                    check_every=args.sbtm_check_every, patience=args.sbtm_patience,
                                    atol=args.sbtm_stop_atol, rtol=args.sbtm_stop_rtol,
                                    monitor_size=args.sbtm_monitor_size, div_mode=args.sbtm_div_mode)
                    fit = {"step": step, "time": t, **fit}
                    if optimization_writer is None:
                        optimization_writer = csv.DictWriter(optimization_file, fieldnames=fit)
                        optimization_writer.writeheader()
                    optimization_writer.writerow(fit)
                    optimization_file.flush()
                    optimization_steps = fit["optimization_steps"]
                s = model(x, v)
            else:
                _, s = gaussian_kde(v, v, scott_bandwidth(v), args.block_size)
            collision = maxwell_collision(v, s)
            snapshot_due = step % args.snapshot_every == 0 or step == num_steps
            if step % args.log_every == 0 or snapshot_due:
                record = dict(step=step, time=t, elapsed_time=t,
                              **benchmark_metrics(v, s, collision, t, initial_mean, initial_energy, args.B),
                              optimization_steps=optimization_steps,
                              wall_seconds=time.perf_counter() - started)
                if metric_writer is None:
                    metric_writer = csv.DictWriter(metrics_file, fieldnames=record)
                    metric_writer.writeheader()
                metric_writer.writerow(record)
                metrics_file.flush()
                records.append(record)
                if run:
                    run.log(record, step=step)
                if snapshot_due:
                    print(f"t={t:.5g}: covariance Frobenius error={record['second_moment_error_fro']:.4g}, "
                          f"entropy rate={record['estimated_entropy_rate']:.4g}", flush=True)
            if snapshot_due:
                v_traj.append(np.asarray(v))
                t_traj.append(t)
            if step < num_steps:
                next_time = min((step + 1) * args.dt, args.final_time)
                v = v - (next_time - t) * args.B * collision

    np.savez_compressed(outdir / "snapshots.npz", t_traj=np.array(t_traj), v_traj=np.stack(v_traj))
    summary = {"initialization": initialization, "initial": records[0], "final": records[-1],
               "num_steps": num_steps, "simulation_wall_seconds": time.perf_counter() - started}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    from src.homogeneous_plots import plot_benchmarks
    plot_benchmarks([dict(config=config, records=records, summary=summary, path=str(outdir))], outdir)
    if run:
        import wandb
        run.log({"fig5_3": wandb.Image(str(outdir / "fig5_3.png"))})
        run.finish()
    print(f"Completed {num_steps} forward Euler steps. Results: {outdir}", flush=True)
    return outdir


if __name__ == "__main__":
    main()
