"""Homogeneous Maxwell-molecule BKW benchmark: IHW25, Example 5.1.

Times are physical BKW times: by default t0=5.5, final_time=9.5 (400 steps).
There are no spatial particles, fields, spatial mollifier, or Coulomb kernel.
The particle weights are 1/n and the collision prefactor B appears exactly once.

The SBTM mode uses the repository's exact-divergence loss and energy-conserving
integrator, with 100 fixed optimizer updates at step start by default. This
differs from the paper's fixed 25 denoising updates with alpha=0.4. The blob
mode uses the repository's Gaussian-KDE score convention (not the variational
entropy gradient of every method called "blob" in the literature).

Example:
    python experiments/homo_BKW.py --n 12800 --score_method sbtm
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
    HomogeneousStepper, gaussian_kde, initialize_model, make_parser, maxwell_collision,
    scott_bandwidth, validate_args,
)


def bkw_parameters(t, d=3, B=1 / 24):
    """Coefficients in u*_t(v) = N(0, K I)(v) (P + Q |v|^2)."""
    K = -jnp.expm1(-2 * B * (d - 1) * t)
    P = ((d + 2) * K - d) / (2 * K)
    Q = (1 - K) / (2 * K**2)
    return K, P, Q


def bkw_density(v, t, B=1 / 24):
    K, P, Q = bkw_parameters(t, v.shape[-1], B)
    r2 = jnp.sum(v**2, axis=-1)
    return jnp.exp(-r2 / (2 * K)) * (P + Q * r2) / (2 * jnp.pi * K)**(v.shape[-1] / 2)


def bkw_score(v, t, B=1 / 24):
    K, P, Q = bkw_parameters(t, v.shape[-1], B)
    r2 = jnp.sum(v**2, axis=-1, keepdims=True)
    return v * (-1 / K + 2 * Q / (P + Q * r2))


def bkw_flow(v, t, B=1 / 24):
    """Continuum transport velocity -int A(v-w)(s(v)-s(w)) u*(w) dw."""
    return -B * (v.shape[-1] - 1) * (bkw_score(v, t, B) + v)


def bkw_fourth_moment(t, d=3, B=1 / 24):
    K, _, _ = bkw_parameters(t, d, B)
    return d * (d + 2) * K * (2 - K)


def sample_bkw(key, n, t=5.5, d=3, B=1 / 24, dtype=jnp.float64):
    """Exact iid samples via a Gaussian/size-biased Gaussian mixture.

    Component masses are P and d*K*Q=1-P. Conditional squared radii are
    K*chi2(d) and K*chi2(d+2), respectively; directions are uniform on the
    sphere. No truncation, recentering, or energy rescaling alters the sample.
    """
    if d < 2 or n < 1 or not math.isfinite(B) or B <= 0:
        raise ValueError("Require d >= 2, n >= 1, and finite B > 0")
    threshold = math.log(d / 2 + 1) / (2 * B * (d - 1))
    if not math.isfinite(t) or t <= threshold:
        raise ValueError(f"t must exceed the nonsingular BKW threshold {threshold:.12g}")
    K, P, _ = bkw_parameters(t, d, B)
    direction_key, component_key, radius_key = jr.split(key, 3)
    direction = jr.normal(direction_key, (n, d), dtype=dtype)
    direction /= jnp.linalg.norm(direction, axis=1, keepdims=True)
    component = jr.uniform(component_key, (n,), dtype=dtype) >= P
    radius2 = 2 * K * jr.gamma(radius_key, d / 2 + component.astype(dtype), dtype=dtype)
    return direction * jnp.sqrt(radius2)[:, None]


def density_l2_error(v, t, B=1 / 24, block_size=128):
    """Full-space ||KDE-u*||_2 using exact Gaussian overlap integrals.

    No velocity box or quadrature error is introduced. The error still depends
    on the reconstruction bandwidth, which is saved with every measurement.
    """
    n, d = v.shape
    h = scott_bandwidth(v)
    K, P, Q = bkw_parameters(t, d, B)
    kde_overlap, _ = gaussian_kde(v, v, jnp.sqrt(2) * h, block_size)
    variance = K + h**2
    gaussian = jnp.exp(-0.5 * jnp.sum(v**2 / variance, axis=1)) / jnp.sqrt((2 * jnp.pi)**d * jnp.prod(variance))
    conditional_r2 = jnp.sum(K * h**2 / variance) + jnp.sum((K * v / variance)**2, axis=1)
    cross = jnp.mean(gaussian * (P + Q * conditional_r2))
    true_norm2 = (4 * jnp.pi * K)**(-d / 2) * (P**2 + P * Q * d * K + Q**2 * d * (d + 2) * K**2 / 4)
    error2 = jnp.mean(kde_overlap) - 2 * cross + true_norm2
    error = jnp.sqrt(jnp.maximum(error2, 0))
    return float(error), float(error / jnp.sqrt(true_norm2)), np.asarray(h)


def parse_args(argv=None):
    p = make_parser(__doc__)
    args = validate_args(p.parse_args(argv), p)
    threshold = math.log(args.dv / 2 + 1) / (2 * args.B * (args.dv - 1))
    if args.t0 <= threshold:
        p.error(f"t0 must exceed {threshold:.12g} for a positive, nonsingular BKW density")
    return args


def benchmark_metrics(v, s, collision, t, initial_mean, initial_energy, B):
    d = v.shape[1]
    mean = jnp.mean(v, axis=0)
    second = v.T @ v / len(v)
    score_true = bkw_score(v, t, B)
    score_mse = jnp.mean(jnp.sum((s - score_true)**2, axis=1))
    flow = -B * collision
    energy = 0.5 * jnp.trace(second)
    fourth = jnp.mean(jnp.sum(v**2, axis=1)**2)
    fourth_true = bkw_fourth_moment(t, d, B)
    values = dict(
        mass=1.0, kinetic_energy=energy, kinetic_energy_exact=d / 2,
        relative_energy_drift=jnp.abs(energy - initial_energy) / initial_energy,
        second_moment_trace_drift=2 * jnp.abs(energy - initial_energy),
        momentum_norm=jnp.linalg.norm(mean), momentum_drift=jnp.linalg.norm(mean - initial_mean),
        second_moment_error_fro=jnp.linalg.norm(second - jnp.eye(d)),
        second_moment_error_fro_squared=jnp.sum((second - jnp.eye(d))**2),
        fourth_moment=fourth, fourth_moment_exact=fourth_true,
        fourth_moment_relative_error=jnp.abs(fourth - fourth_true) / fourth_true,
        score_mse=score_mse,
        score_relative_mse=score_mse / jnp.maximum(jnp.mean(jnp.sum(score_true**2, axis=1)), jnp.finfo(v.dtype).tiny),
        flow_mse=jnp.mean(jnp.sum((flow - bkw_flow(v, t, B))**2, axis=1)),
        estimated_entropy_dissipation=B * jnp.mean(jnp.sum(s * collision, axis=1)),
        collision_momentum_residual=jnp.linalg.norm(jnp.mean(collision, axis=0)),
        collision_energy_residual=jnp.abs(jnp.mean(jnp.sum(v * collision, axis=1))),
    )
    for i in range(d):
        values[f"momentum_{i + 1}"] = mean[i]
        for j in range(d):
            values[f"second_moment_{i + 1}{j + 1}"] = second[i, j]
    result = {name: float(value) for name, value in values.items()}
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError(f"Nonfinite diagnostic at t={t}")
    return result


def save_plots(outdir, records, snapshots, grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = [r["time"] for r in records]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    panels = [("second_moment_11", "Second moment (1,1)", None),
              ("second_moment_error_fro", "Second-moment matrix Frobenius error", None),
              ("score_relative_mse", "Relative score MSE", None),
              ("density_l2", "Full-space density L2 error (KDE)", None),
              ("momentum_drift", "Momentum drift from initial sample", None),
              ("second_moment_trace_drift", "Absolute second-moment trace drift", None)]
    for ax, (name, label, reference) in zip(axes.flat, panels):
        rows = [r for r in records if r.get(name) is not None]
        ax.plot([r["time"] for r in rows], [r[name] for r in rows], label="Particles")
        if reference:
            ax.plot(times, [r[reference] for r in records], "k--", label="BKW")
        elif name == "second_moment_11":
            ax.axhline(1, color="black", linestyle="--", label="BKW")
        ax.set(xlabel="Physical time", ylabel=label)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "benchmark.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, [r["fourth_moment"] for r in records], label="Particles")
    ax.plot(times, [r["fourth_moment_exact"] for r in records], "k--", label="BKW")
    ax.set(xlabel="Physical time", ylabel="Radial fourth moment (extra diagnostic)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fourth_moment.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(snapshots), figsize=(4 * len(snapshots), 6), squeeze=False)
    for col, snap in enumerate(snapshots):
        for row, prefix, label in [(0, "density", "Density slice"), (1, "score", "Score component 1")]:
            ax = axes[row, col]
            ax.plot(grid, snap[f"{prefix}_estimated"], label="KDE" if row == 0 else "Computed score")
            ax.plot(grid, snap[f"{prefix}_exact"], "k--", label="BKW")
            ax.set(xlabel="v1 (other coordinates zero)", ylabel=label, title=f"t={snap['time']:.4g}")
            ax.grid(alpha=0.25)
            ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "density_score_slices.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    jax.config.update("jax_enable_x64", not args.fp32)
    dtype = jnp.float32 if args.fp32 else jnp.float64
    name = args.wandb_run_name or f"BKW_n{args.n}_dt{args.dt}_{args.score_method}_seed{args.seed}"
    outdir = args.output_dir or Path("data/homo_BKW") / f"{name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    # Do not silently overwrite a previous benchmark with different settings.
    outdir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(outdir.resolve() / ".matplotlib"))
    config = {**vars(args), "output_dir": str(outdir), "device": jax.devices()[0].device_kind,
              "example": "bkw", "gamma": 0, "particle_weight": 1 / args.n,
              "density_bandwidth": "diagonal Scott: h_i=std(v_i)*n^(-1/(d+4)); covariance=diag(h_i^2)",
              "reference": "IHW25.pdf, Section 5.2, Example 5.1, pp. 1780-1782"}
    (outdir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Output: {outdir}\nDevice: {config['device']}\n{name}", flush=True)
    run = None
    if args.wandb_mode != "disabled":
        import wandb
        run = wandb.init(project=args.wandb_project, name=name, mode=args.wandb_mode, config=config)

    _, sample_key, init_key = jr.split(jr.PRNGKey(args.seed), 3)
    v = sample_bkw(sample_key, args.n, args.t0, args.dv, args.B, dtype)
    initial_mean = jnp.mean(v, axis=0)
    initial_energy = 0.5 * jnp.mean(jnp.sum(v**2, axis=1))
    initialization = {}
    model = optimizer = None
    if args.score_method == "sbtm":
        model, optimizer, initialization = initialize_model(args, v, init_key, bkw_score(v, args.t0, args.B))
        (outdir / "initialization.json").write_text(json.dumps(initialization, indent=2) + "\n")

    duration = args.final_time - args.t0
    num_steps = math.ceil(max(0, duration / args.dt - 1e-12))
    records, snapshots = [], []
    grid = np.linspace(-4, 4, 401)
    points = jnp.zeros((len(grid), args.dv), dtype=dtype).at[:, 0].set(jnp.asarray(grid, dtype=dtype))
    started = time.perf_counter()
    last_step_metrics = dict(optimization_steps=0, gamma_squared_min=1.0, problematic_particle_count=0)
    with (outdir / "metrics.csv").open("w", newline="") as metrics_file, (outdir / "optimization.csv").open("w", newline="") as optimization_file:
        metric_writer = optimization_writer = None

        def log_fit(fit):
            nonlocal optimization_writer
            if optimization_writer is None:
                optimization_writer = csv.DictWriter(optimization_file, fieldnames=fit)
                optimization_writer.writeheader()
            optimization_writer.writerow(fit)
            optimization_file.flush()

        stepper = HomogeneousStepper(args, model, optimizer, log_fit=log_fit,
                                     exact_score=lambda velocity, time: bkw_score(velocity, time, args.B))
        for step in range(num_steps + 1):
            t = min(args.t0 + step * args.dt, args.final_time)
            # Train only for actual transport steps; final diagnostics reuse weights.
            s, collision = (stepper.start_step(v, t, step) if step < num_steps
                            else stepper.evaluate(v, t))
            snapshot_due = step % args.snapshot_every == 0 or step == num_steps
            density_due = step % args.density_every == 0 or snapshot_due
            if step % args.log_every == 0 or density_due or step == num_steps:
                record = dict(step=step, time=t, elapsed_time=t - args.t0,
                              **benchmark_metrics(v, s, collision, t, initial_mean, initial_energy, args.B),
                              **last_step_metrics, density_l2=None, density_relative_l2=None,
                              **{f"density_bandwidth_{i+1}": None for i in range(args.dv)},
                              wall_seconds=0.0)
                if density_due:
                    absolute, relative, h = density_l2_error(v, t, args.B, args.block_size)
                    if not math.isfinite(absolute) or not math.isfinite(relative):
                        raise FloatingPointError(f"Nonfinite density diagnostic at t={t}")
                    record.update(density_l2=absolute, density_relative_l2=relative)
                    record.update({f"density_bandwidth_{i+1}": float(h[i]) for i in range(args.dv)})
                record["wall_seconds"] = time.perf_counter() - started
                if metric_writer is None:
                    metric_writer = csv.DictWriter(metrics_file, fieldnames=record)
                    metric_writer.writeheader()
                metric_writer.writerow(record)
                metrics_file.flush()
                records.append(record)
                if run:
                    run.log({k: value for k, value in record.items() if value is not None}, step=step)
                if density_due:
                    print(f"t={t:.5g}: relative score MSE={record['score_relative_mse']:.4g}, "
                          f"density L2={record['density_l2']:.4g}, energy drift={record['relative_energy_drift']:.3g}", flush=True)
            if snapshot_due:
                density_estimated, score_kde = gaussian_kde(points, v, scott_bandwidth(v), args.block_size)
                score_exact = bkw_score(points, t, args.B)
                if args.score_method == "sbtm":
                    score_estimated = model(jnp.empty((len(points), 0), dtype=dtype), points)
                else:
                    score_estimated = score_kde if args.score_method == "blob" else score_exact
                snapshots.append(dict(time=t, v=np.asarray(v), density_estimated=np.asarray(density_estimated),
                                      density_exact=np.asarray(bkw_density(points, t, args.B)),
                                      score_estimated=np.asarray(score_estimated[:, 0]), score_exact=np.asarray(score_exact[:, 0])))
            if step < num_steps:
                next_time = min(args.t0 + (step + 1) * args.dt, args.final_time)
                v, last_step_metrics = stepper.advance(v, t, next_time - t, step, (s, collision))

    np.savez_compressed(outdir / "snapshots.npz", t_traj=np.array([s["time"] for s in snapshots]),
                        v_traj=np.stack([s["v"] for s in snapshots]), slice_grid=grid,
                        **{name: np.stack([s[name] for s in snapshots]) for name in
                           ["density_estimated", "density_exact", "score_estimated", "score_exact"]})
    save_plots(outdir, records, snapshots, grid)
    summary = {"initialization": initialization, "initial": records[0], "final": records[-1],
               "num_steps": num_steps, "evolution": stepper.summary(), "simulation_wall_seconds": time.perf_counter() - started}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    from src.homogeneous_plots import plot_benchmarks
    plot_benchmarks([dict(config=config, records=records, summary=summary, path=str(outdir))], outdir)
    if run:
        import wandb
        run.log({"fig5_2": wandb.Image(str(outdir / "fig5_2.png")),
                 "benchmark": wandb.Image(str(outdir / "benchmark.png")),
                 "density_score_slices": wandb.Image(str(outdir / "density_score_slices.png"))})
        run.finish()
    print(f"Completed {num_steps} {args.time_integrator} steps. Results: {outdir}", flush=True)
    return outdir


if __name__ == "__main__":
    main()
