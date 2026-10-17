"""IHW25 Example 5.2: anisotropic homogeneous Landau, Maxwell kernel.

Initial distribution N(0, diag(1.8, 0.2, 1, ...)), dimensions 3 or 10 in
the paper. Default t=0..4, dt=0.01, B=1/24. The exact covariance is known;
the evolving density and score are NOT assumed Gaussian. Figure 5.3's four
panels are written to fig5_3.png. Use homo_benchmark.py for an n sweep and
combined SBTM/blob/score-evolution curves.

Score evolution initializes with the known Gaussian score and then uses the
kernel or learned transport RHS, without analytical score labels. Direct
score-evolution runs default to forward_euler; midpoint and energy_conserving
are also supported. The transport RHS defaults to a tanh network.

The authors' code uses exp(-4*d*B*t); the printed example omits B:
https://github.com/Vilin97/GradientFlows.jl/blob/master/src/problems/landau.jl
"""

import csv
from contextlib import nullcontext
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
    HomogeneousStepper, initialize_model, make_parser, validate_args,
)
from src.score_evolution import (
    ScoreEvolutionStepper, add_score_evolution_arguments, validate_score_evolution_args,
)
from src.score_evolution_diagnostics import DIAGNOSTIC_FIELDS, save_diagnostic_plot


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
    p = make_parser(__doc__, score_methods=("sbtm", "blob", "score_evolution"),
                    time_integrators=("energy_conserving", "forward_euler", "midpoint"),
                    default_time_integrator=None)
    add_score_evolution_arguments(p, rhs_methods=("kernel", "transport"))
    p.set_defaults(t0=0.0, final_time=4.0, sbtm_hidden_dims=[100],
                   wandb_project="homo_anisotropic")
    args = validate_args(validate_score_evolution_args(p.parse_args(argv), p), p)
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
              "example": "anisotropic", "gamma": 0,
              "particle_weight": 1 / args.n, "initial_variances": [1.8, 0.2] + [1.0] * (args.dv - 2),
              "reference": "IHW25.pdf, Example 5.2, Figure 5.3",
              "covariance_normalization": "exp(-4*d*B*t), matching the authors' code and Figure 5.3",
              "density_bandwidth": "diagonal Scott: h_i=std(v_i)*n^(-1/(d+4)); covariance=diag(h_i^2)"}
    (outdir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Output: {outdir}\nDevice: {config['device']}\n{name}", flush=True)
    if args.score_method == "score_evolution":
        print(f"Score evolution: rhs={args.score_evolution_rhs}, "
              f"activation={args.sbtm_activation}", flush=True)
        if args.score_evolution_diagnostics_every:
            print(f"Score evolution trace: every {args.score_evolution_diagnostics_every} step(s), "
                  f"{outdir / 'score_evolution_diagnostics.csv'}", flush=True)
    run = None
    if args.wandb_mode != "disabled":
        import wandb
        run = wandb.init(project=args.wandb_project, name=name, mode=args.wandb_mode, config=config)

    _, sample_key, init_key = jr.split(jr.PRNGKey(args.seed), 3)
    v = sample_anisotropic(sample_key, args.n, args.dv, dtype)
    initial_mean = jnp.mean(v, axis=0)
    initial_energy = 0.5 * jnp.mean(jnp.sum(v**2, axis=1))
    initialization = {}
    model = optimizer = None
    if args.score_method in ("sbtm", "score_evolution"):
        model, optimizer, initialization = initialize_model(
            args, v, init_key, initial_score,
            sample_initial=lambda key, n: sample_anisotropic(key, n, args.dv, dtype))
        (outdir / "initialization.json").write_text(json.dumps(initialization, indent=2) + "\n")

    num_steps = math.ceil(max(0, args.final_time / args.dt - 1e-12))
    records, v_traj, t_traj, theta_traj = [], [], [], []
    started = time.perf_counter()
    last_step_metrics = dict(optimization_steps=0, gamma_squared_min=1.0, problematic_particle_count=0)
    stepper = None
    completed_steps = 0
    step, t = 0, args.t0

    def save_snapshots():
        if not t_traj:
            return
        arrays = dict(t_traj=np.array(t_traj), v_traj=np.stack(v_traj))
        if theta_traj:
            arrays["theta_traj"] = np.stack(theta_traj)
        temporary = outdir / ".snapshots.tmp.npz"
        np.savez_compressed(temporary, **arrays)
        temporary.replace(outdir / "snapshots.npz")

    def save_results(failure=None):
        summary = dict(status="failed" if failure else "completed", initialization=initialization,
                       initial=records[0] if records else None, final=records[-1] if records else None,
                       num_steps=completed_steps, planned_num_steps=num_steps,
                       last_valid_time=records[-1]["time"] if records else None,
                       evolution=stepper.summary() if stepper is not None else None,
                       simulation_wall_seconds=time.perf_counter() - started)
        if failure:
            summary["failure"] = failure
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        save_snapshots()
        save_diagnostic_plot(outdir)
        return summary

    try:
        tracing = args.score_method == "score_evolution" and args.score_evolution_diagnostics_every > 0
        trace_context = ((outdir / "score_evolution_diagnostics.csv").open("w", newline="")
                         if tracing else nullcontext(None))
        with (outdir / "metrics.csv").open("w", newline="") as metrics_file, \
                (outdir / "optimization.csv").open("w", newline="") as optimization_file, \
                trace_context as trace_file:
            metric_writer = optimization_writer = None
            if tracing:
                trace_writer = csv.DictWriter(trace_file, fieldnames=DIAGNOSTIC_FIELDS)
                trace_writer.writeheader()

            def log_diagnostics(row):
                trace_writer.writerow(row)
                trace_file.flush()

            def log_fit(fit):
                nonlocal optimization_writer
                if optimization_writer is None:
                    optimization_writer = csv.DictWriter(optimization_file, fieldnames=fit)
                    optimization_writer.writeheader()
                optimization_writer.writerow(fit)
                optimization_file.flush()

            if args.score_method == "score_evolution":
                # No analytical time-dependent score is known for this initial condition.
                stepper = ScoreEvolutionStepper(args, model, v, log_fit=log_fit,
                    log_diagnostics=log_diagnostics if tracing else None)
                config["score_evolution_bandwidth_resolved"] = (
                    stepper.bandwidth.tolist() if stepper.bandwidth is not None else None)
                (outdir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
                if run:
                    run.config.update({"score_evolution_bandwidth_resolved":
                                       config["score_evolution_bandwidth_resolved"]})
            else:
                stepper = HomogeneousStepper(args, model, optimizer, log_fit=log_fit)
            for step in range(num_steps + 1):
                t = min(step * args.dt, args.final_time)
                # Train only for actual transport steps; final diagnostics reuse weights.
                s, collision = (stepper.start_step(v, t, step) if step < num_steps
                                else stepper.evaluate(v, t))
                if args.score_method == "score_evolution":
                    stepper.observe_state(v, t, step, (s, collision), force=step == num_steps)
                snapshot_due = step % args.snapshot_every == 0 or step == num_steps
                if step % args.log_every == 0 or snapshot_due:
                    try:
                        metrics = benchmark_metrics(v, s, collision, t, initial_mean, initial_energy, args.B)
                    except FloatingPointError:
                        if tracing:
                            stepper.observe_state(v, t, step, (s, collision), force=True)
                        raise
                    record = dict(step=step, time=t, elapsed_time=t, **metrics,
                                  **last_step_metrics, wall_seconds=time.perf_counter() - started)
                    if metric_writer is None:
                        metric_writer = csv.DictWriter(metrics_file, fieldnames=record)
                        metric_writer.writeheader()
                    metric_writer.writerow(record)
                    metrics_file.flush()
                    records.append(record)
                    if run:
                        run.log({k: value for k, value in record.items() if value is not None}, step=step)
                    if snapshot_due:
                        print(f"t={t:.5g}: covariance Frobenius error={record['second_moment_error_fro']:.4g}, "
                              f"entropy rate={record['estimated_entropy_rate']:.4g}", flush=True)
                if snapshot_due:
                    v_traj.append(np.asarray(v))
                    t_traj.append(t)
                    if args.score_method == "score_evolution":
                        theta_traj.append(np.asarray(stepper.theta))
                    save_snapshots()
                if step < num_steps:
                    next_time = min((step + 1) * args.dt, args.final_time)
                    v, last_step_metrics = stepper.advance(v, t, next_time - t, step, (s, collision))
                    completed_steps = step + 1
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error), step=step, time=t)
        try:
            save_results(failure)
            print(f"Run failed. Partial diagnostics saved in {outdir}", file=sys.stderr, flush=True)
        except Exception as output_error:
            print(f"Could not finish saving partial diagnostics: {output_error}", file=sys.stderr, flush=True)
        if run:
            try:
                run.finish(exit_code=1)
            except Exception as tracking_error:
                print(f"Could not close failed tracking run: {tracking_error}", file=sys.stderr, flush=True)
        raise

    summary = save_results()
    from src.homogeneous_plots import plot_benchmarks
    plot_benchmarks([dict(config=config, records=records, summary=summary, path=str(outdir))], outdir)
    if run:
        import wandb
        run.log({"fig5_3": wandb.Image(str(outdir / "fig5_3.png"))})
        run.finish()
    print(f"Completed {num_steps} {args.time_integrator} steps. Results: {outdir}", flush=True)
    return outdir


if __name__ == "__main__":
    main()
