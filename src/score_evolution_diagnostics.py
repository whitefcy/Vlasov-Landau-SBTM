"""Observational traces for the coupled particle/score ODE (no solver changes)."""

import csv
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


PARAMETER_FIELDS = ("theta", "parameter_velocity", "parameter_increment")
ARRAY_FIELDS = (
    "particles", "score", "score_error", "acceleration", "rhs", "rhs_network",
    "rhs_analytic", "rhs_continuum", "rhs_network_error", "rhs_source_error",
    "score_jacobian", "score_jacobian_error", "score_hessian", "score_hessian_error",
    "score_antisymmetry", "transport_advection", "transport_deformation",
    "transport_divergence", "transport_advection_error", "transport_deformation_error",
    "transport_divergence_error", "projected_rhs", "fit_residual", "particle_increment",
    "score_increment", "score_linear_increment", "score_nonlinear_defect",
)
SCALAR_FIELDS = (
    "parameter_increment_relative", "particle_increment_relative",
    "score_nonlinear_defect_relative", "fit_relative_residual", "ridge_penalty",
    "cg_relative_residual", "cg_iterations", "cg_converged",
    "production_projection", "production_rhs", "production_source", "production_motion",
    "production_total", "gamma_squared_min", "problematic_particle_count",
)
DIAGNOSTIC_FIELDS = (
    ["step", "time", "phase", "stage", "dt", "rhs_method", "nonfinite_fields"]
    + [f"{name}_{suffix}" for name in PARAMETER_FIELDS
       for suffix in ("norm", "max_abs", "nonfinite_count")]
    + [f"{name}_{suffix}" for name in ARRAY_FIELDS
       for suffix in ("rms", "max_abs", "nonfinite_count")]
    + list(SCALAR_FIELDS)
)


@jax.jit
def amplitudes(x):
    """RMS vector/Frobenius norm over particles, with scaled sums of squares.

    Scaling avoids an overflowing squared norm hiding still-finite amplitudes.
    Nonfinite values are retained, and counted separately from reduction overflow.
    """
    flat = x.reshape(len(x), -1)
    maximum = jnp.max(jnp.abs(flat))
    scale = jnp.where(maximum > 0, maximum, 1.0)
    rms = scale * jnp.sqrt(jnp.mean(jnp.sum((flat / scale)**2, axis=1)))
    return dict(rms=rms, max_abs=maximum, nonfinite_count=jnp.sum(~jnp.isfinite(x)))


def stats(name, x, *, parameter=False):
    values = amplitudes(x[None, :] if parameter else x)
    if parameter:
        values["norm"] = values.pop("rms")
    return {f"{name}_{suffix}": value for suffix, value in values.items()}


def relative(numerator, denominator):
    return numerator / jnp.maximum(denominator, jnp.finfo(jnp.asarray(denominator).dtype).tiny)


class ScoreEvolutionDiagnostics:
    """Measure the actual target and independent analytic controls at stage times.

    A callback writes each event immediately, before the solve/update can fail.
    All spatial comparisons use the current particles, with equal weights.
    ``rhs_analytic`` keeps those source particles; ``rhs_continuum`` is the
    Eulerian time derivative of the exact score. They are different controls.
    """

    def __init__(self, args, score, exact_score, log):
        from src.score_evolution import make_transport_score_rhs

        self.args, self.score, self.exact_score, self.log = args, score, exact_score, log
        self.every = args.score_evolution_diagnostics_every
        self.network_details = make_transport_score_rhs(score, args.B, args.block_size,
                                                        return_details=True)
        self.analytic_details = None
        if exact_score is not None:
            self.analytic_details = make_transport_score_rhs(
                lambda t, v: exact_score(v, t), args.B, args.block_size, return_details=True)
        self.rows = 0
        self.first_nonfinite = None

    def due(self, step):
        return step % self.every == 0

    def emit(self, step, t, phase, stage, dt, values):
        values = {name: float(value) for name, value in jax.device_get(values).items()}
        bad = [name for name, value in values.items() if not math.isfinite(value)]
        row = dict(step=step, time=float(t), phase=phase, stage=stage, dt=dt,
                   rhs_method=self.args.score_evolution_rhs, nonfinite_fields=";".join(bad), **values)
        self.log(row)
        self.rows += 1
        if bad and self.first_nonfinite is None:
            self.first_nonfinite = {key: row[key] for key in ("step", "time", "phase", "stage", "nonfinite_fields")}

    def state(self, theta, v, t, step, evaluation, *, phase="state", stage=0, dt=None, force=False):
        score, collision = evaluation
        if not (force or self.due(step) or not bool(jnp.all(jnp.isfinite(score))
                                                   & jnp.all(jnp.isfinite(collision))
                                                   & jnp.all(jnp.isfinite(v))
                                                   & jnp.all(jnp.isfinite(theta)))):
            return
        values = {**stats("theta", theta, parameter=True), **stats("particles", v),
                  **stats("score", score), **stats("acceleration", -self.args.B * collision)}
        if self.exact_score is not None:
            values.update(stats("score_error", score - self.exact_score(v, t)))
        self.emit(step, t, phase, stage, dt, values)

    def rhs(self, theta, v, acceleration, target, t, step, stage, dt):
        if not (self.due(step) or not bool(jnp.all(jnp.isfinite(target)))):
            return None
        # Flush the actual inputs/target first, even if a later reference probe fails.
        self.emit(step, t, "rhs", stage, dt,
                  {**stats("theta", theta, parameter=True), **stats("particles", v),
                   **stats("acceleration", acceleration), **stats("rhs", target)})
        weights = jnp.full((len(v),), 1 / len(v), dtype=v.dtype)
        network = self.network_details(theta, v, v, weights)
        ds = network["score_jacobian"]
        values = {**stats("rhs_network", network["rhs"]), **stats("score_jacobian", ds),
                  **stats("score_hessian", network["score_hessian"]),
                  **stats("score_antisymmetry", ds - jnp.swapaxes(ds, -1, -2))}
        for term in ("advection", "deformation", "divergence"):
            values.update(stats(f"transport_{term}", network[term]))
        context = None
        if self.analytic_details is not None:
            time = jnp.asarray(t, dtype=v.dtype)
            analytic = self.analytic_details(time, v, v, weights)
            exact, continuum = jax.jvp(lambda time: self.exact_score(v, time),
                                      (time,), (jnp.ones_like(time),))
            error = self.score(theta, v) - exact
            jac_error = ds - analytic["score_jacobian"]
            values.update({**stats("score_error", error), **stats("rhs_analytic", analytic["rhs"]),
                           **stats("rhs_continuum", continuum),
                           **stats("rhs_network_error", network["rhs"] - analytic["rhs"]),
                           **stats("rhs_source_error", analytic["rhs"] - continuum),
                           **stats("score_jacobian_error", jac_error),
                           **stats("score_hessian_error", network["score_hessian"] - analytic["score_hessian"])})
            for term in ("advection", "deformation", "divergence"):
                values.update(stats(f"transport_{term}_error", network[term] - analytic[term]))
            context = (error, analytic["rhs"], continuum, jac_error)
        self.emit(step, t, "reference", stage, dt, values)
        return context

    def solve(self, theta, v, acceleration, target, velocity, diagnostics, context, t, step, stage, dt):
        if not (self.due(step) or not bool(diagnostics["valid"]) or
                not bool(jnp.all(jnp.isfinite(velocity)))):
            return
        _, projected = jax.jvp(lambda p: self.score(p, v), (theta,), (velocity,))
        fit = stats("fit_residual", projected - target)
        values = {**stats("parameter_velocity", velocity, parameter=True),
                  **stats("projected_rhs", projected), **fit,
                  "fit_relative_residual": relative(fit["fit_residual_rms"], amplitudes(target)["rms"]),
                  "ridge_penalty": self.args.score_evolution_regularization * jnp.vdot(velocity, velocity),
                  "cg_relative_residual": diagnostics["linear_relative_residual"],
                  "cg_iterations": diagnostics["iterations"], "cg_converged": diagnostics["converged"]}
        if dt is not None:
            increment = stats("parameter_increment", dt * velocity, parameter=True)
            values.update(increment)
            values["parameter_increment_relative"] = relative(
                increment["parameter_increment_norm"], amplitudes(theta[None, :])["rms"])
        if context is not None:
            error, analytic_rhs, continuum, jac_error = context

            def inner(x):
                return jnp.mean(jnp.sum(error * x, axis=1))

            # Exact continuous-ODE identity for d/dt (score MSE / 2) at moving particles.
            production = dict(production_projection=inner(projected - target),
                              production_rhs=inner(target - analytic_rhs),
                              production_source=inner(analytic_rhs - continuum),
                              production_motion=inner(jnp.einsum("nij,nj->ni", jac_error, acceleration)))
            values.update(production, production_total=sum(production.values()))
        self.emit(step, t, "solve", stage, dt, values)

    def update(self, theta, theta_new, v, v_new, t, step, dt, gamma_min, count):
        if not (self.due(step) or not bool(jnp.all(jnp.isfinite(theta_new)) & jnp.all(jnp.isfinite(v_new)))):
            return
        delta = theta_new - theta
        old_score, linear = jax.jvp(lambda p: self.score(p, v), (theta,), (delta,))
        actual = self.score(theta_new, v) - old_score
        defect = stats("score_nonlinear_defect", actual - linear)
        parameters = stats("parameter_increment", delta, parameter=True)
        particles = stats("particle_increment", v_new - v)
        values = {**stats("theta", theta_new, parameter=True), **stats("particles", v_new),
                  **stats("score", self.score(theta_new, v_new)), **parameters, **particles,
                  **stats("score_increment", actual), **stats("score_linear_increment", linear), **defect,
                  "parameter_increment_relative": relative(parameters["parameter_increment_norm"],
                                                           amplitudes(theta[None, :])["rms"]),
                  "particle_increment_relative": relative(particles["particle_increment_rms"], amplitudes(v)["rms"]),
                  "score_nonlinear_defect_relative": relative(defect["score_nonlinear_defect_rms"], amplitudes(linear)["rms"]),
                  "gamma_squared_min": gamma_min, "problematic_particle_count": count}
        if self.exact_score is not None:
            values.update(stats("score_error", self.score(theta_new, v_new) - self.exact_score(v_new, t + dt)))
        # time labels the candidate endpoint; step remains the zero-based step being advanced.
        self.emit(step, t + dt, "update", "endpoint", dt, values)

    def summary(self):
        return dict(every=self.every, rows=self.rows, first_nonfinite=self.first_nonfinite,
                    file="score_evolution_diagnostics.csv")


def save_diagnostic_plot(input_dir, output_dir=None):
    """Plot a complete or partial trace without dropping nonfinite event markers."""
    outdir = Path(input_dir if output_dir is None else output_dir)
    path = Path(input_dir) / "score_evolution_diagnostics.csv"
    if not path.exists():
        return
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("Parameters and increments", "solve", [("parameter_velocity_norm", "velocity"),
            ("parameter_increment_norm", "increment")]),
        ("State amplitudes", "state", [("theta_norm", "theta"), ("score_rms", "score"),
            ("particles_rms", "particles"), ("acceleration_rms", "acceleration")]),
        ("Learned transport terms", "reference", [("transport_advection_rms", "-Ds a"),
            ("transport_deformation_rms", "-Da.T s"), ("transport_divergence_rms", "-grad div a")]),
        ("RHS controls on current particles", None, [("rhs_rms", "actual target"), ("rhs_network_rms", "learned RHS"),
            ("rhs_analytic_rms", "analytic-score RHS"), ("rhs_continuum_rms", "continuum RHS"),
            ("rhs_network_error_rms", "network difference"), ("rhs_source_error_rms", "source mismatch")]),
        ("Score and derivative errors", "reference", [("score_error_rms", "score"),
            ("score_jacobian_error_rms", "Jacobian"), ("score_hessian_error_rms", "Hessian"),
            ("score_antisymmetry_rms", "Ds - Ds.T")]),
        ("Solve and update defects", None, [("cg_relative_residual", "CG relative residual"),
            ("fit_relative_residual", "relative fit residual"),
            ("parameter_increment_relative", "relative parameter increment"),
            ("score_nonlinear_defect_relative", "nonlinear score defect")]),
        ("Production of half score MSE", "solve", [("production_projection", "projection"),
            ("production_rhs", "target minus analytic RHS"), ("production_source", "source mismatch"),
            ("production_motion", "particle motion"), ("production_total", "total")]),
        ("Particle correction", "update", [("gamma_squared_min", "minimum Gamma squared"),
            ("problematic_particle_count", "fallback count"),
            ("particle_increment_relative", "relative particle increment")]),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(14, 16), constrained_layout=True)
    bad_time = next((float(row["time"]) for row in rows if row["nonfinite_fields"]), None)
    for index, (ax, (title, phase, fields)) in enumerate(zip(axes.flat, panels)):
        for name, label in fields:
            selected = [r for r in rows if r.get(name) and (phase is None or r["phase"] == phase)]
            # Do not mix solve-time and endpoint values in the same increment curve.
            if name == "parameter_increment_relative":
                selected = [r for r in selected if r["phase"] == "solve"]
            for stage in dict.fromkeys(r["stage"] for r in selected):
                part = [r for r in selected if r["stage"] == stage]
                x = np.array([float(r["time"]) for r in part])
                y = np.array([float(r[name]) for r in part])
                y[~np.isfinite(y)] = np.nan
                if index < 6:
                    y[y <= 0] = np.nan
                suffix = " (midpoint)" if stage == "1" else ""
                if np.any(np.isfinite(y)):
                    ax.plot(x, y, label=label + suffix)
        if index < 6:
            if ax.lines:
                ax.set_yscale("log")
        elif index == 6:
            ax.set_yscale("symlog", linthresh=1e-8)
            ax.axhline(0, color="gray", linewidth=0.5)
        if bad_time is not None:
            ax.axvline(bad_time, color="red", linestyle=":", label="first nonfinite trace")
        ax.set(title=title, xlabel="Physical time")
        ax.grid(alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.savefig(outdir / "score_evolution_diagnostics.png", dpi=150)
    plt.close(fig)
