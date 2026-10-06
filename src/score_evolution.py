"""Homogeneous score evolution from ``Latex/VML equations.tex``.

The network satisfies the Eulerian equation d_t s = R at fixed velocity.
Either Gaussian particles or the score transport equation supplies R, and a
weighted ridge least-squares solve gives the parameter velocity. Forward Euler
and explicit midpoint advance particles and parameters together. The
energy-conserving option uses the project's
three-stage particle update with the old network held fixed, while retaining
the forward-Euler parameter update. No implicit score matching is performed
after initialization.
"""

from functools import partial
import math
import time
import warnings

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from src.homogeneous import maxwell_collision, scott_bandwidth
from src.time_integrators import homogeneous_step


def add_score_evolution_arguments(parser):
    group = parser.add_argument_group("score evolution")
    group.add_argument("--score_evolution_rhs", choices=("kernel", "transport"), default="kernel",
                       help="Eulerian score derivative R: Gaussian kernel approximation (default), "
                            "or transport equation with velocity derivatives of the score network")
    group.add_argument("--score_evolution_regularization", type=float, default=1e-4,
                       help="Positive lambda in sum_p w_p |J xi - R|^2 + lambda |xi|^2")
    group.add_argument("--score_evolution_bandwidth", type=float, nargs="+", default=None,
                       help="Kernel RHS only: fixed Gaussian standard deviation(s), one or dv values; "
                            "default is diagonal Scott bandwidth from the initial particles; "
                            "unused by transport")
    group.add_argument("--score_evolution_cg_tol", type=float, default=1e-6,
                       help="Relative normal-equation residual tolerance for conjugate gradient")
    group.add_argument("--score_evolution_cg_maxiter", type=int, default=200,
                       help="Maximum conjugate-gradient iterations per stage")


def validate_score_evolution_args(args, parser):
    if args.time_integrator is None:
        args.time_integrator = ("forward_euler" if args.score_method == "score_evolution"
                                else "energy_conserving")
    if args.score_method == "score_evolution":
        if args.time_integrator not in ("forward_euler", "midpoint", "energy_conserving"):
            parser.error("score_evolution requires --time_integrator forward_euler, midpoint, or energy_conserving")
    elif args.time_integrator == "midpoint":
        parser.error("midpoint is currently supported only with --score_method score_evolution")
    for name in ("score_evolution_regularization", "score_evolution_cg_tol",
                 "score_evolution_cg_maxiter"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be finite and positive")
    if args.score_evolution_cg_tol >= 1:
        parser.error("score_evolution_cg_tol must be less than 1")
    h = args.score_evolution_bandwidth
    if h is not None and (len(h) not in (1, args.dv)
                          or any(not math.isfinite(value) or value <= 0 for value in h)):
        parser.error("score_evolution_bandwidth requires one or dv finite positive standard deviations")
    return args


@partial(jax.jit, static_argnames=("block_size",))
def gaussian_score_rhs(points, centers, accelerations, h, weights=None, block_size=128):
    """Evaluate R = d_t grad(log f_h) at *fixed* query points and bandwidth.

    ``h`` contains Gaussian standard deviations (covariance diag(h**2)).
    The centers move with the supplied accelerations; weights stay constant.
    Including self terms agrees with the paper's particle reconstruction.
    We evaluate f_t/f and grad(f)_t/f using normalized kernel weights, avoiding
    division by a possibly underflowed density. No material/convective term
    involving the motion of a query point is added.

    Work is O(m*n*d), with O(block_size*n*d) pairwise storage. ``weights``
    defaults to 1/n; supplied weights must be nonnegative with positive sum.
    """
    m, d = points.shape
    n = centers.shape[0]
    if weights is None:
        weights = jnp.full((n,), 1 / n, dtype=centers.dtype)
    log_weights = jnp.log(weights)
    padding = (-m) % block_size
    blocks = jnp.pad(points, ((0, padding), (0, 0))).reshape(-1, block_size, d)

    def evaluate(block):
        delta = centers[None, :, :] - block[:, None, :]
        grad_log_kernel = delta / h**2
        exponent = -0.5 * jnp.sum((delta / h)**2, axis=-1) + log_weights
        shifted = jnp.exp(exponent - jnp.max(exponent, axis=1, keepdims=True))
        probabilities = shifted / jnp.sum(shifted, axis=1, keepdims=True)
        score = jnp.sum(probabilities[:, :, None] * grad_log_kernel, axis=1)
        # K_t/K = -a_q . grad(log K); (grad K)_t/K = a_q/h^2 + grad(log K)*K_t/K.
        kernel_time_ratio = -jnp.sum(grad_log_kernel * accelerations[None, :, :], axis=-1)
        return jnp.sum(probabilities[:, :, None] * (
            accelerations[None, :, :] / h**2
            + (grad_log_kernel - score[:, None, :]) * kernel_time_ratio[:, :, None]
        ), axis=1)

    return jax.lax.map(evaluate, blocks).reshape(-1, d)[:m]


def make_transport_score_rhs(score_function, B, block_size=128):
    """Build R = -Ds a - Da.T s - grad(div a) for Maxwell particles.

    ``score_function(theta, velocities)`` returns a batch of score vectors.
    The returned function takes ``(theta, points, centers, weights)``; weights
    must be nonnegative with positive sum. Source particles, source scores,
    and their moments stay fixed during differentiation of a query velocity.
    Each stage recomputes them from that stage's particles and parameters.

    A centered moment expansion evaluates the Maxwell particle sum exactly,
    avoiding pairwise derivative arrays. Only the query's d velocity inputs
    are differentiated, including second derivatives for grad(div a). A smooth
    score model is recommended. Blocking bounds derivative activation storage.
    The collision prefactor B is included exactly once here.
    """
    @jax.jit
    def evaluate(theta, points, centers, weights):
        mass = jnp.sum(weights)
        probabilities = weights / mass
        source_scores = score_function(theta, centers)
        mean_v = jnp.sum(probabilities[:, None] * centers, axis=0)
        mean_s = jnp.sum(probabilities[:, None] * source_scores, axis=0)
        u = centers - mean_v
        q = source_scores - mean_s
        second = u.T @ (probabilities[:, None] * u)
        cross = u.T @ (probabilities[:, None] * q)
        radius_score = jnp.sum(
            probabilities[:, None] * jnp.sum(u**2, axis=1, keepdims=True) * q, axis=0)
        mixed = jnp.sum(
            probabilities[:, None] * u * jnp.sum(u * q, axis=1, keepdims=True), axis=0)

        def score(v):
            return score_function(theta, v[None, :])[0]

        def acceleration(v):
            w = v - mean_v
            r = score(v) - mean_s
            return -B * mass * (
                (jnp.dot(w, w) + jnp.trace(second)) * r
                - w * jnp.dot(w, r) - r @ second - radius_score
                + 2 * w @ cross - w * jnp.trace(cross) - w @ cross.T + mixed)

        jac_a = jax.jacfwd(acceleration)
        jac_s = jax.jacfwd(score)
        grad_div_a = jax.grad(lambda v: jnp.trace(jac_a(v)))

        def rhs(v):
            return -jac_s(v) @ acceleration(v) - jac_a(v).T @ score(v) - grad_div_a(v)

        m, d = points.shape
        padding = (-m) % block_size
        blocks = jnp.pad(points, ((0, padding), (0, 0))).reshape(-1, block_size, d)
        return jax.lax.map(jax.vmap(rhs), blocks).reshape(-1, d)[:m]

    return evaluate


def _conjugate_gradient(operator, rhs, tol, maxiter):
    """Zero-start CG for a positive-definite operator, including iteration count."""
    rhs_squared = jnp.vdot(rhs, rhs).real
    threshold = tol**2 * rhs_squared
    initial = (jnp.array(0), jnp.zeros_like(rhs), rhs, rhs, rhs_squared, jnp.array(True))

    def condition(state):
        count, _, _, _, residual_squared, valid = state
        return (count < maxiter) & (residual_squared > threshold) & valid

    def iteration(state):
        count, solution, residual, direction, residual_squared, _ = state
        product = operator(direction)
        curvature = jnp.vdot(direction, product).real
        valid = jnp.isfinite(curvature) & (curvature > 0)
        alpha = residual_squared / jnp.where(valid, curvature, 1.0)
        solution = solution + alpha * direction
        residual = residual - alpha * product
        updated_squared = jnp.vdot(residual, residual).real
        beta = updated_squared / residual_squared
        direction = residual + beta * direction
        return (count + 1, solution, residual, direction, updated_squared,
                valid & jnp.isfinite(updated_squared))

    count, solution, _, _, _, valid = jax.lax.while_loop(condition, iteration, initial)
    return solution, count, valid


def make_parameter_velocity_solver(score_function, regularization, tol, maxiter):
    """Build a matrix-free solve of (J.T W J + lambda I) xi = J.T W R.

    ``score_function(theta, v)`` returns an (n, d) score array. JAX's linearized
    forward map and its transpose provide JVPs/VJPs without storing either J
    or the parameter-by-parameter normal matrix. The particle weights multiply
    the squared vector residual; there is no extra division by d or n.
    """
    @jax.jit
    def solve(theta, v, target, weights):
        _, push = jax.linearize(lambda parameters: score_function(parameters, v), theta)
        pull = jax.linear_transpose(push, jnp.zeros_like(theta))

        def normal(vector):
            return pull(weights[:, None] * push(vector))[0] + regularization * vector

        rhs = pull(weights[:, None] * target)[0]
        velocity, iterations, valid = _conjugate_gradient(normal, rhs, tol, maxiter)
        residual = push(velocity) - target
        fit_residual_squared = jnp.sum(weights[:, None] * residual**2)
        rhs_norm = jnp.linalg.norm(rhs)
        # Measure the true residual as well as CG's recursive stopping criterion.
        linear_residual = jnp.linalg.norm(normal(velocity) - rhs)
        relative_residual = linear_residual / jnp.maximum(rhs_norm, jnp.finfo(rhs.dtype).tiny)
        velocity_squared = jnp.vdot(velocity, velocity).real
        return velocity, dict(
            iterations=iterations, valid=valid,
            converged=valid & (linear_residual <= tol * rhs_norm),
            initial_loss=jnp.sum(weights[:, None] * target**2),
            final_loss=fit_residual_squared + regularization * velocity_squared,
            fit_residual_squared=fit_residual_squared,
            linear_residual=linear_residual, linear_relative_residual=relative_residual,
            parameter_velocity_norm=jnp.sqrt(velocity_squared),
        )

    return solve


class ScoreEvolutionStepper:
    """Coupled equal-weight Maxwell particles and network parameter evolution.

    Matches the BKW experiment's start_step/evaluate/advance/summary interface.
    The supplied model must already be fitted to the initial score. Its public
    NNX parameters are updated in-place after each complete step, so off-particle
    score plots use the evolved network too. Intermediate states are private.
    Energy-conserving steps use one score-evolution solve at step start and
    reevaluate that step's unchanged network at all particle stages. The coupled
    method remains first order because the parameter update is forward Euler.
    """

    def __init__(self, args, model, initial_particles, *, log_fit=None):
        from flax import nnx

        if model is None:
            raise ValueError("Score evolution requires an initialized score network")
        if args.time_integrator not in ("forward_euler", "midpoint", "energy_conserving"):
            raise ValueError("Score evolution supports forward_euler, midpoint, and energy_conserving")
        self.args = args
        self.model = model
        self.log_fit = log_fit
        self.rhs_method = args.score_evolution_rhs
        self.weights = jnp.full((len(initial_particles),), 1 / len(initial_particles),
                                dtype=initial_particles.dtype)
        self.bandwidth = None
        if self.rhs_method == "kernel":
            # Freeze h: changing it would require additional time derivatives.
            supplied_h = args.score_evolution_bandwidth
            self.bandwidth = (scott_bandwidth(initial_particles) if supplied_h is None
                              else jnp.broadcast_to(jnp.asarray(supplied_h, dtype=initial_particles.dtype),
                                                    (initial_particles.shape[1],)))
        graphdef, parameters, other_state = nnx.split(model, nnx.Param, ...)
        # NNX Linear's param_dtype can differ from its computation dtype.
        # Solve/update in the particle dtype, including in float64 experiments.
        parameters = jax.tree_util.tree_map(lambda value: value.astype(initial_particles.dtype), parameters)
        self.theta, self._unravel = ravel_pytree(parameters)
        nnx.update(model, parameters)
        unravel = self._unravel

        def score(parameters, v):
            network = nnx.merge(graphdef, unravel(parameters), other_state)
            return network(jnp.empty((len(v), 0), dtype=v.dtype), v)

        self._score = jax.jit(score)
        self._transport_rhs = (make_transport_score_rhs(score, args.B, args.block_size)
                               if self.rhs_method == "transport" else None)
        self._solve = make_parameter_velocity_solver(
            score, args.score_evolution_regularization,
            args.score_evolution_cg_tol, args.score_evolution_cg_maxiter)
        self.optimization_steps = 0
        self.total_optimization_steps = 0
        self.linear_solves = 0
        self.unconverged_linear_solves = 0
        self.max_linear_relative_residual = 0.0
        self.total_problematic_particles = 0

    def evaluate(self, v, t):
        """Evaluate the current network without fitting or advancing it."""
        score = self._score(self.theta, v)
        return score, maxwell_collision(v, score)

    def start_step(self, v, t, step):
        self.optimization_steps = 0
        return self.evaluate(v, t)

    def _parameter_velocity(self, theta, v, acceleration, t, step, stage):
        started = time.perf_counter()
        if self.rhs_method == "transport":
            target = self._transport_rhs(theta, v, v, self.weights)
        else:
            target = gaussian_score_rhs(v, v, acceleration, self.bandwidth,
                                        self.weights, self.args.block_size)
        velocity, diagnostics = self._solve(theta, v, target, self.weights)
        values = {key: float(value) for key, value in diagnostics.items()
                  if key not in ("iterations", "valid", "converged")}
        if (not bool(diagnostics["valid"]) or not all(map(math.isfinite, values.values()))
                or not bool(jnp.all(jnp.isfinite(velocity)))):
            raise FloatingPointError(f"Invalid score-evolution solve at t={t}, stage={stage}")
        iterations = int(diagnostics["iterations"])
        converged = bool(diagnostics["converged"])
        self.optimization_steps += iterations
        self.total_optimization_steps += iterations
        self.linear_solves += 1
        self.unconverged_linear_solves += int(not converged)
        self.max_linear_relative_residual = max(self.max_linear_relative_residual,
                                                values["linear_relative_residual"])
        if not converged:
            warnings.warn("Score-evolution CG missed its residual tolerance; inspect optimization.csv "
                          "and consider increasing --score_evolution_cg_maxiter or regularization.",
                          RuntimeWarning, stacklevel=2)
        if self.log_fit is not None:
            self.log_fit(dict(step=step + 1, time=t, stage=stage,
                              stage_name="n" if stage == 0 else "midpoint",
                              score_evolution_rhs=self.rhs_method,
                              optimization_steps=iterations, cg_converged=converged,
                              stop_reason="converged" if converged else "residual_tolerance_not_met",
                              **values, optimization_seconds=time.perf_counter() - started))
        return velocity

    def advance(self, v, t, dt, step, initial_evaluation=None):
        from flax import nnx

        self.optimization_steps = 0
        initial = self.evaluate(v, t) if initial_evaluation is None else initial_evaluation
        # maxwell_collision excludes B; apply the prefactor exactly once.
        acceleration = -self.args.B * initial[1]
        theta_n = self.theta
        parameter_velocity = self._parameter_velocity(theta_n, v, acceleration, t, step, 0)
        if self.args.time_integrator == "midpoint":
            theta_half = theta_n + (0.5 * dt) * parameter_velocity
            v_half = v + (0.5 * dt) * acceleration
            score_half = self._score(theta_half, v_half)
            acceleration = -self.args.B * maxwell_collision(v_half, score_half)
            parameter_velocity = self._parameter_velocity(
                theta_half, v_half, acceleration, t + 0.5 * dt, step, 1)
        # Both full updates start at n, including for the midpoint method.
        theta_new = theta_n + dt * parameter_velocity
        if self.args.time_integrator == "energy_conserving":
            # Keep theta_n at every velocity stage. Only the particle time
            # update changes; the score still uses one Euler/CG update at n.
            v_new, _, gamma_squared, problematic = homogeneous_step(
                v, t, dt, self.args.B,
                lambda velocity, time, stage: self.evaluate(velocity, time),
                time_integrator="energy_conserving", initial_evaluation=initial)
            count = int(jnp.sum(problematic))
            gamma_min = float(jnp.min(gamma_squared))
        else:
            v_new = v + dt * acceleration
            count, gamma_min = 0, 1.0
        if not bool(jnp.all(jnp.isfinite(theta_new))) or not bool(jnp.all(jnp.isfinite(v_new))):
            raise FloatingPointError(f"Nonfinite score-evolution state after t={t}")
        self.theta = theta_new
        nnx.update(self.model, self._unravel(theta_new))
        self.total_problematic_particles += count
        return v_new, dict(optimization_steps=self.optimization_steps,
                           gamma_squared_min=gamma_min if math.isfinite(gamma_min) else None,
                           problematic_particle_count=count)

    def summary(self):
        return dict(training_passes=0, linear_solves=self.linear_solves,
                    optimization_method="conjugate_gradient",
                    score_evolution_rhs=self.rhs_method,
                    total_optimization_steps=self.total_optimization_steps,
                    unconverged_linear_solves=self.unconverged_linear_solves,
                    max_linear_relative_residual=self.max_linear_relative_residual,
                    total_problematic_particle_count=self.total_problematic_particles,
                    score_evolution_bandwidth=(self.bandwidth.tolist()
                                               if self.bandwidth is not None else None))
