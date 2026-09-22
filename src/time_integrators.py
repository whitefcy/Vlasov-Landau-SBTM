"""Velocity updates shared with the project's energy-conserving PIC scheme."""

import jax
import jax.numpy as jnp


@jax.jit
def gamma_correct_velocity(v, v_star, v_dagger):
    """Per-particle Gamma correction from scheme (3.7).

    Return corrected velocities, raw Gamma^2, and the fallback mask. As in
    the PIC solver, invalid corrections use Gamma=1 and are reported; energy
    conservation is not guaranteed when a fallback occurs. This correction
    does not impose momentum conservation.
    """
    delta_v = v_dagger - v
    energy_direction = v_star - 0.5 * (v_dagger + v)
    speed_squared = jnp.sum(v_dagger ** 2, axis=1)
    correction = 2.0 * jnp.sum(delta_v * energy_direction, axis=1)
    has_valid_denominator = speed_squared > jnp.finfo(v.dtype).tiny
    safe_denominator = jnp.where(has_valid_denominator, speed_squared, 1.0)
    raw_gamma_squared = 1.0 + correction / safe_denominator
    problematic_particles = (
        (~has_valid_denominator)
        | (~jnp.isfinite(raw_gamma_squared))
        | (raw_gamma_squared < 0.0)
    )
    safe_gamma_squared = jnp.where(problematic_particles, 1.0, raw_gamma_squared)
    v_new = jnp.sqrt(safe_gamma_squared)[:, None] * v_dagger
    return v_new, raw_gamma_squared, problematic_particles


def homogeneous_step(v, t, dt, B, collision_evaluator, *,
                     time_integrator="energy_conserving", initial_evaluation=None):
    """Collision-only specialization of utils.energy_conserving_step.

    evaluator(v_stage, time, stage) returns (score, collision), excluding B.
    Stages n, **, * use physical times t, t+dt/2, t+dt/2. A previously
    prepared stage-0 evaluation can be reused without a second training pass.
    """
    if time_integrator not in ("forward_euler", "energy_conserving"):
        raise ValueError(f"Unsupported time integrator: {time_integrator}")
    initial = initial_evaluation
    if initial is None:
        initial = collision_evaluator(v, t, 0)
    stages = [initial]
    if time_integrator == "forward_euler":
        return (v - dt * B * initial[1], stages, jnp.ones(len(v), dtype=v.dtype),
                jnp.zeros(len(v), dtype=bool))

    half_dt = 0.5 * dt
    v_starstar = v - half_dt * B * initial[1]
    stages.append(collision_evaluator(v_starstar, t + half_dt, 1))
    v_star = v - half_dt * B * stages[1][1]
    stages.append(collision_evaluator(v_star, t + half_dt, 2))
    v_dagger = v - dt * B * stages[2][1]
    v_new, gamma_squared, problematic = gamma_correct_velocity(v, v_star, v_dagger)
    return v_new, stages, gamma_squared, problematic
