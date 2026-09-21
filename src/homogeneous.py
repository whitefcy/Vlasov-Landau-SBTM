"""Shared Maxwell collision, KDE, CLI, and initialization for homogeneous experiments."""

import argparse
from functools import partial
import math
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import jax.random as jr


@jax.jit
def maxwell_collision(v, score):
    """Exactly evaluate mean_j (|z|^2 I - z z^T)(s_i-s_j), excluding B.

    Expanding the polynomial Maxwell kernel reduces the all-pairs sum to
    moments in O(n*d^2) work and O(n*d) storage. Centering velocities and
    scores improves conditioning and leaves every pairwise difference intact.
    """
    u = v - jnp.mean(v, axis=0)
    s = score - jnp.mean(score, axis=0)
    n = v.shape[0]
    r2 = jnp.sum(u**2, axis=1, keepdims=True)
    us = jnp.sum(u * s, axis=1, keepdims=True)
    second = u.T @ u / n
    cross = u.T @ s / n
    return (
        (r2 + jnp.trace(second)) * s - u * us - s @ second
        - jnp.mean(r2 * s, axis=0) + 2 * u @ cross
        - u * jnp.trace(cross) - u @ cross.T + jnp.mean(u * us, axis=0)
    )


def scott_bandwidth(v):
    """Per-coordinate standard deviations h; Gaussian covariance is diag(h^2)."""
    n, d = v.shape
    return jnp.maximum(jnp.std(v, axis=0, ddof=1), jnp.finfo(v.dtype).eps) * n**(-1 / (d + 4))


@partial(jax.jit, static_argnames=("block_size",))
def gaussian_kde(points, centers, h, block_size=128):
    """Gaussian KDE density and score, including self terms at particle queries.

    Pairwise work is O(m*n*d), but only block_size*n*d entries are materialized.
    Log-sum-exp scaling keeps scores stable for queries in the tails.
    """
    m, d = points.shape
    padding = (-m) % block_size
    blocks = jnp.pad(points, ((0, padding), (0, 0))).reshape(-1, block_size, d)

    def evaluate(block):
        delta = centers[None, :, :] - block[:, None, :]
        exponent = -0.5 * jnp.sum((delta / h)**2, axis=-1)
        shift = jnp.max(exponent, axis=1, keepdims=True)
        weights = jnp.exp(exponent - shift)
        total = jnp.sum(weights, axis=1)
        density = jnp.exp(shift[:, 0]) * total / (centers.shape[0] * (2 * jnp.pi)**(d / 2) * jnp.prod(h))
        score = jnp.sum(weights[:, :, None] * delta, axis=1) / (total[:, None] * h**2)
        return density, score

    density, score = jax.lax.map(evaluate, blocks)
    return density.reshape(-1)[:m], score.reshape(-1, d)[:m]


def make_parser(description, *, score_methods=("sbtm", "blob", "exact")):
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    p.add_argument("--n", type=int, default=12800)
    p.add_argument("--dv", type=int, default=3)
    p.add_argument("--B", type=float, default=1 / 24)
    p.add_argument("--t0", type=float, default=5.5)
    p.add_argument("--final_time", type=float, default=9.5, help="Physical end time, not duration")
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--score_method", choices=score_methods, default="sbtm")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", default=None, help="Optional CUDA index; otherwise preserve Slurm visibility")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--density_every", type=int, default=100, help="Frequency of the O(n^2) density L2 measurement")
    p.add_argument("--snapshot_every", type=int, default=100)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--sbtm_hidden_dims", type=int, nargs="+", default=[100, 100])
    p.add_argument("--sbtm_batch_size", type=int, default=1024)
    p.add_argument("--sbtm_num_epochs", type=int, default=1000, help="Maximum supervised initialization epochs")
    p.add_argument("--sbtm_abs_tol", type=float, default=1e-4)
    p.add_argument("--sbtm_lr", type=float, default=4e-4)
    p.add_argument("--sbtm_num_batch_steps", type=int, default=100)
    p.add_argument("--sbtm_adaptive", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sbtm_div_mode", choices=["exact", "approximate_rademacher"], default="exact")
    p.add_argument("--sbtm_min_steps", type=int, default=10)
    p.add_argument("--sbtm_check_every", type=int, default=5)
    p.add_argument("--sbtm_patience", type=int, default=3)
    p.add_argument("--sbtm_stop_atol", type=float, default=1e-5)
    p.add_argument("--sbtm_stop_rtol", type=float, default=1e-4)
    p.add_argument("--sbtm_monitor_size", type=int, default=2048)
    p.add_argument("--wandb_project", default="homo_BKW")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="disabled")
    return p


def validate_args(args, p):
    positive = ["B", "dt", "block_size", "log_every", "density_every", "snapshot_every",
                "sbtm_batch_size", "sbtm_num_epochs", "sbtm_lr", "sbtm_num_batch_steps",
                "sbtm_min_steps", "sbtm_check_every", "sbtm_patience", "sbtm_monitor_size"]
    if any(not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0 for name in positive):
        p.error("Step sizes, counts, B, and learning rate must be finite and positive")
    if args.n < 2 or args.dv < 2 or min(args.sbtm_hidden_dims) < 1:
        p.error("Require n >= 2, dv >= 2, and positive hidden dimensions")
    if not math.isfinite(args.t0):
        p.error("t0 must be finite")
    if not math.isfinite(args.final_time) or args.final_time < args.t0:
        p.error("final_time must be finite and >= t0 (both are physical times)")
    if args.sbtm_adaptive and args.sbtm_min_steps > args.sbtm_num_batch_steps:
        p.error("sbtm_min_steps must not exceed sbtm_num_batch_steps in adaptive mode")
    for name in ["sbtm_abs_tol", "sbtm_stop_atol", "sbtm_stop_rtol"]:
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            p.error(f"{name} must be finite and nonnegative")
    return args


def initialize_model(args, v, key, target):
    from flax import nnx
    import optax
    from src.score_model import MLPScoreModel

    # Empty position inputs retain the shared model/loss API with only d inputs.
    x = jnp.empty((len(v), 0), dtype=v.dtype)
    model = MLPScoreModel(0, args.dv, hidden_dims=tuple(args.sbtm_hidden_dims), seed=args.seed, dtype=v.dtype)
    optimizer = nnx.Optimizer(model, optax.adam(args.sbtm_lr), wrt=nnx.Param)

    @nnx.jit
    def initial_update(model, optimizer, vb, sb):
        def objective(model):
            pred = model(jnp.empty((len(vb), 0), dtype=vb.dtype), vb)
            return jnp.mean(jnp.sum((pred - sb)**2, axis=1))
        value, grads = nnx.value_and_grad(objective)(model)
        if hasattr(optimizer, "model"):
            optimizer.update(grads)
        else:
            optimizer.update(model, grads)
        return value

    started = time.perf_counter()
    for epoch in range(args.sbtm_num_epochs):
        loss = float(jnp.mean(jnp.sum((model(x, v) - target)**2, axis=1)))
        if not math.isfinite(loss):
            raise FloatingPointError("Nonfinite initial score loss")
        if epoch % 100 == 0:
            print(f"Initial score epoch {epoch}: MSE={loss:.6g}", flush=True)
        if loss <= args.sbtm_abs_tol:
            break
        key, batch_key = jr.split(key)
        order = jr.permutation(batch_key, len(v))
        for offset in range(0, len(v), args.sbtm_batch_size):
            ids = order[offset:offset + args.sbtm_batch_size]
            initial_update(model, optimizer, v[ids], target[ids])
    loss = float(jnp.mean(jnp.sum((model(x, v) - target)**2, axis=1)))
    if not math.isfinite(loss):
        raise FloatingPointError("Nonfinite final initialization loss")
    print(f"Initial score fit finished: MSE={loss:.6g}, tolerance_met={loss <= args.sbtm_abs_tol}", flush=True)
    # Use a fresh optimizer for implicit fitting after the supervised t0 fit.
    optimizer = nnx.Optimizer(model, optax.adam(args.sbtm_lr), wrt=nnx.Param)
    return model, optimizer, dict(initial_score_mse=loss, initial_fit_seconds=time.perf_counter() - started,
                                 initial_tolerance_met=loss <= args.sbtm_abs_tol)

