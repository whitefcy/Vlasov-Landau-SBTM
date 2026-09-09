"""Score fitting with a fixed monitoring batch and per-evaluation diagnostics."""
import time

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx

from src.loss import implicit_score_matching_loss


@nnx.jit(static_argnames=('div_mode',))
def _update(model, optimizer, x, v, key, div_mode):
    def objective(model):
        return implicit_score_matching_loss(model, x, v, key, div_mode=div_mode)
    value, grads = nnx.value_and_grad(objective)(model)
    # Flax <0.11 keeps the model on the optimizer; newer releases require it.
    if hasattr(optimizer, 'model'):
        optimizer.update(grads)
    else:
        optimizer.update(model, grads)
    return value


def fit_score(model, optimizer, x, v, key, *, batch_size, max_steps=100,
              adaptive=True, min_steps=10, check_every=5, patience=3,
              atol=1e-5, rtol=1e-4, monitor_size=2048, div_mode='exact'):
    """Stop after patience consecutive small changes in a fixed-batch loss.

    The monitor always uses exact divergence, including in Hutchinson ablations.
    A small change means abs(new-old) <= atol + rtol*max(abs(old), abs(new)).
    This is a plateau heuristic, not a bound on score error. The optimizer and
    model are warm-started across calls; monitoring state resets at each stage.
    """
    if min(batch_size, max_steps, min_steps, check_every, patience, monitor_size) < 1:
        raise ValueError('Step counts and batch sizes must be positive')
    if adaptive and min_steps > max_steps:
        raise ValueError('min_steps must not exceed max_steps')
    if not (0 <= atol < float('inf') and 0 <= rtol < float('inf')):
        raise ValueError('Tolerances must be finite and nonnegative')
    if x.shape[0] != v.shape[0] or x.shape[0] == 0:
        raise ValueError('Expected equal, nonempty particle arrays')
    started = time.perf_counter()
    key, monitor_key = jr.split(key)
    ids = jr.permutation(monitor_key, x.shape[0])[:monitor_size]
    xm, vm = x[ids], v[ids]

    def monitor():
        value = float(implicit_score_matching_loss(
            model, xm, vm, monitor_key, div_mode='exact'))
        if not jnp.isfinite(value):
            raise FloatingPointError('Nonfinite score-matching monitor loss')
        return value

    initial = previous = monitor()
    stable = 0
    checks = 0
    reason = 'max_steps' if adaptive else 'fixed_steps'
    # Random batches without replacement within each pass through the particles.
    offset = x.shape[0]
    for steps in range(1, max_steps + 1):
        if offset >= x.shape[0]:
            key, permutation_key = jr.split(key)
            order = jr.permutation(permutation_key, x.shape[0])
            offset = 0
        ids = order[offset:offset + batch_size]
        offset += batch_size
        key, update_key = jr.split(key)
        train_loss = _update(model, optimizer, x[ids], v[ids], update_key, div_mode)
        if steps % check_every == 0 or steps == max_steps:
            if not bool(jnp.isfinite(train_loss)):
                raise FloatingPointError('Nonfinite score-matching training loss')
            current = monitor()
            checks += 1
            small = abs(current - previous) <= atol + rtol * max(abs(previous), abs(current))
            stable = stable + 1 if small and steps >= min_steps else 0
            previous = current
            if adaptive and stable >= patience:
                reason = 'plateau'
                break
    return dict(optimization_steps=steps, stop_reason=reason,
                initial_loss=initial, final_loss=previous, monitor_checks=checks,
                optimization_seconds=time.perf_counter() - started)
