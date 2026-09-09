import jax
import jax.numpy as jnp
from flax import nnx
from typing import Callable

def divergence_wrt_v(f: Callable, mode: str, n: int = 100):
    """
    Compute the divergence of a vector field f with respect to v, where f(x,v) -> output with shape of v.
    
    Args:
        f: Callable
            Function of form f(x,v) that returns output with same shape as v.
        mode: str
            Mode of divergence computation.
        n: int
            Number of samples for stochastic estimation.
    
    Returns:
        Callable that computes the divergence at a point (x,v) with respect to v.
    """
    # gaussian and rademacher are ~30% faster than exact methods and denoised
    assert mode in ['forward', 'reverse', 'approximate_gaussian', 'approximate_rademacher', 'denoised'], "Invalid mode"
    
    # Create a wrapper that treats only v as the variable for differentiation
    def f_wrapper(v, x):
        return f(x, v)
    
    if mode == 'forward':
        @jax.jit
        def div(x, v):
            return jnp.trace(jax.jacfwd(f_wrapper, argnums=0)(v, x))
        return div
        
    if mode == 'reverse':
        @jax.jit
        def div(x, v):
            return jnp.trace(jax.jacrev(f_wrapper, argnums=0)(v, x))
        return div
        
    if mode == 'denoised':
        alpha = jnp.float32(0.1)
        @jax.jit
        def div(x, v, key):
            def denoise(key):
                epsilon = jax.random.normal(key, v.shape, dtype=v.dtype)
                return jnp.sum(
                    (f(x, v + alpha * epsilon) - f(x, v - alpha * epsilon)) * epsilon
                ) / (2*alpha)
            return jax.vmap(denoise)(jax.random.split(key, n)).mean()
        return div
    else:
        @jax.jit
        def div(x, v, key):
            def vJv(key):
                # Define a partial function that fixes x
                fixed_x_f = lambda v_: f(x, v_)
                # Get vector-Jacobian product function
                _, vjp_fun = jax.vjp(fixed_x_f, v)
                # Generate random vector
                rand_gen = jax.random.normal if mode == 'approximate_gaussian' else jax.random.rademacher
                epsilon = rand_gen(key, v.shape, dtype=v.dtype)
                # Compute vᵀ(∂f/∂v)ᵀv = vᵀJ_v[f]ᵀv
                return jnp.sum(vjp_fun(epsilon)[0] * epsilon)
            return jax.vmap(vJv)(jax.random.split(key, n)).mean()
        return div

@nnx.jit
def explicit_score_matching_loss(s, x_batch, v_batch, target_score_values):
    """
    Compute the score matching loss between the score function s and the true score.
    1/n ∑ᵢ ||s(xᵢ,vᵢ) - ∇ᵥlog p(xᵢ,vᵢ)||²
    
    Args:
        s: Score model function that takes (x,v) and returns score
        x_batch: Position batch of shape (batch_size, dx)
        v_batch: Velocity batch of shape (batch_size, dv)
        target_score_values: True score values of shape (batch_size, dv)
    
    Returns:
        Mean squared error loss
    """
    # Compute predictions for all samples in the batch
    predicted_scores = s(x_batch, v_batch)
    # predicted_scores = jax.vmap(s)(x_batch, v_batch)
    
    # Compute mean squared error
    return jnp.mean(jnp.sum(jnp.square(predicted_scores - target_score_values), axis=-1))

@jax.jit
def mse(predictions, targets):
    return jnp.sum(jnp.square(predictions - targets)) / predictions.shape[0]

@nnx.jit
def weighted_explicit_score_matching_loss(s, x_batch, v_batch, target_score_values, weighting):
    """
    Compute the weighted score matching loss between the score function s and the true score.
    1/n ∑ᵢ ⟨s(xᵢ,vᵢ) - ∇ᵥlog p(xᵢ,vᵢ), D[i] (s(xᵢ,vᵢ) - ∇ᵥlog p(xᵢ,vᵢ))⟩
    
    Args:
        s: Score model function that takes (x,v) and returns score
        x_batch: Position batch of shape (batch_size, dx)
        v_batch: Velocity batch of shape (batch_size, dv)
        target_score_values: True score values of shape (batch_size, dv)
        weighting: Weighting matrices of shape (batch_size, dv, dv) or (dv, dv)
    
    Returns:
        Weighted mean squared error loss
    """
    def weighted_loss(x, v, target, D):
        pred = s(x, v)
        diff = pred - target
        # If D is (dv, dv), dot product is diff^T @ D @ diff
        # If D is a scalar, it's equivalent to diff^T @ D*I @ diff = D * (diff^T @ diff)
        if D.ndim == 2:
            return jnp.dot(diff, jnp.dot(D, diff))
        else:
            return jnp.dot(diff, diff) * D
    
    return jnp.mean(jax.vmap(weighted_loss)(x_batch, v_batch, target_score_values, weighting))

@nnx.jit(static_argnames=('div_mode', 'n_samples'))
def implicit_score_matching_loss(s, x_batch, v_batch, key, div_mode='approximate_rademacher', n_samples: int = 1):
    """
    1/|B| ∑ (‖s(x,v)‖² + 2 div_v s(x,v)).
    Exact mode traces the velocity Jacobian; approximate_rademacher uses
    independent Hutchinson probes per particle. Exact mode ignores key.
    """
    if div_mode not in ('exact', 'approximate_rademacher'):
        raise ValueError(f"Unsupported divergence mode: {div_mode}")
    if x_batch.ndim < v_batch.ndim:
        x_batch = x_batch[:, None]
    if div_mode == 'exact':
        # Hold x fixed: div_v s = sum_j (d s_j / d v_j).
        # jacfwd uses the dv coordinate directions, with no random probes.
        def exact_one(x, v):
            score = s(x, v)
            jac = jax.jacfwd(lambda vv: s(x, vv))(v)
            return jnp.sum(score * score) + 2.0 * jnp.trace(jac)
        return jax.vmap(exact_one)(x_batch, v_batch).mean()
    # ε tensor:  (n_samples, B, dv)
    eps = jax.random.rademacher(key, (n_samples,) + v_batch.shape, dtype=v_batch.dtype)

    def loss_one(x, v, eps_v):
        score = s(x, v)
        # vmap over ε samples for this (x,v)
        def one_eps(e):
            _, jvp = jax.jvp(lambda vv: s(x, vv), (v,), (e,))
            return jnp.vdot(jvp, e)
        div = jax.vmap(one_eps)(eps_v).mean()
        return jnp.sum(score * score) + 2.0 * div

    # transpose so each particle gets its (n_samples,dv) slice
    eps_per_particle = eps.transpose(1, 0, 2)          # (B, n_samples, dv)
    batch_loss = jax.vmap(loss_one)(x_batch, v_batch, eps_per_particle)
    return batch_loss.mean()