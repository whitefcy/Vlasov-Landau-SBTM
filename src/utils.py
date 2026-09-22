import jax
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
import math

from src import loss
from src.time_integrators import gamma_correct_velocity
from flax import nnx
import optax

# ------------------------------------------------------------------------------
# Visualization utilities
# ------------------------------------------------------------------------------
def plot_U_quiver_pred(v, U, label, U_true=None, num_points=500, figsize=(5, 5), scale=1):
    if U_true is None:
        assert v.shape == U.shape
    else:
        assert v.shape == U.shape == U_true.shape

    n = v.shape[0]
    step_sub = max(1, n // num_points)
    v_plot = v[::step_sub]
    U_plot = U[::step_sub]
    U_true_plot = None if U_true is None else U_true[::step_sub]

    if U_true is None:
        plot_label = label
    else:
        mse = float(jnp.mean((U - U_true) ** 2))
        plot_label = f"{label} flow n={n:.0e} mse={mse:.5f}"

    fig_quiver = plt.figure(figsize=figsize)
    plt.quiver(
        v_plot[:, 0],
        v_plot[:, 1],
        U_plot[:, 0],
        U_plot[:, 1],
        # color="tab:blue",
        alpha=0.8,
        scale=scale,
        angles="xy",
        scale_units="xy",
        label=plot_label,
    )

    if U_true is not None:
        plt.quiver(
            v_plot[:, 0],
            v_plot[:, 1],
            U_true_plot[:, 0],
            U_true_plot[:, 1],
            color="tab:red",
            alpha=0.5,
            scale=scale,
            angles="xy",
            scale_units="xy",
            label="Reference flow",
        )

    plt.scatter(v_plot[:, 0], v_plot[:, 1], s=2, alpha=0.3)
    plt.axis("equal")
    plt.xlabel("v1")
    plt.ylabel("v2")
    plt.title(f"Estimated flow U: {label}")
    plt.legend(loc="best")
    plt.tight_layout()
    return fig_quiver

def plot_score_quiver_pred(v, score, label, num_points=500, figsize=(5, 5), scale=5):
    assert v.shape == score.shape
    n = v.shape[0]
    step_sub = max(1, n // num_points)
    v_plot = v[::step_sub]
    score_plot = score[::step_sub]

    fig_quiver = plt.figure(figsize=figsize)
    plt.quiver(
        v_plot[:, 0],
        v_plot[:, 1],
        score_plot[:, 0],
        score_plot[:, 1],
        alpha=0.8,
        scale=scale,
        angles="xy",
        scale_units="xy",
        label=label,
    )
    plt.scatter(v_plot[:, 0], v_plot[:, 1], s=2, alpha=0.3)
    plt.axis("equal")
    plt.xlabel("v1")
    plt.ylabel("v2")
    plt.title(f"Estimated scores: {label}")
    plt.legend(loc="best")
    plt.tight_layout()
    return fig_quiver

def plot_score_quiver(v, score, score_true, label, num_points=500, figsize=(5, 5), scale=5):
    assert v.shape == score.shape == score_true.shape
    n = v.shape[0]
    step_sub = max(1, n // num_points)
    v_plot = v[::step_sub]
    score_plot = score[::step_sub]
    score_true_plot = score_true[::step_sub]

    fig_quiver = plt.figure(figsize=figsize)
    plt.quiver(
        v_plot[:, 0],
        v_plot[:, 1],
        score_plot[:, 0],
        score_plot[:, 1],
        color="tab:blue",
        alpha=0.8,
        scale=scale,
        angles="xy",
        scale_units="xy",
        label=f"{label} score n={n:.0e} mse={float(jnp.mean((score - score_true)**2)):.5f}",
    )
    plt.quiver(
        v_plot[:, 0],
        v_plot[:, 1],
        score_true_plot[:, 0],
        score_true_plot[:, 1],
        color="tab:red",
        alpha=0.5,
        scale=scale,
        angles="xy",
        scale_units="xy",
        label="Reference score",
    )
    plt.scatter(v_plot[:, 0], v_plot[:, 1], s=2, c="k", alpha=0.3)
    plt.axis("equal")
    plt.xlabel("v1")
    plt.ylabel("v2")
    plt.title(f"Estimated scores: {label}")
    plt.legend(loc="best")
    plt.tight_layout()
    return fig_quiver

def visualize_initial(x, v, cells, E, rho, eta, L, spatial_density, v_target):
    """Visualize initial data and return the figure."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Histogram of x and desired density
    axs[0].hist(x, bins=50, density=True, alpha=0.4, label='Sampled $x$')
    x_grid = jnp.linspace(0, L, 200)
    axs[0].plot(x_grid, spatial_density(x_grid), 'r-', label='Target density')
    axs[0].plot(cells, rho / L, 'g-', label='$\\rho/L$')
    axs[0].set_title('Position $x$')
    axs[0].set_xlabel('$x$')
    axs[0].legend()

    # 2. Histogram of v and target mixture
    axs[1].hist(v, bins=50, density=True, alpha=0.4, label='Sampled $v$')
    v_grid = jnp.linspace(v.min() - 1, v.max() + 1, 200)
    axs[1].plot(v_grid, v_target(v_grid), 'r-', label='Target $f(v)$')
    axs[1].set_title('Velocity $v$')
    axs[1].set_xlabel('$v$')
    axs[1].legend()

    # 3. E, dE/dx, and rho
    axs[2].plot(cells, E, label='$E$')
    dE_dx = jnp.gradient(E, eta)
    axs[2].plot(cells, dE_dx, label='$dE/dx$')
    axs[2].plot(cells, rho - 1, label=r'$\rho - \rho_i$')
    axs[2].set_title('Field $E$, $dE/dx$, and $\\rho$')
    axs[2].set_xlabel('x')
    axs[2].legend()

    plt.tight_layout()
    return fig

def plot_phase_space_snapshots(x_traj, v_traj, t_traj, L, title, outdir=None, fname=None, bins=None, save=True):
    num_snaps = len(x_traj)
    if num_snaps == 0:
        return None, None
    if bins is None:
        k = int(math.sqrt(x_traj[0].shape[0] / 50))
        bins = [max(20, k), max(20, k)]

    cols = min(3, num_snaps)
    rows = int(np.ceil(num_snaps / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows),
                            sharex=True, sharey=True)
    axs = np.array(axs).reshape(-1)

    v_all = np.concatenate([np.asarray(v_snap)[:, 0] for v_snap in v_traj])
    vmin, vmax = float(v_all.min()), float(v_all.max())

    last_img = None
    for i, (x_snap, v_snap, t_snap) in enumerate(zip(x_traj, v_traj, t_traj)):
        ax = axs[i]
        xs = np.asarray(x_snap) % float(L)
        vs = np.asarray(v_snap)[:, 0]
        _, _, _, img = ax.hist2d(
            xs, vs,
            bins=bins,
            range=[[0.0, float(L)], [vmin, vmax]],
            cmap="jet",
            density=True,
        )
        last_img = img
        ax.set_title(f"t = {t_snap:.1f}")
        ax.set_xlim(0.0, float(L))
        ax.set_ylim(vmin, vmax)
        ax.set_xlabel("x")
        ax.set_ylabel("v")

    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    if last_img is not None:
        cbar = fig.colorbar(last_img, ax=axs.tolist(),
                            orientation="vertical", fraction=0.02, pad=0.02)
        cbar.set_label("Density")

    plt.suptitle(title)

    if save:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, fname)
        plt.savefig(path)
    else:
        path = None

    return fig, path

#------------------------------------------------------------------------------
# Reconstruct density from particles
#------------------------------------------------------------------------------
import jax
import jax.numpy as jnp

def make_grid_density_core(z, *, bounds, bins, smooth_sigma_bins=0.0):
    """
    z: (n,D) particles
    bounds: (D,2)
    bins: int or (D,)
    Returns eval(z_query)->(m,)
    """
    z = jnp.asarray(z)
    D = z.shape[1]
    if not (1 <= D <= 3):
        raise ValueError("D must be 1..3")

    bounds = jnp.asarray(bounds)
    bins = jnp.asarray([bins]*D) if jnp.ndim(bins)==0 else jnp.asarray(bins)
    bins = bins.astype(jnp.int32)

    lo, hi = bounds[:,0], bounds[:,1]
    dx = (hi - lo) / bins

    H, _ = jnp.histogramdd(
        z,
        bins=bins.tolist(),
        range=[(float(lo[i]), float(hi[i])) for i in range(D)],
    )
    rho = H / (z.shape[0] * jnp.prod(dx))

    # optional smoothing (separable Gaussian)
    if smooth_sigma_bins and smooth_sigma_bins > 0:
        def gauss1d(s):
            r = int(jnp.ceil(3*s))
            t = jnp.arange(-r, r+1)
            k = jnp.exp(-0.5*(t/s)**2)
            return k / jnp.sum(k)

        k = gauss1d(float(smooth_sigma_bins))
        pad = k.size // 2

        def conv_axis(a, axis):
            a = jnp.moveaxis(a, axis, -1)
            a = jnp.pad(a, [(0,0)]*(a.ndim-1)+[(pad,pad)], mode="edge")
            a2 = a.reshape(-1, a.shape[-1])
            a2 = jax.vmap(lambda r: jnp.convolve(r, k, mode="valid"))(a2)
            a = a2.reshape(*a.shape[:-1], -1)
            return jnp.moveaxis(a, -1, axis)

        for ax in range(D):
            rho = conv_axis(rho, ax)

    @jax.jit
    def eval(zq):
        zq = jnp.asarray(zq)
        u = (zq - lo) / dx
        u = jnp.clip(u, 0.0, bins - 1.000001)
        i0 = jnp.floor(u).astype(jnp.int32)
        t = u - i0

        if D == 1:
            i = i0[:,0]; w = t[:,0]
            return (1-w)*rho[i] + w*rho[i+1]

        if D == 2:
            i,j = i0[:,0], i0[:,1]
            tx,ty = t[:,0], t[:,1]
            return (
                (1-tx)*(1-ty)*rho[i,  j  ] +
                 tx   *(1-ty)*rho[i+1,j  ] +
                (1-tx)* ty   *rho[i,  j+1] +
                 tx   * ty   *rho[i+1,j+1]
            )

        # D == 3
        i,j,k = i0[:,0], i0[:,1], i0[:,2]
        tx,ty,tz = t[:,0], t[:,1], t[:,2]
        c00 = (1-tx)*rho[i,  j,  k  ] + tx*rho[i+1,j,  k  ]
        c10 = (1-tx)*rho[i,  j+1,k  ] + tx*rho[i+1,j+1,k  ]
        c01 = (1-tx)*rho[i,  j,  k+1] + tx*rho[i+1,j,  k+1]
        c11 = (1-tx)*rho[i,  j+1,k+1] + tx*rho[i+1,j+1,k+1]
        c0 = (1-ty)*c00 + ty*c10
        c1 = (1-ty)*c01 + ty*c11
        return (1-tz)*c0 + tz*c1

    return eval

def make_density_v(v, *, bounds_v, bins_v, smooth_sigma_bins=1.5):
    return make_grid_density_core(
        v,
        bounds=bounds_v,
        bins=bins_v,
        smooth_sigma_bins=smooth_sigma_bins,
    )

def make_density_xv(x, v, *, bounds_x, bounds_v, bins_x, bins_v, smooth_sigma_bins=1.5):
    z = jnp.concatenate([x[:,None], v], axis=1)
    bounds = jnp.concatenate([jnp.asarray(bounds_x)[None,:], bounds_v], axis=0)
    bins = jnp.concatenate([jnp.asarray([bins_x]), jnp.asarray(bins_v)])
    core = make_grid_density_core(
        z,
        bounds=bounds,
        bins=bins,
        smooth_sigma_bins=smooth_sigma_bins,
    )

    def eval(vq, xq):
        zq = jnp.concatenate([xq[:,None], vq], axis=1)
        return core(zq)

    return eval

def density_on_regular_grid(
    v, *, bounds_v, bins_per_side, x=None, bounds_x=None,
    smooth_sigma_bins=0.0,
):
    """
    Calculates the density on a regular grid based on the provided velocity and position data.
    This function can evaluate the density either in velocity space or in both velocity and position spaces, depending on the input parameters.

    v: Array of velocity data, shape (N, dv), where N is the number of samples and dv is the number of velocity dimensions.
    bounds_v: Tuple or list specifying the bounds for each velocity dimension, shape (dv, 2).
    bins_per_side: Number of bins per side for the grid. Can be:
        - int: Same number of bins for all dimensions
        - tuple/list: Specific number of bins for each dimension (dv,) for velocity-only or (1+dv,) when x is provided
    x: Optional array of position data, shape (M, dx), where M is the number of samples and dx is the number of position dimensions.
    bounds_x: Optional tuple or list specifying the bounds for each position dimension, shape (dx, 2). Required if x is provided.
    smooth_sigma_bins: Standard deviation for Gaussian smoothing in bins, default is 0.0 (no smoothing).
    return: A density array reshaped according to the number of bins and dimensions.
    """
    dv = v.shape[1]

    # Convert bins_per_side to array
    if isinstance(bins_per_side, (int, float)):
        # Scalar: use same number of bins for all dimensions
        if x is None:
            bins = jnp.array([int(bins_per_side)] * dv)
        else:
            bins = jnp.array([int(bins_per_side)] * (1 + dv))
    else:
        # Tuple/list: use specified bins per dimension
        bins = jnp.array([int(b) for b in bins_per_side])
        expected_dims = dv if x is None else (1 + dv)
        if len(bins) != expected_dims:
            raise ValueError(
                f"bins_per_side must have length {expected_dims} "
                f"({'dv' if x is None else '1+dv'}), got {len(bins)}"
            )

    if x is None:
        eval_fn = make_density_v(
            v,
            bounds_v=jnp.asarray(bounds_v),
            bins_v=tuple(bins),
            smooth_sigma_bins=smooth_sigma_bins,
        )
        bounds = jnp.asarray(bounds_v)
    else:
        if bounds_x is None:
            raise ValueError("bounds_x required when x is provided.")
        eval_fn = make_density_xv(
            x, v,
            bounds_x=jnp.asarray(bounds_x),
            bounds_v=jnp.asarray(bounds_v),
            bins_x=int(bins[0]),
            bins_v=tuple(bins[1:]),
            smooth_sigma_bins=smooth_sigma_bins,
        )
        bounds = jnp.concatenate(
            [jnp.asarray(bounds_x)[None, :], jnp.asarray(bounds_v)],
            axis=0,
        )

    D = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    dx = (hi - lo) / bins

    axes = [lo[d] + (jnp.arange(bins[d]) + 0.5) * dx[d] for d in range(D)]
    mesh = jnp.meshgrid(*axes, indexing="ij")
    pts = jnp.stack([m.reshape(-1) for m in mesh], axis=1)

    if x is None:
        rho = eval_fn(pts)
        return rho.reshape(tuple(bins))
    else:
        rho = eval_fn(pts[:, 1:], pts[:, 0])
        return rho.reshape(tuple(bins))


#------------------------------------------------------------------------------
# Rejection sampling
#------------------------------------------------------------------------------
def rejection_sample(key, density_fn, domain, max_value, num_samples=1):
    """Parallel rejection sampling on [domain[0], domain[1]]."""
    domain_width = domain[1] - domain[0]
    proposal_fn = lambda x: jnp.where((x >= domain[0]) & (x <= domain[1]), 1.0 / domain_width, 0.0)
    max_ratio = max_value / (1.0 / domain_width) * 1.2  # 20% margin
    key, key_propose, key_accept = jr.split(key, 3)

    num_candidates = int(num_samples * max_ratio * 2)
    candidates = jr.uniform(key_propose, minval=domain[0], maxval=domain[1], shape=(num_candidates,))
    proposal_values = proposal_fn(candidates)
    target_values = density_fn(candidates)

    accepted = jr.uniform(key_accept, (num_candidates,)) * max_ratio * proposal_values <= target_values
    samples = candidates[accepted]
    return samples[:num_samples]


#------------------------------------------------------------------------------
# PIC pieces (1D x, dv-D v)
#------------------------------------------------------------------------------
@jax.jit
def evaluate_charge_density(x, cells, eta, w):
    """ρ_j = w * Σ_p ψ_eta(X_p − cell_j) with ψ the hat kernel."""
    M = cells.size
    idx_f = x / eta - 0.5
    i0 = jnp.floor(idx_f).astype(jnp.int32) % M
    i1 = (i0 + 1) % M
    f = idx_f - jnp.floor(idx_f)
    w0, w1 = 1 - f, f
    counts = (
        jnp.zeros(M)
          .at[i0].add(w0)
          .at[i1].add(w1)
    )
    return w / eta * counts


@jax.jit
def evaluate_field_at_particles(E, x, cells, eta):
    """E(x) = Σ_j ψ(x − cell_j) E_j (linear-hat kernel, periodic)."""
    M = cells.size
    idx_f = x / eta - 0.5
    i0 = jnp.floor(idx_f).astype(jnp.int32) % M
    f = idx_f - jnp.floor(idx_f)
    i1 = (i0 + 1) % M
    return (1.0 - f) * E[i0] + f * E[i1]


@jax.jit
def update_electric_field(E, x, v, cells, eta, w, dt):
    """E_j^{n+1} = E_j^n - dt * w * Σ_i ψ(x_i - cell_j) v_i (linear-hat, periodic)."""
    M = cells.size
    idx_f = x / eta - 0.5
    i0 = jnp.floor(idx_f).astype(jnp.int32) % M
    i1 = (i0 + 1) % M
    f = idx_f - jnp.floor(idx_f)
    w0, w1 = 1 - f, f
    J = (
        jnp.zeros(M)
          .at[i0].add(w0 * v[:, 0])
          .at[i1].add(w1 * v[:, 0])
    )
    dEdt = w / eta * J
    return (E - dt * dEdt).astype(E.dtype)


@jax.jit
def vlasov_step(x, v, E, cells, eta, dt, box_length, w):
    """Forward Euler time stepping for Vlasov part."""
    E_at_particles = evaluate_field_at_particles(E, x, cells, eta)
    v = v.at[:, 0].add(dt * E_at_particles)
    x = jnp.mod(x + dt * v[:, 0], box_length)
    E = update_electric_field(E, x, v, cells, eta, w, dt)
    return x, v, E


def energy_conserving_step(
    x,
    v,
    E,
    cells,
    eta,
    dt,
    box_length,
    w,
    collision_strength,
    collision_evaluator=None,
):
    """Energy-conserving electrostatic particle/field step.

    This is scheme (3.7) specialized to q=1 and B_ext=0. When collisions
    are enabled, ``collision_evaluator(x_stage, v_stage, stage)`` must return
    ``(score, U)`` and is called at the three velocity stages n, **, and *.
    This function is not jitted because the evaluator may train a score model
    between stages; its array-valued building blocks are jitted.
    """
    half_dt = 0.5 * dt
    x_star = jnp.mod(x + half_dt * v[:, 0], box_length)
    stage_results = []

    def evaluate_collision(v_stage, stage):
        if collision_strength <= 0:
            return jnp.zeros_like(v_stage)
        if collision_evaluator is None:
            raise ValueError("collision_evaluator is required when collision_strength > 0")
        score, U = collision_evaluator(x_star, v_stage, stage)
        stage_results.append((score, U))
        return U

    # U(x*, v^n) and the first particle/field half-step.
    U_n = evaluate_collision(v, 0)
    E_n_at_star = evaluate_field_at_particles(E, x_star, cells, eta)
    v_starstar = v - half_dt * collision_strength * U_n
    v_starstar = v_starstar.at[:, 0].add(half_dt * E_n_at_star)
    E_star = update_electric_field(
        E, x_star, v_starstar, cells, eta, w, half_dt
    )

    # U(x*, v**) and the second particle half-step.
    U_starstar = evaluate_collision(v_starstar, 1)
    E_star_at_star = evaluate_field_at_particles(E_star, x_star, cells, eta)
    v_star = v - half_dt * collision_strength * U_starstar
    v_star = v_star.at[:, 0].add(half_dt * E_star_at_star)

    x_new = jnp.mod(x + dt * v_star[:, 0], box_length)
    E_new = update_electric_field(E, x_star, v_star, cells, eta, w, dt)

    # U(x*, v*) and the full-step provisional velocity.
    U_star = evaluate_collision(v_star, 2)
    E_half_at_star = evaluate_field_at_particles(
        0.5 * (E + E_new), x_star, cells, eta
    )
    v_dagger = v - dt * collision_strength * U_star
    v_dagger = v_dagger.at[:, 0].add(dt * E_half_at_star)

    # Per-particle Gamma correction in scheme (3.7).
    v_new, raw_gamma_squared, problematic_particles = gamma_correct_velocity(
        v, v_star, v_dagger
    )

    return (
        x_new,
        v_new,
        E_new,
        stage_results,
        raw_gamma_squared,
        problematic_particles,
        x_star,
        v_star,
    )


#------------------------------------------------------------------------------
# KDE in phase space and Landau collision
#------------------------------------------------------------------------------
"KDE score model for 1D space"
def _silverman_bandwidth(v, eps=1e-12):
        n, dv = v.shape
        sigma = jnp.std(v, axis=0, ddof=1) + eps
        return sigma * n ** (-1.0 / (dv + 4.0))  # (dv,)

@partial(jax.jit, static_argnames=['max_ppc'])
def _score_blob_local_impl(x, v, cells, eta, eps=1e-12, hv=None, max_ppc=4096):
    if x.ndim == 2:
        x = x[:, 0]
    if hv is None:
        hv = _silverman_bandwidth(v, eps)

    n, dv = v.shape
    M = cells.size
    L = eta * M
    inv_hv = 1.0 / hv
    inv_hv2 = inv_hv ** 2

    idx = jnp.floor(x / eta).astype(jnp.int32) % M
    order = jnp.argsort(idx)
    x_s = x[order]
    v_s = v[order]
    idx_s = idx[order]

    counts = jnp.bincount(idx_s, length=M).astype(jnp.int32)
    cell_ofs = jnp.cumsum(
        jnp.concatenate([jnp.array([0], dtype=jnp.int32), counts[:-1]]),
        dtype=jnp.int32,
    )

    Xc = jnp.zeros((M, max_ppc), x.dtype)
    Vc = jnp.zeros((M, max_ppc, dv), v.dtype)
    maskc = jnp.zeros((M, max_ppc), x.dtype)
    idx_map = -jnp.ones((M, max_ppc), jnp.int32)
    ar_ppc = jnp.arange(max_ppc, dtype=jnp.int32)

    def fill_cell(c, carry):
        Xc, Vc, maskc, idx_map = carry
        cnt = counts[c]
        base = cell_ofs[c]
        valid = ar_ppc < cnt
        gidx = base + ar_ppc
        gidx = jnp.where(valid, gidx, 0)
        Xc = Xc.at[c].set(jnp.where(valid, x_s[gidx], 0.0))
        Vc = Vc.at[c].set(jnp.where(valid[:, None], v_s[gidx], 0.0))
        maskc = maskc.at[c].set(valid.astype(x.dtype))
        idx_map = idx_map.at[c].set(jnp.where(valid, gidx, -1))
        return Xc, Vc, maskc, idx_map

    Xc, Vc, maskc, idx_map = lax.fori_loop(
        0, M, fill_cell, (Xc, Vc, maskc, idx_map)
    )

    Uc = Vc * inv_hv
    U2c = jnp.sum(Uc * Uc, axis=-1, keepdims=True)

    Zc = jnp.zeros((M, max_ppc, 1), v.dtype)
    Mc = jnp.zeros((M, max_ppc, dv), v.dtype)

    def body_cell(c, carry):
        Zc, Mc = carry
        Xi = Xc[c]                    # (max_ppc,)
        Vi = Vc[c]                    # (max_ppc,dv)
        Ui = Uc[c]
        Ui2 = U2c[c]                  # (max_ppc,1)
        mask_i = maskc[c][:, None]    # (max_ppc,1)

        c0 = (c - 1) % M
        c1 = c
        c2 = (c + 1) % M
        Xj = jnp.concatenate([Xc[c0], Xc[c1], Xc[c2]], axis=0)        # (3*max_ppc,)
        Vj = jnp.concatenate([Vc[c0], Vc[c1], Vc[c2]], axis=0)        # (3*max_ppc,dv)
        Uj = jnp.concatenate([Uc[c0], Uc[c1], Uc[c2]], axis=0)
        Uj2 = jnp.concatenate([U2c[c0], U2c[c1], U2c[c2]], axis=0)    # (3*max_ppc,1)
        mask_j = jnp.concatenate(
            [maskc[c0], maskc[c1], maskc[c2]], axis=0
        )[:, None]                                                     # (3*max_ppc,1)

        dx = Xi[:, None] - Xj[None, :]
        dx = (dx + 0.5 * L) % L - 0.5 * L
        psi = jnp.maximum(0.0, 1.0 - jnp.abs(dx) / eta)               # hat in x

        G = Ui @ Uj.T
        K = jnp.exp(G - 0.5 * Ui2 - 0.5 * Uj2.T)

        mask = mask_i * mask_j.T
        w = (psi * K + eps) * mask

        Z_local = jnp.sum(w, axis=1, keepdims=True) * mask_i
        M_local = (w @ Vj) * mask_i

        Zc = Zc.at[c].set(Z_local)
        Mc = Mc.at[c].set(M_local)
        return Zc, Mc

    Zc, Mc = lax.fori_loop(0, M, body_cell, (Zc, Mc))

    idx_flat = idx_map.reshape(-1)
    Z_flat = Zc.reshape(-1, 1)
    M_flat = Mc.reshape(-1, dv)
    valid = idx_flat >= 0
    idx_valid = jnp.where(valid, idx_flat, 0)
    Z_contrib = Z_flat * valid[:, None]
    M_contrib = M_flat * valid[:, None]

    Zs = jnp.zeros((n, 1), v.dtype)
    Ms = jnp.zeros((n, dv), v.dtype)
    Zs = Zs.at[idx_valid].add(Z_contrib)
    Ms = Ms.at[idx_valid].add(M_contrib)

    inv_order = jnp.empty_like(order)
    inv_order = inv_order.at[order].set(jnp.arange(n, dtype=order.dtype))

    Z = Zs[inv_order]
    M = Ms[inv_order]
    
    Z_safe = jnp.where(Z > 0, Z, eps)
    mu = M / Z_safe
    return (mu - v) * inv_hv2

# this is ~11 times faster than score_blob_blocked with n=1e5 and M=50
def score_blob(x, v, cells, eta, eps=1e-12, hv=None):
    if hv is None:
        hv = _silverman_bandwidth(v, eps)

    if x.ndim == 2:
        x1d = x[:, 0]
    else:
        x1d = x
    M = cells.size
    idx = jnp.floor(x1d / eta).astype(jnp.int32) % M
    counts = jnp.bincount(idx, length=M)
    max_count = int(jax.device_get(jnp.max(counts)))
    m = max(1, max_count)
    max_ppc = ((m + 99) // 100) * 100  # next multiple of 100 >= m

    return _score_blob_local_impl(x, v, cells, eta, eps, hv, max_ppc)


def scaled_score_blob(x, v, cells, eta, eta_scale=4, hv_scale=4, output_scale=1.3, **kwargs):
    """Empirically tuned scaled KDE score."""
    hv = _silverman_bandwidth(v) * hv_scale
    s_blob = score_blob(x, v, cells, eta * eta_scale, hv=hv, **kwargs) * output_scale
    return s_blob

#------------------------------------------------------------------------------
# Landau collision operator
#------------------------------------------------------------------------------
def compute_window_params(N, eta, box_length, bucket_size=100, safety_factor=1.2):
    """
    Calculates the required window size based on density and buckets it.
    
    Args:
        N: Number of particles
        eta: Interaction radius
        box_length: Domain size
        bucket_size: Round up to multiples of this (prevents frequent recompilation)
        safety_factor: Multiplier to account for particle clustering (non-uniformity)
    """
    # Average density of particles
    density = N / box_length
    
    # Expected neighbors in range [-eta, +eta] (so 2 * eta)
    # We only need 'window_size' which is the radius (neighbors on ONE side)
    avg_neighbors_radius = density * eta
    
    # Add safety margin for clustering
    target_size = avg_neighbors_radius * safety_factor
    
    # Round up to nearest bucket_size (e.g., 100)
    # Formula: ceil(x / 100) * 100
    window_size = math.ceil(target_size / bucket_size) * bucket_size
    
    # Clamp: Window cannot be larger than N//2 (wrapping around the world)
    window_size = min(int(window_size), N // 2)
    
    # Ensure at least 1 to avoid shape errors
    return max(window_size, 1)

@jax.jit
def A_apply(dv, ds, gamma, eps=1e-14):
    # v2 = jnp.sum(dv * dv, axis=-1, keepdims=True) + eps
    # vg = v2 ** (gamma / 2)
    # dvds = jnp.sum(dv * ds, axis=-1, keepdims=True)
    # return vg * (v2 * ds - dvds * dv)
    r2 = jnp.sum(dv * dv, axis=-1, keepdims=True)
    r2_safe = r2 + eps
    vg = r2_safe ** (gamma / 2)
    dvds = jnp.sum(dv * ds, axis=-1, keepdims=True)

    return vg * (r2 * ds - dvds * dv)

@partial(jax.jit, static_argnames=['window_size'])
def collision_rolling(x, v, s, eta, gamma, box_length, w, window_size):
    """
    O(N) Memory implementation using Vectorized Rolling.
    """
    if x.ndim == 2:
        x = x[:, 0]
    
    N, d = v.shape

    # 1. Sort (Essential for windowing)
    order = jnp.argsort(x)
    x_sorted = x[order]
    v_sorted = v[order]
    s_sorted = s[order]

    # 2. Define the loop body
    # Instead of a massive matrix, we compute one "offset layer" at a time.
    def body_fn(i, val):
        Q_accum = val
        
        # Map loop index i (0 to 2*window) to offset (-window to +window)
        offset = i - window_size
        
        # Fetch Neighbor Data using Roll
        # jnp.roll is O(N) and very fast on GPU
        x_neighbor = jnp.roll(x_sorted, shift=offset, axis=0)
        v_neighbor = jnp.roll(v_sorted, shift=offset, axis=0)
        s_neighbor = jnp.roll(s_sorted, shift=offset, axis=0)

        # Periodic Distance
        dx = x_sorted - x_neighbor
        dx = (dx + box_length / 2) % box_length - box_length / 2
        dist = jnp.abs(dx)
        
        # Physics & Masking
        # We multiply by (offset != 0) to avoid self-interaction
        mask = (dist <= eta) & (dist > 0.0) & (offset != 0)
        
        psi = jnp.maximum(0.0, 1.0 - dist / eta) / eta
        psi = psi * mask

        # Since we are 1D in the loop, we need to expand dims for broadcasting
        # psi is (N,), we need (N, 1) to multiply against (N, d)
        psi = psi[:, None]

        dv = v_sorted - v_neighbor
        ds = s_sorted - s_neighbor
        
        interaction = A_apply(dv, ds, gamma)
        
        # Accumulate result
        return Q_accum + interaction * psi

    # 3. Run the Loop
    # We iterate 2*window_size + 1 times.
    # Unlike your original code, the work inside here is perfectly balanced (size N).
    Q_init = jnp.zeros_like(v)
    Q_sorted = lax.fori_loop(0, 2 * window_size + 1, body_fn, Q_init)

    # 4. Unsort
    rev = jnp.empty_like(order).at[order].set(jnp.arange(N))
    return w * Q_sorted[rev]

# on rtx6k, with fp64, n=1e6, M=100, dv=2, takes ~16s and 448Mb memory
def collision(x, v, s, eta, gamma, box_length, w):
    """
    Driver function that calculates window size and dispatches to JIT kernel.
    """
    n = v.shape[0]
    
    # 1. Pre-compute the window size (Python int)
    # This is fast and happens outside JAX.
    win_size = compute_window_params(n, eta, box_length)
    
    # 2. Call the JIT-compiled kernel
    return collision_rolling(x, v, s, eta, gamma, box_length, w, window_size=win_size)

#------------------------------------------------------------------------------
# SBTM helpers
#------------------------------------------------------------------------------
def mse_loss(model, batch):
    x, v, s = batch
    pred = model(x, v)
    return loss.mse(pred, s)

def ism_loss(model, batch, key):
    x, v = batch
    return loss.implicit_score_matching_loss(model, x, v, key=key)

@nnx.jit
def supervised_step(model, optimizer, batch):
    loss_val, grads = nnx.value_and_grad(mse_loss)(model, batch)
    if hasattr(optimizer, 'model'):
        optimizer.update(grads)
    else:
        optimizer.update(model, grads)
    return loss_val

@nnx.jit
def score_step(model, optimizer, batch, key):
    loss_val, grads = nnx.value_and_grad(ism_loss)(model, batch, key)
    optimizer.update(grads)
    return loss_val

def train_initial_model(model, x, v, score, batch_size, num_epochs, abs_tol, lr, verbose=False, print_every=10):
    optimizer = nnx.Optimizer(model, optax.adamw(lr), wrt=nnx.Param)
    n = x.shape[0]
    full_loss_hist = []
    for epoch in range(num_epochs):
        full_loss = mse_loss(model, (x, v, score))
        full_loss_hist.append(full_loss)
        if verbose and epoch % print_every == 0:
            print(f"Epoch {epoch}: loss = {full_loss:.5f}")
        if full_loss < abs_tol:
            if verbose:
                print(f"Stopping at epoch {epoch} with loss {full_loss:.5f} < {abs_tol}")
            break
        key = jr.PRNGKey(epoch)
        perm = jr.permutation(key, n)
        x_sh, v_sh, s_sh = x[perm], v[perm], score[perm]
        for i in range(0, n, batch_size):
            batch = (
                x_sh[i:i+batch_size],
                v_sh[i:i+batch_size],
                s_sh[i:i+batch_size],
            )
            supervised_step(model, optimizer, batch)
    return full_loss_hist

def train_score_model(model, optimizer, x, v, key, batch_size, num_batch_steps):
    n = x.shape[0]
    losses = []
    batch_count = 0
    while batch_count < num_batch_steps:
        key, subkey = jr.split(key)
        perm = jr.permutation(subkey, n)
        x = x[perm]
        v = v[perm]
        start = 0
        while start < n and batch_count < num_batch_steps:
            end = min(start + batch_size, n)
            batch = (x[start:end], v[start:end])
            key, subkey = jr.split(key)
            loss_val = score_step(model, optimizer, batch, subkey)
            losses.append(loss_val)
            start = end
            batch_count += 1
    return losses

#------------------------------------------------------------------------------
# Density visualization and analysis for Weibel experiment
#------------------------------------------------------------------------------
def compute_l2_distance_to_gaussian(v, dv, theta=1.0, bounds_v=None, bins=200):
    """
    Computes L2 distance between velocity distribution and Gaussian with variance theta.

    Args:
        v: Velocity samples (N, dv)
        dv: Number of velocity dimensions
        theta: Variance of the target Gaussian (default 1.0 for standard Gaussian)
               For Weibel: theta = (beta + c**2) / 2 if dv == 2, or beta/2 + c**2/3 if dv == 3
        bounds_v: Bounds for each velocity dimension, defaults to [(-3, 3)] * dv
        bins: Number of bins per dimension

    Returns:
        L2 distance (scalar)
    """
    if bounds_v is None:
        bounds_v = [(-3.0, 3.0)] * dv

    # Get density on regular grid
    density = density_on_regular_grid(
        v, bounds_v=bounds_v, bins_per_side=bins, smooth_sigma_bins=0.0
    )

    # Create coordinate grids
    bounds_v = jnp.asarray(bounds_v)
    lo, hi = bounds_v[:, 0], bounds_v[:, 1]
    dx = (hi - lo) / bins

    # Create meshgrid for all dimensions
    axes = [lo[d] + (jnp.arange(bins) + 0.5) * dx[d] for d in range(dv)]
    mesh = jnp.meshgrid(*axes, indexing="ij")

    # Compute Gaussian with variance theta: (2πθ)^(-dv/2) * exp(-||v||²/(2θ))
    v_squared = sum(m**2 for m in mesh)
    gaussian = jnp.exp(-0.5 * v_squared / theta) / ((2.0 * jnp.pi * theta) ** (dv / 2.0))

    # Compute grid cell volume
    dx_volume = jnp.prod(dx)

    # L2 distance
    l2_dist = jnp.sqrt(jnp.sum((density - gaussian) ** 2) * dx_volume)

    return float(l2_dist)


def plot_v1v2_marginal_snapshots(v_traj, t_traj, bounds_v=None, bins_per_side=200, smooth_sigma_bins=0.0, title=None):
    """
    Creates multi-panel figure showing v1-v2 marginal distribution at multiple timesteps.

    Args:
        v_traj: List of velocity snapshots
        t_traj: List of times
        bounds_v: Velocity bounds, defaults to [(-0.7, 0.7), (-0.7, 0.7)]
        bins_per_side: Number of bins per side
        smooth_sigma_bins: Smoothing sigma in bins
        title: Optional title for the figure
    Returns:
        Figure object
    """
    if bounds_v is None:
        bounds_v = [(-0.7, 0.7), (-0.7, 0.7)]

    num_snaps = len(v_traj)
    cols = min(3, num_snaps)
    rows = int(np.ceil(num_snaps / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=True, sharey=True)
    axs = np.array(axs).reshape(-1)

    # Setup colormap
    cmap = plt.cm.jet.copy()
    cmap.set_bad(color="black")

    # First pass: compute all densities and find global vmin/vmax
    density_list = []
    for v_snap in v_traj:
        density_vals = density_on_regular_grid(
            v_snap[:, :2], bounds_v=bounds_v, bins_per_side=bins_per_side, smooth_sigma_bins=smooth_sigma_bins
        )
        density_vals = np.where(np.asarray(density_vals) > 0, np.asarray(density_vals), np.nan)
        density_list.append(density_vals)

    # Compute global vmin/vmax using 1st and 99th percentiles
    all_densities = np.concatenate([d.flatten() for d in density_list])
    vmin = np.nanpercentile(all_densities, 1)
    vmax = np.nanpercentile(all_densities, 99)

    # Second pass: plot with consistent colormap
    last_img = None
    for i, (density_vals, t_snap) in enumerate(zip(density_list, t_traj)):
        ax = axs[i]

        # Plot with log scale
        img = ax.imshow(
            density_vals.T,
            origin="lower",
            extent=[bounds_v[0][0], bounds_v[0][1], bounds_v[1][0], bounds_v[1][1]],
            aspect="auto",
            cmap=cmap,
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
        last_img = img

        ax.set_title(f"t = {t_snap:.1f}")
        ax.set_xlabel("v1")
        ax.set_ylabel("v2")

    # Hide unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    # Add colorbar
    if last_img is not None:
        cbar = fig.colorbar(last_img, ax=axs.tolist(), orientation="vertical", fraction=0.02, pad=0.02)
        cbar.set_label("Density")

    plt.suptitle(title if title else "V1-V2 Marginal Distribution")

    return fig


def plot_v1v2_spatial_slice_snapshots(x_traj, v_traj, t_traj, bounds_v=None, bounds_x=None, bins_per_side=None, smooth_sigma_bins=0.0, title=None):
    """
    Creates multi-panel figure showing v1-v2 distribution in a spatial slice.

    Args:
        x_traj: List of position snapshots
        v_traj: List of velocity snapshots
        t_traj: List of times
        bounds_v: Velocity bounds, defaults to [(-0.7, 0.7), (-0.7, 0.7)]
        bounds_x: Spatial bounds, defaults to auto-computed around median
        bins_per_side: Bins per dimension, defaults to (1, 200, 200)
        smooth_sigma_bins: Smoothing sigma in bins
        title: Optional title for the figure
    Returns:
        Figure object
    """
    if bounds_v is None:
        bounds_v = [(-0.7, 0.7), (-0.7, 0.7)]
    if bins_per_side is None:
        bins_per_side = (1, 200, 200)

    # Auto-compute bounds_x if not provided (centered around median with width 0.1)
    if bounds_x is None:
        x_first = np.asarray(x_traj[0])
        x_median = float(np.median(x_first))
        bounds_x = (x_median - 0.05, x_median + 0.05)

    num_snaps = len(v_traj)
    cols = min(3, num_snaps)
    rows = int(np.ceil(num_snaps / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=True, sharey=True)
    axs = np.array(axs).reshape(-1)

    # Setup colormap
    cmap = plt.cm.jet.copy()
    cmap.set_bad(color="black")

    # First pass: compute all densities and find global vmin/vmax
    density_2d_list = []
    for x_snap, v_snap in zip(x_traj, v_traj):
        density_vals = density_on_regular_grid(
            v_snap[:, :2],
            bounds_v=bounds_v,
            bins_per_side=bins_per_side,
            x=x_snap,
            bounds_x=bounds_x,
            smooth_sigma_bins=smooth_sigma_bins,
        )
        # Extract 2D slice (single x bin, 200x200 in v1-v2)
        density_2d = density_vals[0, :, :]
        density_2d = np.where(np.asarray(density_2d) > 0, np.asarray(density_2d), np.nan)
        density_2d_list.append(density_2d)

    # Compute global vmin/vmax using 1st and 99th percentiles
    all_densities = np.concatenate([d.flatten() for d in density_2d_list])
    vmin = np.nanpercentile(all_densities, 1)
    vmax = np.nanpercentile(all_densities, 99)
    
    # Second pass: plot with consistent colormap
    last_img = None
    for i, (density_2d, t_snap) in enumerate(zip(density_2d_list, t_traj)):
        ax = axs[i]

        # Plot with log scale
        img = ax.imshow(
            density_2d.T,
            origin="lower",
            extent=[bounds_v[0][0], bounds_v[0][1], bounds_v[1][0], bounds_v[1][1]],
            aspect="auto",
            cmap=cmap,
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
        last_img = img

        ax.set_title(f"t = {t_snap:.1f}")
        ax.set_xlabel("v1")
        ax.set_ylabel("v2")

    # Hide unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    # Add colorbar
    if last_img is not None:
        cbar = fig.colorbar(last_img, ax=axs.tolist(), orientation="vertical", fraction=0.02, pad=0.02)
        cbar.set_label("Density")

    default_title = f"V1-V2 Spatial Slice (x ≈ {(bounds_x[0] + bounds_x[1]) / 2:.3f})"
    plt.suptitle(title if title else default_title)

    return fig


def plot_v2_at_v1_zero_evolution(v_traj, t_traj, bounds_v=None, bins_per_side=None, theta=1.0, logy=False, title=None):
    """
    Creates line plot showing v2 distribution at v1≈0 for all timesteps.

    Args:
        v_traj: List of velocity snapshots
        t_traj: List of times
        bounds_v: Velocity bounds, defaults to [(-0.01, 0.01), (-3, 3)]
        bins_per_side: Bins per dimension, defaults to (1, 200)
        theta: Variance of the steady-state Gaussian (default 1.0)
               For Weibel: theta = (beta + c**2) / 2 if dv == 2, or beta/2 + c**2/3 if dv == 3
        logy: If True, use logarithmic y-axis (default False)
        title: Optional title for the figure

    Returns:
        Figure object
    """
    if bounds_v is None:
        bounds_v = [(-0.01, 0.01), (-3.0, 3.0)]
    if bins_per_side is None:
        bins_per_side = (1, 200)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot each timestep
    for v_snap, t_snap in zip(v_traj, t_traj):
        # Get density on regular grid (use first 2 velocity components)
        density_vals = density_on_regular_grid(
            v_snap[:, :2], bounds_v=bounds_v, bins_per_side=bins_per_side, smooth_sigma_bins=0.0
        )

        # Extract 1D (single v1 bin, 200 v2 bins)
        density_1d = density_vals[0, :]

        # Create v2 grid
        bounds_v_arr = np.asarray(bounds_v)
        v2_grid = np.linspace(bounds_v_arr[1, 0], bounds_v_arr[1, 1], bins_per_side[1])

        ax.plot(v2_grid, np.asarray(density_1d), label=f"t={t_snap:.1f}")

    # Add steady state (2D Gaussian evaluated at v1=0)
    # f(v1=0, v2) = exp(-v2^2/(2*theta)) / (2*pi*theta)
    v2_grid = np.linspace(bounds_v[1][0], bounds_v[1][1], bins_per_side[1])
    gaussian_steady = np.exp(-0.5 * v2_grid**2 / theta) / (2.0 * np.pi * theta)
    ax.plot(v2_grid, gaussian_steady, 'k--', label=fr'$N(0,{theta:.4f})$')

    ax.set_xlabel("v2")
    ax.set_ylabel("Density")
    if logy:
        ax.set_yscale('log')
    ax.set_title(title if title else "V2 Distribution at v1 ≈ 0")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_v2_marginal_evolution(v_traj, t_traj, bounds_v=None, bins_per_side=200, theta=1.0, title=None, logy=False):
    """
    Creates line plot showing v2 marginal distribution for all timesteps.

    Args:
        v_traj: List of velocity snapshots
        t_traj: List of times
        bounds_v: Velocity bounds, defaults to [(-3, 3)]
        bins_per_side: Number of bins
        theta: Variance of the steady-state Gaussian (default 1.0)
               For Weibel: theta = (beta + c**2) / 2 if dv == 2, or beta/2 + c**2/3 if dv == 3
        title: Optional title for the figure
        logy: If True, use logarithmic y-axis (default False)

    Returns:
        Figure object
    """
    if bounds_v is None:
        bounds_v = [(-3.0, 3.0)]

    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot each timestep
    for v_snap, t_snap in zip(v_traj, t_traj):
        # Get density on regular grid (use v2 only, which is v[:, 1:2])
        density_vals = density_on_regular_grid(
            v_snap[:, 1:2], bounds_v=bounds_v, bins_per_side=bins_per_side, smooth_sigma_bins=0.0
        )

        # Create v2 grid
        bounds_v_arr = np.asarray(bounds_v)
        v2_grid = np.linspace(bounds_v_arr[0, 0], bounds_v_arr[0, 1], bins_per_side)

        ax.plot(v2_grid, np.asarray(density_vals), label=f"t={t_snap:.1f}")

    # Add steady state (Gaussian with variance theta)
    v2_grid = np.linspace(bounds_v[0][0], bounds_v[0][1], bins_per_side)
    gaussian_steady = np.exp(-0.5 * v2_grid**2 / theta) / np.sqrt(2.0 * np.pi * theta)
    ax.plot(v2_grid, gaussian_steady, 'k--', label=fr'$N(0,{theta:.4f})$')

    ax.set_xlabel("v2")
    ax.set_ylabel("Density")
    if logy:
        ax.set_yscale('log')
    ax.set_title(title if title else "V2 Marginal Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig
