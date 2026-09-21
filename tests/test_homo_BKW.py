"""Numerical validation of the BKW benchmark and its Maxwell reduction.

Run: python -m unittest discover -s tests -p test_homo_BKW.py
"""

import contextlib
import io
import unittest

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma

from experiments.homo_BKW import (
    benchmark_metrics, bkw_density, bkw_flow, bkw_fourth_moment,
    bkw_parameters, bkw_score, density_l2_error, gaussian_kde,
    maxwell_collision, parse_args, sample_bkw, scott_bandwidth,
)

jax.config.update("jax_enable_x64", True)


class BKWTests(unittest.TestCase):
    def test_density_moments_by_radial_quadrature(self):
        for d, t in [(2, 9.0), (3, 5.5), (3, 9.5)]:
            K, P, Q = map(float, bkw_parameters(t, d))
            area = 2 * np.pi**(d / 2) / gamma(d / 2)
            def moment(order):
                return quad(lambda r: area * r**(d - 1 + order) *
                            np.exp(-r*r/(2*K)) * (P + Q*r*r) / (2*np.pi*K)**(d/2),
                            0, np.inf, epsabs=1e-11)[0]
            self.assertAlmostEqual(moment(0), 1, places=10)
            self.assertAlmostEqual(moment(2), d, places=10)
            self.assertAlmostEqual(moment(4), float(bkw_fourth_moment(t, d)), places=9)

    def test_score_and_continuity_equation(self):
        v = jnp.array([0.3, -0.6, 1.1])
        for t in [5.5, 7.2, 9.5]:
            for B in [1/24, 0.1]:
                expected = jax.grad(lambda u: jnp.log(bkw_density(u, t, B)))(v)
                np.testing.assert_allclose(bkw_score(v, t, B), expected, atol=1e-12)
                time_derivative = jax.grad(lambda time: bkw_density(v, time, B))(t)
                flux_divergence = jnp.trace(jax.jacfwd(
                    lambda u: bkw_density(u, t, B) * bkw_flow(u, t, B))(v))
                self.assertAlmostEqual(float(time_derivative + flux_divergence), 0, places=12)

    def test_exact_sampler_moments_and_reproducibility(self):
        v = np.asarray(sample_bkw(jr.key(15), 100_000))
        np.testing.assert_allclose(v.mean(axis=0), 0, atol=0.012)
        np.testing.assert_allclose(v.T @ v / len(v), np.eye(3), atol=0.014)
        self.assertLess(abs(np.mean(np.sum(v*v, axis=1)**2) - float(bkw_fourth_moment(5.5))), 0.15)
        np.testing.assert_array_equal(sample_bkw(jr.key(2), 17), sample_bkw(jr.key(2), 17))
        with self.assertRaises(ValueError):
            sample_bkw(jr.key(2), 10, t=0)

    def test_maxwell_reduction_against_pairwise_and_invariants(self):
        for d in [2, 3, 5]:
            v = jr.normal(jr.key(d), (23, d)) + 2.3
            s = jr.normal(jr.key(d + 10), (23, d)) - 1.7
            z = v[:, None] - v[None, :]
            ds = s[:, None] - s[None, :]
            direct = jnp.mean(jnp.sum(z*z, axis=-1, keepdims=True) * ds -
                              jnp.sum(z*ds, axis=-1, keepdims=True) * z, axis=1)
            actual = maxwell_collision(v, s)
            np.testing.assert_allclose(actual, direct, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(actual.mean(axis=0), 0, atol=1e-12)
            self.assertLess(abs(float(jnp.mean(jnp.sum(v * actual, axis=1)))), 1e-12)
            self.assertGreaterEqual(float(jnp.mean(jnp.sum(s * actual, axis=1))), 0)
            np.testing.assert_allclose(maxwell_collision(v, -v), 0, atol=1e-12)

    def test_kde_l2_against_independent_volume_quadrature(self):
        v = sample_bkw(jr.key(31), 12)
        h = np.asarray(scott_bandwidth(v))
        # Integrate (KDE-u*)^2 directly over a box with negligible Gaussian tails.
        nodes, weights = np.polynomial.legendre.leggauss(96)
        nodes, weights = 9 * nodes, 9 * weights
        grid = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 3)
        volume_weights = np.einsum("i,j,k->ijk", weights, weights, weights).ravel()
        density = np.zeros(len(grid))
        for center in np.asarray(v):
            density += np.exp(-0.5 * np.sum(((grid-center)/h)**2, axis=1)) / (len(v) * (2*np.pi)**1.5 * h.prod())
        exact = np.asarray(bkw_density(jnp.asarray(grid), 6.0))
        integrated = np.sqrt(np.sum(volume_weights * (density - exact)**2))
        computed, relative, _ = density_l2_error(v, 6.0, block_size=5)
        self.assertAlmostEqual(computed, integrated, places=7)
        self.assertGreater(relative, 0)
        # Non-divisible query blocks and scores agree with differentiation.
        points = v[:7] + 0.2
        den, score = gaussian_kde(points, v, jnp.asarray(h), block_size=5)
        def scalar_log_kde(point):
            exps = -0.5 * jnp.sum(((point - v) / h)**2, axis=1)
            return jax.scipy.special.logsumexp(exps) - jnp.log(len(v) * (2*jnp.pi)**1.5 * np.prod(h))
        np.testing.assert_allclose(den, jnp.exp(jax.vmap(scalar_log_kde)(points)), rtol=1e-12)
        np.testing.assert_allclose(score, jax.vmap(jax.grad(scalar_log_kde))(points), atol=1e-12)

    def test_time_validation_and_initial_conservation_diagnostics(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in [["--t0", "0"], ["--dt", "nan"], ["--final_time", "4"], ["--B", "0"]]:
                with self.assertRaises(SystemExit):
                    parse_args(argv)
        args = parse_args([])
        self.assertEqual(args.t0, 5.5)
        self.assertEqual(args.final_time, 9.5)
        v = sample_bkw(jr.key(42), 100)
        s = bkw_score(v, 5.5)
        metrics = benchmark_metrics(v, s, maxwell_collision(v, s), 5.5,
                                    v.mean(axis=0), 0.5*jnp.mean(jnp.sum(v*v, axis=1)), 1/24)
        self.assertLess(metrics["relative_energy_drift"], 1e-14)
        self.assertEqual(metrics["momentum_drift"], 0)
        self.assertEqual(metrics["score_relative_mse"], 0)
        self.assertGreater(metrics["flow_mse"], 0)  # finite-particle error remains with exact scores


if __name__ == "__main__":
    unittest.main()
