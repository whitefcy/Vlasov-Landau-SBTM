"""Check Example 5.2's physical scaling, moment diagnostics, and aggregation."""

import contextlib
from copy import deepcopy
import io
import unittest

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from experiments.homo_anisotropic import (
    benchmark_metrics, covariance_exact, initial_score, initial_variances,
    parse_args, sample_anisotropic,
)
from src.homogeneous import maxwell_collision
from src.homogeneous_plots import convergence_data, trajectory, validate_runs

jax.config.update("jax_enable_x64", True)


class AnisotropicTests(unittest.TestCase):
    def test_reference_covariance_normalization_and_limits(self):
        for d in [3, 10]:
            sigma = np.asarray(initial_variances(d))
            np.testing.assert_allclose(covariance_exact(0, d), np.diag(sigma), atol=1e-14)
            np.testing.assert_allclose(covariance_exact(1000, d), np.eye(d), atol=1e-14)
            self.assertAlmostEqual(float(jnp.trace(covariance_exact(2.3, d))), d)
            derivative = jax.jacfwd(lambda t: covariance_exact(t, d))(0.)
            np.testing.assert_allclose(derivative, -d / 6 * np.diag(sigma - 1), atol=1e-14)
        self.assertAlmostEqual(float(covariance_exact(4)[0, 0]), 1 + 0.8*np.exp(-2))
        np.testing.assert_allclose(covariance_exact(1, B=1/12), covariance_exact(2, B=1/24))

    def test_initial_variances_not_standard_deviations(self):
        for d in [3, 10]:
            v = np.asarray(sample_anisotropic(jr.key(d), 100_000, d))
            np.testing.assert_allclose(v.var(axis=0), initial_variances(d), atol=0.022)
            np.testing.assert_allclose(initial_score(jnp.asarray(v[:8])), -v[:8] / np.asarray(initial_variances(d)), atol=1e-14)

    def test_gaussian_cubature_collision_moment_rate_and_entropy_sign(self):
        # Equal-weight cubature exact through degree 5 in each coordinate.
        # It tests the collision law independently of the closed moment formula.
        nodes = np.array([-np.sqrt(3), 0, 0, 0, 0, np.sqrt(3)])
        grid = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 3)
        v = jnp.asarray(grid * np.sqrt([1.8, 0.2, 1.]))
        s = initial_score(v)
        for B in [1/24, 1/12]:
            collision = maxwell_collision(v, s)
            flow = -B * collision
            covariance_rate = (v.T @ flow + flow.T @ v) / len(v)
            expected_rate = -4 * 3 * B * np.diag([0.8, -0.8, 0])
            np.testing.assert_allclose(covariance_rate, expected_rate, atol=1e-12)
            metrics = benchmark_metrics(v, s, collision, 0., v.mean(axis=0), 1.5, B)
            analytic_entropy_rate = -0.5 * np.trace(np.diag([1/1.8, 1/0.2, 1]) @ expected_rate)
            self.assertAlmostEqual(metrics["estimated_entropy_rate"], analytic_entropy_rate, places=12)
            self.assertLess(metrics["estimated_entropy_rate"], 0)
            self.assertAlmostEqual(metrics["estimated_entropy_dissipation"], -analytic_entropy_rate, places=12)
            self.assertLess(metrics["second_moment_error_fro"], 1e-12)
            self.assertNotIn("score_relative_mse", metrics)
            self.assertNotIn("density_l2", metrics)

    def test_no_exact_score_mode_for_unknown_solution(self):
        args = parse_args([])
        self.assertEqual(args.t0, 0)
        self.assertEqual(args.final_time, 4)
        self.assertEqual(args.sbtm_hidden_dims, [100])
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in [["--score_method", "exact"], ["--t0", "1"]]:
                with self.assertRaises(SystemExit):
                    parse_args(argv)


class FigureAggregationTests(unittest.TestCase):
    def make_run(self, n, seed, final_error):
        config = dict(example="anisotropic", dv=3, B=1/24, dt=0.01, t0=0., final_time=4.,
                      time_integrator="forward_euler", fp32=False, initial_variances=[1.8, 0.2, 1.],
                      score_method="sbtm", n=n, seed=seed)
        initial = dict(time=0., second_moment_error_fro=10., second_moment_11=1.8)
        final = dict(time=4., second_moment_error_fro=final_error, second_moment_11=1.1)
        return dict(config=config, records=[initial, final], summary=dict(initial=initial, final=final), path="test")

    def test_final_errors_and_seed_means(self):
        runs = [self.make_run(100, 1, 0.1), self.make_run(100, 2, 0.3), self.make_run(200, 1, 0.05)]
        validate_runs(runs)
        rows = convergence_data(runs, "second_moment_error_fro")
        self.assertEqual([r["n"] for r in rows], [100, 200])
        self.assertAlmostEqual(rows[0]["mean"], 0.2)
        self.assertAlmostEqual(rows[0]["std"], 0.1)
        times, values = trajectory(runs[:2], "second_moment_11")
        np.testing.assert_array_equal(times, [0, 4])
        np.testing.assert_allclose(values, [1.8, 1.1])

    def test_incompatible_or_duplicate_runs_rejected(self):
        run = self.make_run(100, 1, 0.1)
        with self.assertRaises(ValueError):
            validate_runs([run, deepcopy(run)])
        other = self.make_run(100, 2, 0.1)
        other["config"]["B"] = 1.
        with self.assertRaises(ValueError):
            validate_runs([run, other])
        other = self.make_run(100, 2, 0.1)
        other["records"][-1]["time"] = 3.9
        with self.assertRaises(ValueError):
            validate_runs([other])


if __name__ == "__main__":
    unittest.main()
