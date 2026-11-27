"""Numerical and exact-moment checks for the projected examples."""

import unittest

import numpy as np

from projected_examples import (examples, nested_gradient, population_loss,
                                population_numbers, project_ball,
                                projected_iteration)


class ProjectedExamplesTest(unittest.TestCase):
    def test_exact_distribution_bounds(self):
        cases = examples()
        simple = population_numbers(cases["simple"])
        self.assertAlmostEqual(simple["lambda_min"], 1)
        self.assertAlmostEqual(simple["R_min"], np.sqrt(2))
        self.assertAlmostEqual(simple["R_max"], np.sqrt(2))
        nonteacher = population_numbers(cases["general_l2"])
        self.assertAlmostEqual(nonteacher["Sigma_X"][0][0], 2.5)
        self.assertAlmostEqual(nonteacher["b_E_XY"][0], 1.25)
        self.assertAlmostEqual(nonteacher["P_min"], 0.5)
        self.assertAlmostEqual(nonteacher["R_min"], 1.0)
        self.assertAlmostEqual(population_loss(np.array([[1.], [.5]]), 0, nonteacher), .125)
        higher = population_numbers(cases["general_l3"])
        self.assertAlmostEqual(higher["lambda_min"], 1 / 3)
        self.assertAlmostEqual(higher["Sigma_X"][0][1], 1 / 3)
        self.assertLess(higher["R_min"], higher["R_max"])

    def test_nested_estimator_targets_smoothed_gradient(self):
        case = examples()["general_l3"]
        numbers = population_numbers(case)
        weights = case.initial.copy()
        eta = .25
        gradient = nested_gradient(weights, eta, case.distribution, 80000, 1,
                                   np.random.default_rng(0))
        numerical = np.empty_like(weights)
        for layer, coordinate in np.ndindex(weights.shape):
            plus, minus = weights.copy(), weights.copy()
            plus[layer, coordinate] += 1e-5
            minus[layer, coordinate] -= 1e-5
            numerical[layer, coordinate] = (
                population_loss(plus, eta, numbers) - population_loss(minus, eta, numbers)
            ) / 2e-5
        np.testing.assert_allclose(gradient, numerical, atol=.035, rtol=.04)

    def test_projection_and_iterates(self):
        np.testing.assert_allclose(np.linalg.norm(project_ball(np.array([3., 4.]), 2.)), 2.)
        case = examples()["simple"]
        numbers = population_numbers(case)
        radius = numbers["R_max"] + .1
        path, active = projected_iteration(case, numbers, steps=30, outer_batch=2,
                                            inner_batch=2, eta0=.7, alpha0=1.,
                                            radius=radius, seed=4)
        self.assertGreater(active, 0)
        self.assertTrue((np.linalg.norm(path.reshape(len(path), -1), axis=1) <= radius + 1e-12).all())


if __name__ == "__main__":
    unittest.main()
