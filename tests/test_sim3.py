import numpy as np
import pytest

from davio.mapper import sim3

X = np.array([.3, -.2, .1, .05, -.4, .2, .3])


def test_exp_and_log_are_inverse_and_the_last_component_is_the_log_of_the_scale():
    T = sim3.exp(X)
    np.testing.assert_allclose(sim3.log(T), X, atol=1e-10)
    np.testing.assert_allclose(sim3.exp(np.zeros(7)), np.eye(4))
    assert sim3.scale(T) == pytest.approx(np.exp(.3))
    np.testing.assert_allclose(sim3.pose(T)[:3, :3] @ sim3.pose(T)[:3, :3].T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(sim3.pose(T)[:3, 3], T[:3, 3])
    np.testing.assert_allclose(sim3.inverse(T) @ T, np.eye(4), atol=1e-12)


def test_the_adjoint_carries_an_increment_from_the_right_to_the_left():
    T, step = sim3.exp(X), np.array([.01, .02, -.01, .03, -.02, .01, .02])
    np.testing.assert_allclose(T @ sim3.exp(step), sim3.exp(sim3.adjoint(T) @ step) @ T, atol=1e-12)


def test_log_jacobians_follow_an_increment_on_either_side():
    left, right = sim3.log_jacobians(X)
    step = 1e-6 * np.array([1., -2., .5, 1.5, -1., 2., 1.])
    np.testing.assert_allclose(sim3.log(sim3.exp(step) @ sim3.exp(X)) - X, left @ step, atol=1e-10)
    np.testing.assert_allclose(sim3.log(sim3.exp(X) @ sim3.exp(step)) - X, right @ step, atol=1e-10)


@pytest.mark.parametrize('matrix', [np.diag([1., 2., 3., 1.]),       # affine
                                    np.diag([-1., 1., 1., 1.]),      # a reflection
                                    np.full((4, 4), np.nan), np.eye(3)])
def test_what_is_not_a_similarity_is_refused(matrix):
    with pytest.raises(ValueError):
        sim3.validate(matrix)
