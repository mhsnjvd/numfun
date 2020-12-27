import numpy as np
from scipy import linalg

from numfun import chebyshev


def test_coefficients_to_values() -> None:
    tol = 100 * np.spacing(1)

    # Test that a single value is converted correctly
    c = np.array([np.sqrt(2)])
    v = chebyshev.chebyshev_coefficients_to_values(c)
    assert c == v

    # Some simple data
    c = np.r_[1:6]
    # Exact coefficients
    v_true = np.array([3.0, -4.0+np.sqrt(2.0), 3.0, -4.0-np.sqrt(2.0), 15.0])

    # Test real branch
    v = chebyshev.chebyshev_coefficients_to_values(c)
    assert np.linalg.norm(v - v_true, np.inf) < tol
    assert not np.any(v.imag)

    # Test imaginary branch
    v = chebyshev.chebyshev_coefficients_to_values(1j * c)
    assert np.linalg.norm(v - 1j * v_true, np.inf) < tol
    assert not np.any(v.real)

    # Test general branch
    v = chebyshev.chebyshev_coefficients_to_values((1+1j) * c)
    assert np.linalg.norm(v - (1 + 1j) * v_true, np.inf) < tol

    # Test for symmetry preservation
    c = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    v = chebyshev.chebyshev_coefficients_to_values(c)
    assert np.linalg.norm(v - np.flipud(v), np.inf) == 0.0

    c = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    v = chebyshev.chebyshev_coefficients_to_values(c)
    assert np.linalg.norm(v + np.flipud(v), np.inf) == 0.0


def test_values_to_coefficients() -> None:
    tol = 100 * np.spacing(1)

    # Test that a single value is converted correctly
    v = np.array([np.sqrt(2)])
    c = chebyshev.chebyshev_values_to_coefficients(v)
    assert v == c

    # Some simple data
    v = np.r_[1:6]
    # Exact coefficients
    c_true = np.array([3, 1 + 1/np.sqrt(2), 0, 1 - 1/np.sqrt(2), 0])

    # Test real branch
    c = chebyshev.chebyshev_values_to_coefficients(v)
    assert np.linalg.norm(c - c_true, np.inf) < tol
    assert not np.any(c.imag)

    # Test imaginary branch
    c = chebyshev.chebyshev_values_to_coefficients(1j * v)
    assert np.linalg.norm(c - 1j * c_true, np.inf) < tol
    assert not np.any(c.real)

    # Test general branch
    c = chebyshev.chebyshev_values_to_coefficients((1 + 1j) * v)
    assert np.linalg.norm(c - (1 + 1j) * c_true, np.inf) < tol

    # Test for symmetry preservation
    v = np.array([1.1, -2.2, 3.3, -2.2, 1.1])
    c = chebyshev.chebyshev_values_to_coefficients(v)
    assert np.linalg.norm(c[1::2], np.inf) == 0.0
    v = np.array([1.1, -2.2, 0.0, 2.2, -1.1])
    c = chebyshev.chebyshev_values_to_coefficients(v)
    assert linalg.norm(c[::2], np.inf) == 0.0


def test_chebyshev_points() -> None:
    # Test that n = 0 returns empty results:
    x = chebyshev.chebyshev_points(0)
    assert x.size == 0
    assert len(x) == 0

    # Test n = 1:
    x = chebyshev.chebyshev_points(1)
    assert x.size == 1
    assert len(x) == 1
    assert x[0] == 0.0

    # Test that n = 2 returns [-1 , 1]:
    x = chebyshev.chebyshev_points(2)
    assert len(x) == 2
    assert np.all(x == np.array([-1.0, 1.0]))

    # Test that n = 3 returns [-1, 0, 1]:
    x = chebyshev.chebyshev_points(3)
    assert len(x) == 3
    assert np.all(x == np.array([-1.0, 0.0, 1.0]))

    # % Test that n = 129 returns vectors of the correct size:
    n = 129
    x = chebyshev.chebyshev_points(n)
    assert len(x) == n
    assert np.linalg.norm(x[0:int((n-1)/2)] + np.flipud(x[int((n+1)/2):]), np.inf) == 0.0
    assert x[int((n-1)/2)] == 0.0


def test_alias() -> None:
    tol = 100 * np.spacing(1)

    # Testing a vector of coefficients.
    c0 = np.r_[10:0:-1]

    # Padding:
    c1 = chebyshev.alias_chebyshev_coefficients(c0, 11)
    assert linalg.norm(np.r_[c0, 0.0] - c1, np.inf) == 0.0

    # Aliasing:
    c2 = chebyshev.alias_chebyshev_coefficients(c0, 9)
    assert linalg.norm(np.r_[10:3:-1, 4.0, 2.0] - c2, np.inf) == 0.0
    c3 = chebyshev.alias_chebyshev_coefficients(c0, 3)
    assert linalg.norm(np.array([18.0, 25.0, 12.0]) - c3, np.inf) == 0.0

    # Compare against result of evaluating on a smaller grid:
    v = chebyshev.chebyshev_clenshaw_evaluation(chebyshev.chebyshev_points(9), c0)
    assert linalg.norm(chebyshev.chebyshev_values_to_coefficients(v) - c2, np.inf) < tol

    v = chebyshev.chebyshev_clenshaw_evaluation(chebyshev.chebyshev_points(3), c0)
    assert linalg.norm(chebyshev.chebyshev_values_to_coefficients(v) - c3, np.inf) < tol

    #
    # Test aliasing a large tail.
    c0 = 1.0 / np.arange(1.0, 1001.0)**5
    n = 17
    c1 = chebyshev.alias_chebyshev_coefficients(c0, n)
    assert len(c1) == n
    # This should give the same result as evaluating via clenshaw
    v2 = chebyshev.chebyshev_clenshaw_evaluation(chebyshev.chebyshev_points(n), c0)
    c2 = chebyshev.chebyshev_values_to_coefficients(v2)
    assert linalg.norm(c1 - c2, np.inf) < n*tol


def test_derivative() -> None:
    tol = 100 * np.spacing(1)

    # Testing a vector of coefficients.
    d = chebyshev.chebyshev_coefficients_of_derivative([])
    assert len(d) == 0

    c = np.array([0.0])
    d = chebyshev.chebyshev_coefficients_of_derivative(c)
    assert np.array_equal(c, d)

    c = np.array([10.0])
    d = chebyshev.chebyshev_coefficients_of_derivative(c)
    assert np.array_equal(0.0 * c, d)

    c = np.array([0.0, 4.0])
    d = chebyshev.chebyshev_coefficients_of_derivative(c)
    assert np.array_equal(np.array([4.0]), d)

    # x**2
    c = np.array([1.0/2.0, 0, 1.0/2.0])
    # 2*x
    c_exact = np.array([0.0, 2.0])
    d = chebyshev.chebyshev_coefficients_of_derivative(c)
    assert np.linalg.norm(d - c_exact, np.inf) < tol

    # This function takes about 41 points to resolve:
    xx = np.linspace(-1.0, 1.0, 2001)

    def f(x):
        return x + np.sin(2.0 * np.pi * x) + np.cos(2.0 * np.pi * x**2)

    def fp(x):
        return 1.0 + 2.0 * np.pi * np.cos(2.0 * np.pi * x) - 4.0 * np.pi * x * np.sin(2.0 * np.pi * x**2)

    x_ = chebyshev.chebyshev_points(60)
    c = chebyshev.chebyshev_values_to_coefficients(f(x_))
    d = chebyshev.chebyshev_coefficients_of_derivative(c)

    error = chebyshev.chebyshev_clenshaw_evaluation(xx, c) - f(xx)
    assert np.linalg.norm(error, np.inf) < 10 * tol
    error = chebyshev.chebyshev_clenshaw_evaluation(xx, d) - fp(xx)
    assert np.linalg.norm(error, np.inf) < 10 * tol


def test_cumsum() -> None:
    tol = 100 * np.spacing(1)

    # Testing a vector of coefficients.
    d = chebyshev.chebyshev_coefficients_of_integral([])
    assert len(d) == 0

    c = np.array([0.0])
    d = chebyshev.chebyshev_coefficients_of_integral(c)
    assert np.array_equal(np.array([0.0, 0.0]), d)

    c = np.array([10.0])
    d = chebyshev.chebyshev_coefficients_of_integral(c)
    # 10 * x + 10
    assert np.array_equal(np.array([10.0, 10.0]), d)

    # x
    c = np.array([0.0, 1.0])
    d = chebyshev.chebyshev_coefficients_of_integral(c)
    # integral is x^2/2 - 1/2
    assert np.array_equal(np.array([-1.0/4.0, 0.0, 1.0/4.0]), d)

    # This function takes about 41 points to resolve:
    xx = np.linspace(-1.0, 1.0, 2001)

    def f(x):
        return 1.0 + 2.0 * np.pi*np.cos(2.0 * np.pi * x) - 4.0 * np.pi * x * np.sin(2.0 * np.pi * x**2)

    def F(x):  # noqa
        return x + np.sin(2.0 * np.pi*x) + np.cos(2.0 * np.pi*x**2)

    def integral(x):
        return F(x) - F(-1)

    x_ = chebyshev.chebyshev_points(60)
    c = chebyshev.chebyshev_values_to_coefficients(f(x_))
    d = chebyshev.chebyshev_coefficients_of_integral(c)

    error = chebyshev.chebyshev_clenshaw_evaluation(xx, c) - f(xx)
    assert np.linalg.norm(error, np.inf) < 10 * tol
    error = chebyshev.chebyshev_clenshaw_evaluation(xx, d) - integral(xx)
    assert np.linalg.norm(error, np.inf) < 10 * tol


def test_clenshaw_evaluation():
    # Set a tolerance (pref.chebfunnp.spacing(1) doesn't matter)
    tol = 10*np.spacing(1)

    # Test that a single coefficient is evaluated correctly:
    # For an evaluation with one point only
    c = np.array([np.sqrt(2.0)])
    v = chebyshev.chebyshev_clenshaw_evaluation([0], c)
    assert c == v

    # For a vector evaluation:
    x = np.array([-.5, 1.0])
    v = chebyshev.chebyshev_clenshaw_evaluation(x, c)
    assert len(v) == 2
    assert np.array_equal([c[0], c[0]], v)

    # Test that a vector coefficient is evaluated correctly:
    # Some simple data :
    c = np.arange(5, 0, -1)
    x = np.array([-.5, -.1, 1.0])

    # Scalar coefficient
    v = chebyshev.chebyshev_clenshaw_evaluation(x, c)
    # Exact values:
    v_true = np.array([3.0, 3.1728, 15.0])
    assert np.linalg.norm(v - v_true, np.inf) < tol

    # another set of coefficients
    v_true2 = np.array([0, 3.6480, 15.0])
    v = chebyshev.chebyshev_clenshaw_evaluation(x, np.flipud(c))
    assert np.linalg.norm(v - v_true2, np.inf) < tol


# TODO: the following can be appended to a test:
# from chebyshev import chebyshev_coefficients_to_values
#
# p_n = lambda n: np.concatenate([np.zeros(n), [1.0]])
#
# print(chebyshev_coefficients_to_values(p_n(0)))
# print(chebyshev_coefficients_to_values(p_n(1)))
# print(chebyshev_coefficients_to_values(p_n(2)))
# print(chebyshev_coefficients_to_values(p_n(3)))
# print(chebyshev_coefficients_to_values(p_n(4)))
# print(chebyshev_coefficients_to_values(p_n(5)))
# # Coefficients of x**2 in the Chebyshev T basis
# print(chebyshev_coefficients_to_values(np.array([1/2, 0, 1/2])))


# if __name__ == '__main__':
#     # f = chebyshev('x + np.cos(x) - 1')
#     # f = chebyshev('1 + x + x**2 + x**3 + x**4')
#     # f.poly()
#
#     suite = unittest.TestLoader().loadTestsFromTestCase(TestChebyshevMethods)
#     unittest.TextTestRunner(verbosity=2).run(suite)
#     #suite = unittest.TestSuite()
#     #suite.addTest(TestchebyshevMethods('test_mul'))
#     #runner = unittest.TextTestRunner(verbosity=2)
#     #runner.run(suite)
#
#     #coeffs = np.array([\
#     #    -2.348023134420334e-01, \
#     #     1.000000000000000e+00, \
#     #    -2.298069698638009e-01, \
#     #                         0, \
#     #     4.953277928219901e-03, \
#     #     6.902385958693666e-18, \
#     #    -4.187667600481507e-05, \
#     #    -2.043108576058326e-17, \
#     #     1.884468834568858e-07, \
#     #    -1.268726182963996e-17, \
#     #    -5.261229800551637e-10, \
#     #     1.157775533567783e-17, \
#     #     9.999481711403524e-13, \
#     #     2.134620574478668e-17, \
#     #    -1.373900992973631e-15, \
#     #     4.163336342344337e-17, \
#     #    -1.387778780781446e-17, \
#     #                         0, \
#     #    -4.163336342344337e-17, \
#     #                         0, \
#     #    -1.799775606325937e-17, \
#     #    -3.576819105040345e-17, \
#     #    -1.927846987950788e-17, \
#     #    -2.545554314349229e-17, \
#     #     2.095665390625546e-18, \
#     #    -2.656504963745442e-17, \
#     #     4.582787057824667e-17, \
#     #     3.430887356839772e-17, \
#     #     1.431146867680866e-17, \
#     #    -6.981586047542980e-20, \
#     #     1.387778780781446e-17, \
#     #                         0, \
#     #                         0])
#     #print(chebyshev.standard_chop(coeffs, np.spacing(1)))
