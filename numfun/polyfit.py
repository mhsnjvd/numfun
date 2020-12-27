from typing import Sequence, Union

import numpy as np
from numba import njit

from numfun.function import Function


# noinspection PyPep8Naming
@njit
def polyfit_jit(x: np.array, y: np.array, n: int, a: float, b: float) -> tuple:
    """Jitted version of polyfit, see polyfit for details.

    :param x: nodes
    :param y: data
    :param n: degree of the polynomial fit
    :param a: left end point of domain
    :param b: right end point of domain
    :return: the output of np.linalg.lstsq()
    """
    m = len(x)
    Tx = np.zeros((m, n + 1))
    Tx[:, 0] = np.ones(m)
    x_map = 2.0 * (x - a) / (b - a) - 1.0
    Tx[:, 1] = x_map
    for k in range(1, n):
        Tx[:, k + 1] = 2.0 * x_map * Tx[:, k] - Tx[:, k-1]

    # TODO: this is not compiling :(
    # Initialize variables for jit:
    c = np.zeros((n+1,))  # noqa
    residuals = np.zeros((1,))  # noqa
    rank = int(0)  # noqa
    singular_values = np.zeros((n+1,))  # noqa

    c, residuals, rank, singular_values = np.linalg.lstsq(Tx, y, rcond=None)

    return c, residuals, rank, singular_values


def polyfit(
        x: np.ndarray,
        y: np.ndarray,
        degree: Union[int, Sequence[int]] = 1,
        domain: Union[None, np.ndarray, Sequence[float]] = None
) -> Function:
    """Least squares polynomial fitting to discrete data with piecewise domain splitting handled.

    :param x:
    :param y:
    :param degree: an array or a double len(degree) = len(domain) - 1
    :param domain: an array to specify where the breakpionts are
    :return:
    """

    if domain is None or len(domain) == 2:
        assert isinstance(degree, int)
        return polyfit_global(x, y, degree, domain)
    if len(domain) > 2:
        n_pieces = len(domain) - 1
        if isinstance(degree, Sequence):  # A list of degrees is passed
            assert n_pieces == len(degree), 'must specify degree for each domain'
            degrees = degree
        else:  # The degree passed is just an integer
            degrees = n_pieces * [int(degree)]
    else:
        raise AssertionError(f'domain = {domain}, domain must be for the form [a, b]')

    all_coefficients = n_pieces * [None]
    for j in range(n_pieces):
        a, b = domain[j], domain[j+1]
        idx = (a <= x) & (x <= b)
        if np.sum(idx) == 0:
            c = np.array([])
        else:
            xj, yj = x[idx], y[idx]
            f = polyfit_global(xj, yj, degree=degrees[j], domain=[a, b])
            c = f.coefficients[0].copy()
        all_coefficients[j] = c

    return Function(coefficients=all_coefficients, domain=domain)


def polyfit_global(
        x: np.ndarray,
        y: np.ndarray,
        degree: int = 1,
        domain: Union[None, Sequence[float], np.ndarray] = None
) -> Function:
    """Degree n least squares polynomial approximation of data y taken on points x in a domain.

    :param x: x-values, np array
    :param y: y-values, i.e., data values, np array
    :param degree: degree of approximation, an integer
    :param domain: domain of approximation
    :return:
    """

    if domain is None:
        domain = [np.min(x), np.max(x)]

    domain = 1.0 * np.array(domain)
    a = domain[0]
    b = domain[-1]
    assert len(x) == len(y), f"len(x) = {len(x)}, while len(y) = {len(y)}, these must be equal"
    assert degree == int(degree), f'degree = {degree}, degree must be an integer'

    n = int(degree)

    # map points to [-1, 1]
    m = len(x)
    x_normalized = 2.0 * (x - a) / (b - a) - 1.0
    # construct the Chebyshev-Vandermonde matrix:
    Tx = np.zeros((m, n+1))
    Tx[:, 0] = np.ones(m)
    if n > 0:
        Tx[:, 1] = x_normalized
        for k in range(1, n):
            Tx[:, k + 1] = 2.0 * x_normalized * Tx[:, k] - Tx[:, k-1]

    # c, residuals, rank, singular_values = polyfit_jit(x, y, n, a, b)
    c, residuals, rank, singular_values = np.linalg.lstsq(Tx, y, rcond=None)

    # Make a function:
    return Function(coefficients=c, domain=domain)


def main() -> None:
    import matplotlib.pyplot as plt
    x = np.linspace(0, 10, 11)
    y = x ** 2
    n = 2
    domain = [0, 10]
    f = polyfit(x, y, n, domain)
    plt.plot(x, y, '.')
    f.plot()

    x = np.linspace(0, 5, 201)
    y = x * (x < 2.0) + x ** 2 * ((x >= 2.0) & (x < 3.0)) + \
        x ** 3 * ((x >= 3.0) & (x < 4.0)) + x ** 4 * (x >= 4.0)
    domain = [0, 1, 2, 3, 4, 5]
    degrees = [0, 1, 2, 3, 4]
    f = polyfit(x, y, degree=degrees, domain=domain)
    f.plot()


if __name__ == '__main__':
    main()
