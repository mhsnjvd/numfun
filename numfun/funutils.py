import numpy as np
from numba import njit


#@njit
def split_array_at_boundaries(x: np.array, boundary: np.array, out: list):
    m = len(boundary) - 1
    n = len(x)
    if n == 0 or m == 0:
        return

    mask = np.zeros((n,))
    for k in range(m):
        a = boundary[k]
        b = boundary[k + 1]
        if k == m - 1:
            mask = np.bitwise_and(a <= x, x <= b)
        else:
            mask = np.bitwise_and(a <= x, x < b)
        nk = int(np.sum(mask))
        xk = np.zeros((nk,))
        xk = x[mask]
        out.append(xk)
    return


def inf_norm_of_derivatives(f, domain=np.array([-1.0, 1.0]), order=4, grid_size=50):
    """Compute the approximate infinity norm of derivatives of order up to order"""
    a = domain[0]
    b = domain[-1]

    max_der = np.zeros(order)
    na = a * np.zeros(order)
    nb = b * np.zeros(order)

    dx = (b - a) / (grid_size - 1.0)
    x = np.linspace(a, b, grid_size, endpoint=True)
    y = f(x)

    dy = y.copy()
    for j in range(order):
        dy = np.abs(np.diff(dy))
        x = (x[:-1] + x[1:]) / 2.0
        ind = np.argmax(np.abs(dy))
        max_der[j] = np.abs(dy[ind])
        if ind > 0:
            na[j] = x[ind-1]
        if ind < len(x) - 2:
            nb[j] = x[ind+1]

    if dx**order <= np.spacing(0):
        # Avoid divisions by zero
        max_der = np.inf + max_der
    else:
        # Get norm_inf of derivatives.
        max_der = max_der / dx**np.arange(1, order + 1)

    return max_der, na, nb


def detect_edge(f, domain=np.array([-1.0, 1.0]), v_scale=1.0, h_scale=2.0) -> float:

    # By default, there is no edge
    edge = np.nan

    order = 4                   # Maximum number of derivatives to be tested.
    grid_size_1 = 50            # Grid size for 1st finite difference computations.
    grid_size_234 = 15          # Grid size for higher derivative computations in loop.

    # Compute norm_inf of first derivatives up to and including order
    max_der, new_a, new_b = inf_norm_of_derivatives(f, domain, order, grid_size_1)

    ends = np.array([new_a[-1], new_b[-1]])

    while (not np.isinf(max_der[-1]) and
           not np.isnan(max_der[-1]) and
           np.diff(ends)[0] > np.spacing(1) * h_scale):

        # Keep track of previous max derivatives:
        max_der_prev = max_der.copy()

        max_der, new_a, new_b = inf_norm_of_derivatives(f, ends, order, grid_size_234)
        # Choose how many derivatives to test in this iteration:
        mask_a = max_der > (5.5 - np.arange(1.0, order + 1.0)) * max_der_prev
        mask_b = max_der > (10.0 * v_scale / (h_scale**np.arange(1.0, order + 1.0)))
        mask = np.bitwise_and(mask_a, mask_b)
        nz = np.nonzero(mask)[0]

        print(ends)

        if len(nz) == 0:
            # Derivatives are not growing;
            return np.zeros((0,))
        else:
            order = nz[0] + 1

        if (order == 1) and (np.diff(ends)[0] < 1e-3*h_scale):
            edge = find_discontinuity(f, ends, v_scale, h_scale)

        ends = np.array([new_a[order-1],  new_b[order-1]])

    edge = np.mean(ends)
    return edge


def find_discontinuity(f, domain=np.array([-1.0, 1.0]), v_scale=1.0, h_scale=2.0):
    a = domain[0]
    b = domain[-1]

    edge = np.nan

    fa = f(a)
    fb = f(b)

    max_der = np.abs(fa - fb) / (b - a)

    if max_der < 1.0e-5 * v_scale / h_scale:
        return edge

    # Keep track how many times derivative stopped growing:
    cont = 0

    # Estimate edge location:
    e1 = (b + a) / 2.0

    # Main loop: (Note that max_der = inf whenever dx < realmin)
    # make e0 and e1 unequal to force start the loop
    e0 = e1 + 1
    while ((cont < 2) or np.any(np.isinf(max_der))) and (e0 != e1):
        # Evaluate OP at c, the center of the interval[a, b]:
        c = (a + b) / 2.0
        fc = f(c)

        # Find the undivided difference on each side of interval
        dyl = np.abs(fc - fa)
        dyr = np.abs(fb - fc)

        # Keep track of maximum value:
        max_der_prev = max_der

        if dyl > dyr:
            # Blow - up seems to be in [a, c].Bisect:
            b = c
            fb = fc
            # Update maxd:
            max_der = dyl / (b - a)
        else:
            # Blow - up seems to be in [c, b].Bisect:
            a = c
            fa = fc
            # Update maxd:
            max_der = dyr / (b - a)

        # Update edge location:
        e0 = e1
        e1 = (a + b) / 2.0

        # Test must fail twice before breaking the loop:
        if max_der < max_der_prev * 1.5:
            cont = cont + 1

    if (e0 - e1) <= 2 * np.spacing(e0):
        # Look at the floating point at the right:
        yright = f(b + np.spacing(b))
        # If there is a small jump, that is it!
        if np.abs(yright - fb) > np.spacing(1) * 100 * v_scale:
            edge = b
        else:
            edge = a

    return edge
