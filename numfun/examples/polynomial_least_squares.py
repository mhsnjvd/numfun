import numpy as np
import matplotlib.pyplot as plt
from polyfit import polyfit

n_pts = 101
x = np.linspace(-1, 1, n_pts)
g = lambda x: 1.0 / (1.0 + 25.0 * x**2) + 1.0e-1*np.random.randn(len(x))
n = 15
f = polyfit(g(x), x, n, [-1, 1])

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x, g(x), 'xr')
f.plot()
plt.grid(True)
plt.title(f'Discrete polynomial least-squares fit of degree {n}')

plt.subplot(2, 1, 2)
plt.plot(x, g(x) - f(x), '.k')
plt.grid(True)
plt.title(f'residuals')


g = lambda x: np.abs(x + .2) - .5 * np.sign(x - .5)
x = np.linspace(-1, 1, 201)
n = 11
f = polyfit(g(x), x, n, [-1, 1])

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x, g(x), 'k')
f.plot()
plt.title('Continuous polynomial least-squares fit')
plt.subplot(2, 1, 2)
plt.plot(x, g(x) - f(x), 'k.')
plt.grid(True)
plt.title('residuals')
