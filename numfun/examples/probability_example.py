import matplotlib.pyplot as plt
import numpy as np
from function import Function

f = Function(lambda x: np.heaviside(x, 0), domain=[-1, 0, 1])

# Define a probability distribution
a = 0
b = 100
domain = [a, b]
x = Function(lambda x: x, domain=domain)
f = Function(lambda x: 2.0 * np.exp(-2.0 * x), domain=domain)
f.plot()

# What is the expected value and the variance:
E = (x * f).definite_integral()
V = (x**2 * f).definite_integral() - E**2



#%%
f = Function(lambda x: np.sin(x) + np.sin(5*x**2), domain=[0, 10])
x = f.roots()
y = f(x)
plt.figure(figsize=(10, 8))
f.plot()
plt.plot(x, y, '.')


#%%
f = Function(fun=[lambda x: np.exp(-1.0/x**2), lambda x: 0 * x, lambda x: np.exp(-1./x**2)], domain=[-1, -np.spacing(1), np.spacing(1), 1])