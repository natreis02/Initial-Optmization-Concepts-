import numpy as np
from scipy.optimize import minimize

# Definir a função Rosenbrock
def rosenbrock(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

# Definir um ponto inicial aleatório
x0 = np.random.randn(4)

# Chamar a função minimize com o método Nelder-Mead
res = minimize(rosenbrock, x0, method='Nelder-Mead')

# Imprimir o resultado
print(res)
