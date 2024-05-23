import numpy as np
from scipy.optimize import minimize

# Função a ser interpolada
def f(x):
    return 2 * x ** 3 - 7 * x ** 2 + 3 * x + 5

# Pontos de dados
x_data = np.array([-2, -1, 0, 1, 2])
y_data = f(x_data)


# Definir a função de erro para minimizar
def error_func(c):
    return np.sum((np.polyval(c, x_data) - y_data) ** 2)

# Grau do polinômio
poly_degree = 3

# Chute inicial para os coeficientes do polinômio
c0 = np.zeros(poly_degree + 1)

# Minimizar a função de erro usando o método de Powell
result = minimize(error_func, c0, method='Powell')

# Imprimir os coeficientes do polinômio interpolador
print('Coeficientes do polinômio interpolador:', result.x)

# Plotar a função original e o polinômio interpolador
import matplotlib.pyplot as plt

x_plot = np.linspace(-3, 3, 100)
y_plot = f(x_plot)
y_poly = np.polyval(result.x, x_plot)

plt.plot(x_plot, y_plot, label='Função original')
plt.plot(x_plot, y_poly, label='Polinômio interpolador')
plt.scatter(x_data, y_data, color='red', label='Pontos de dados')
plt.legend()
plt.show()

