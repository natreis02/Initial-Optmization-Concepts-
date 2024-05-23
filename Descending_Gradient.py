import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(gradient, start, learn_rate, n_iter=50, tolerance=1e-06):
    vector = start
     history = [vector]  # Armazena os valores intermediários do vetor
     for _ in range(n_iter):
        diff = -learn_rate * gradient(vector)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
        history.append(vector)
    return vector, history # Retorna tanto o vetor mínimo quanto a lista de histórico

# Exemplo de uso:
gradient = lambda x: 2 * x  # Gradiente da função x^2
minimum, history = gradient_descent(gradient, start=10.0, learn_rate=0.1)
print(f"Minimum of x^2 is at: {minimum}")

# Plotando o gráfico da função x^2
x = np.linspace(-10, 10, 100)
y = x ** 2
plt.plot(x, y)

# Plotando as flechas do gradiente descendente
history = np.array(history)
plt.quiver(history[:-1], history[:-1] ** 2,
           history[1:] - history[:-1], (history[1:] ** 2 - history[:-1] ** 2),
           scale_units='xy', angles='xy', scale=1)

plt.scatter(minimum, minimum ** 2,color='red')
plt.show()
