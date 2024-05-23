import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x[0]**2 + 2*x[1]**2 + 2*x[0]*x[1]

def grad_f(x):
    return np.array([2*x[0] + 2*x[1], 4*x[1] + 2*x[0]])

def conjugate_gradient(f, grad_f, x0, n_iter=100):
    x = x0
    p = -grad_f(x)
    history = [x]
    for i in range(n_iter):
        denominator = np.dot(p, np.dot(hessian_f(x), p))
	if denominator == 0:
	      alpha = 0
	else:
        	       alpha = -np.dot(grad_f(x), p) / (denominator + 1e-6)
        x = x + alpha*p
        beta = np.dot(grad_f(x), np.dot(hessian_f(x), p)) / np.dot(p, np.dot(hessian_f(x), p))
        p = -grad_f(x) + beta*p
        history.append(x)
    return history

def hessian_f(x):
    return np.array([[2, 2], [2, 4]])

x0 = np.array([1, 1])
history = conjugate_gradient(f, grad_f, x0)

# Plotando a convergência do método
history = np.array(history)

# Calculando os valores da função objetivo para uma grade de pontos no espaço de soluções
xgrid, ygrid = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
zgrid = np.zeros_like(xgrid)

for i in range(zgrid.shape[0]):
    for j in range(zgrid.shape[1]):
        zgrid[i,j] = f(np.array([xgrid[i,j], ygrid[i,j]]))


# Gerando o gráfico de contorno da função objetivo e a trajetória das estimativas de solução        
plt.contour(xgrid, ygrid, zgrid, cmap='jet')

# Usando plt.scatter() 
plt.scatter(history[:, 0], history[:, 1], c=np.arange(len(history)), cmap='rainbow')
plt.scatter(history[0, 0], history[0, 1], color='green', label='Ponto inicial')
plt.scatter(history[-1, 0], history[-1, 1], color='red', label='Ponto final')

# Plotando as linhas que ligam os pontos
plt.plot(history[:, 0], history[:, 1], '-k')

# Plotando as flechas do gradiente em cada ponto
for point in history:
    dx , dy= grad_f(point) # Calculando o gradiente no ponto
    plt.arrow(point[0], point[1], dx*0.01 , dy*0.01 , fc="k", ec="k", head_width=0.09 , head_length=0.08) # Desenhando a flecha com uma escala de tamanho

# Definindo os limites dos eixos (xmin=-10,xmax=10,ymin=-10,xmax=10)
plt.xlim(-4 ,4 )
plt.ylim(-4 ,4 )

print("Solução encontrada: ", history[-1])

plt.show()
