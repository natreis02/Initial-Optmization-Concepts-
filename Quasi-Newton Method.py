import numpy as np
from scipy.optimize import line_search
import matplotlib.pyplot as plt 

# Definir a função e seu gradiente
def f(x):
    return x[0]**4 - 3*x[0]**3 + 2*x[1]**2
def df(x):
    return np.array([4*x[0]**3 - 9*x[0]**2, 4*x[1]])

# Definir o chute inicial e a tolerância
x0 = np.array([-1, 1]) # chute inicial
tol = 1e-6 # tolerância

# Criar uma lista para armazenar os valores de x em cada iteração
x_list = [x0]

# Iniciar a matriz inversa da hessiana aproximada como a identidade
H = np.eye(2)

# Iniciar o loop do método BFGS
while True:

 # Calcular o gradiente no ponto atual
    g = df(x0)

 # Verificar se o gradiente é menor que a tolerância
    if np.linalg.norm(g) < tol:
        break # sair do loop se for verdadeiro
    else:
  # Calcular a direção de busca como o produto da matriz inversa da hessiana aproximada       pelo gradiente negativo
        d = -H.dot(g)

   # Fazer uma busca linear na direção d usando a condição de Armijo
   alpha = line_search(f, df, x0, d)[0]

   # Calcular o próximo x usando a fórmula x + alpha*d
        x1 = x0 + alpha*d

   # Calcular a diferença entre os pontos x1 e x0 
        s = x1 - x0 

    # Calcular a diferença entre os gradientes em x1 e x0 
        y = df(x1) - g 

   # Atualizar a matriz inversa da hessiana aproximada usando a fórmula BFGS 
        H = H + (s.dot(y) + y.dot(H).dot(y))*(np.outer(s,s))/(s.dot(y))**2 - (np.outer(H.dot(y),s) + np.outer(s,y.dot(H)))/(s.dot(y))

   # Atualizar o valor de x0 
        x0 = x1 

    # Adicionar o valor de x0 na lista 
        x_list.append(x0)

# Imprimir o resultado 
print("O ponto crítico aproximado é:", x1)
