import matplotlib.pyplot as plt
import numpy as np

# Definir a função e sua derivada e hessiana
def f(x):
    return x**4 - 3*x**3 + 2 
def df(x):
    return 4*x**3 - 9*x**2 
def ddf(x):
    return 12*x**2 - 18*x 

# Definir o chute inicial e a tolerância
x0 = -1 # chute inicial
tol = 1e-6 # tolerância

# Criar uma lista para armazenar os valores de x em cada iteração
x_list = [x0]

# Iniciar o loop do método de Newton
while True:
    # Calcular o próximo x usando a fórmula
    x1 = x0 - df(x0)/ddf(x0)
    # Verificar se a diferença entre x1 e x0 é menor que a tolerância
    if abs(x1 - x0) < tol:
        break # sair do loop se for verdadeiro
    else:
        x0 = x1 # atualizar o valor de x0
        x_list.append(x0) # adicionar o valor de x0 na lista

# Imprimir o resultado
print("O ponto crítico aproximado é:", x1)

# Criar um vetor com valores de x entre -2 e 3 para plotar a função f(x)
x_vec = np.linspace(-2,3,30)

# Plotar a função f(x) em azul com uma linha contínua
plt.plot(x_vec,f(x_vec),color="blue",linestyle="-")

# Plotar os valores de x em cada iteração em vermelho com marcadores circulares
plt.plot(x_list,f(np.array(x_list)),color="red",marker="o")

# Calcular a derivada da função em cada ponto do vetor x_vec
df_vec = df(x_vec)

# Adicionar legendas aos eixos e ao gráfico
plt.xlabel("x")
plt.ylabel("f(x)")

# Mostrar o gráfico na tela
plt.show()
