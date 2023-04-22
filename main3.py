import turtle
import numpy as np

# Configurações da tela
WIDTH = 800
HEIGHT = 600
turtle.setup(WIDTH, HEIGHT)
turtle.title("Projeto Turtle - Curva de Hilbert")

# Criação da Tartaruga vermelha
tartaruga_vermelha = turtle.Turtle(shape='turtle')
tartaruga_vermelha.color('red')
tartaruga_vermelha.goto(-300, 0)

# Criação da Tartaruga verde
tartaruga_verde = turtle.Turtle(shape='turtle')
tartaruga_verde.color('green')
tartaruga_verde.goto(-300, 100)

# Criação da Tartaruga para exibição da distância percorrida
tartaruga_distancia = turtle.Turtle()
tartaruga_distancia.penup()
tartaruga_distancia.goto(0, 250)

# Matriz de transição de estado
A = np.array([[1, 0], [0, 1]])

# Matriz de medida
H = np.array([[1, 0], [0, 1]])

# Matriz de erro da medida
R = np.array([[10, 0], [0, 10]])

# Matriz de erro do estado inicial
P = np.array([[10, 0], [0, 10]])

# Variância do ruído do processo
Q = np.array([[1, 0], [0, 1]])

# Estado inicial estimado
x = np.array([[-300, 0], [0, -100]]).reshape(-1, 1)

# Covariância inicial do estado estimado
cov = P

# Curva de Hilbert
def hilbert_curve(n, turtle, tamanho):
    if n <= 0:
        return

    turtle.right(90)
    hilbert_curve(n - 1, turtle, tamanho)
    turtle.forward(tamanho)
    turtle.left(90)
    hilbert_curve(n - 1, turtle, tamanho)
    turtle.forward(tamanho)
    hilbert_curve(n - 1, turtle, tamanho)
    turtle.left(90)
    turtle.forward(tamanho)
    hilbert_curve(n - 1, turtle, tamanho)
    turtle.right(90)

# Loop infinito para desenhar a curva de Hilbert continuamente
tamanho = 30
n = 4
while True:
    # Movimento da Tartaruga vermelha
    hilbert_curve(n, tartaruga_vermelha, tamanho)

    # Verificar se a tartaruga vermelha atingiu as bordas da tela e mudar de direção
    if tartaruga_vermelha.xcor() < -WIDTH / 2 or tartaruga_vermelha.xcor() > WIDTH / 2:
        tartaruga_vermelha.right(180)
    if tartaruga_vermelha.ycor() < -HEIGHT / 2 or tartaruga_vermelha.ycor() > HEIGHT / 2:
        tartaruga_vermelha.right(180)

    # Predição do movimento da Tartaruga verde usando filtro de Kalman
    x_pred = A @ x
    cov_pred = A @ cov @ A.T + Q

    # Medição do movimento da Tartaruga vermelha
    z = np.array([[tartaruga_vermelha.xcor()], [tartaruga_vermelha.ycor()]])
    y = z - H @ x_pred
    S = H @ cov_pred @ H.T + R

    # Atualização do estado estimado e da covariância
    K = cov_pred @ H.T @ np.linalg.inv(S)
    x = x_pred + K @ y
    cov = (np.eye(2) - K @ H) @ cov_pred

    # Movimento da Tartaruga verde baseado na estimativa do filtro de Kalman
    tartaruga_verde.setx(x[0][0])
    tartaruga_verde.sety(x[1][0])

    # Movimento da Tartaruga verde tentando prever o movimento da Tartaruga vermelha
    z_pred = H @ x_pred
    tartaruga_verde.setheading(tartaruga_verde.towards(z_pred[0], z_pred[1]))
    tartaruga_verde.forward(5)

    # Verificar se a tartaruga verde atingiu as bordas da tela e mudar de direção
    if tartaruga_verde.xcor() < -WIDTH / 2 or tartaruga_verde.xcor() > WIDTH / 2:
        tartaruga_verde.right(180)
    if tartaruga_verde.ycor() < -HEIGHT / 2 or tartaruga_verde.ycor() > HEIGHT / 2:
        tartaruga_verde.right(180)

turtle.mainloop()
