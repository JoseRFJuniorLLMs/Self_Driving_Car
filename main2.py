import turtle
import random

# Configurações da tela
WIDTH = 800
HEIGHT = 600
turtle.setup(WIDTH, HEIGHT)
turtle.title("Projeto Turtle - Curva de Hilbert")

# Criação da Tartaruga
tartaruga = turtle.Turtle(shape='turtle')
tartaruga.color('red')
tartaruga.penup()
tartaruga.goto(-300, 0)
tartaruga.pendown()

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
    turtle.left(90)
    hilbert_curve(n - 1, turtle, tamanho)
    turtle.forward(tamanho)
    turtle.right(90)
    hilbert_curve(n - 1, turtle, tamanho)
    turtle.right(90)
    hilbert_curve(n - 1, turtle, tamanho)
    turtle.forward(tamanho)
    turtle.left(90)
    hilbert_curve(n - 1, turtle, tamanho)
    turtle.forward(tamanho)
    turtle.left(90)
    hilbert_curve(n - 1, turtle, tamanho)
    turtle.forward(tamanho)
    turtle.right(90)
    hilbert_curve(n - 1, turtle, tamanho)

# Loop infinito para desenhar a curva de Hilbert continuamente
tamanho_inicial = 5
n = 4
tamanho_minimo = 1
tamanho_maximo = 30
while True:
    tamanho = random.randint(tamanho_minimo, tamanho_maximo)
    hilbert_curve(n, tartaruga, tamanho)

    # Verificar se a tartaruga atingiu as bordas da tela e mudar de direção
    if tartaruga.xcor() < -WIDTH/2 or tartaruga.xcor() > WIDTH/2:
        tartaruga.right(180)
    if tartaruga.ycor() < -HEIGHT/2 or tartaruga.ycor() > HEIGHT/2:
        tartaruga.right(180)

    # Verificar se o tamanho está abaixo do mínimo, aumentar tamanho e espessura da caneta
    if tamanho < tamanho_minimo:
        tamanho = tamanho_minimo
    tartaruga.pensize(tamanho/5)
    tartaruga.shapesize(tamanho/10)

turtle.mainloop()
