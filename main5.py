import turtle
import random
import time

# Configurações da janela do Turtle
tela = turtle.Screen()
tela.bgcolor("white")
tela.title("Roleta")

# Cria uma tartaruga para exibir o texto
texto_tartaruga = turtle.Turtle()
texto_tartaruga.hideturtle()
texto_tartaruga.penup()
texto_tartaruga.goto(0, -100)

# Lista dos números e cores da roleta
numeros = ["0", "00", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36"]
cores = {"0": "verde", "00": "verde", "1": "vermelho", "2": "preto", "3": "vermelho", "4": "preto", "5": "vermelho", "6": "preto", "7": "vermelho", "8": "preto", "9": "vermelho", "10": "preto", "11": "preto", "12": "vermelho", "13": "preto", "14": "vermelho", "15": "preto", "16": "vermelho", "17": "preto", "18": "vermelho", "19": "vermelho", "20": "preto", "21": "vermelho", "22": "preto", "23": "vermelho", "24": "preto", "25": "vermelho", "26": "preto", "27": "vermelho", "28": "preto", "29": "preto", "30": "vermelho", "31": "preto", "32": "vermelho", "33": "preto", "34": "vermelho", "35": "preto", "36": "vermelho"}

# Loop infinito para sortear e salvar números
while True:
    # Seleciona um número e cor aleatórios
    numero_sorteado = random.choice(numeros)
    cor_sorteada = cores[numero_sorteado]

    # Exibe o número e cor selecionados
    print("Número sorteado: " + numero_sorteado)
    print("Cor sorteada: " + cor_sorteada.capitalize())

    # Define a cor do texto de acordo com a cor sorteada
    if cor_sorteada == "vermelho":
        texto_tartaruga.color("red")
    elif cor_sorteada == "preto":
        texto_tartaruga.color("black")

    # Escreve o número e cor selecionados na janela do Turtle
    texto_tartaruga.clear()
    texto_tartaruga.write(f"Número sorteado: {numero_sorteado}\nCor sorteada: {cor_sorteada.capitalize()}", align="center", font=("Arial", 24, "normal"))

    # Salva o número e cor selecionados em um arquivo txt
    with open("numeros_sorteados.txt", "a") as f:
        f.write(numero_sorteado + "," + cor_sorteada + "\n")

    # Espera 5 segundos antes de sortear outro número
    time.sleep(5)
