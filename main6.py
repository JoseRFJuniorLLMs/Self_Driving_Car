import numpy as np

# Lendo arquivo com os dados
dados = np.loadtxt('numeros_sorteados.txt')

# Definindo amplitude mínima para considerar um movimento como onda ou correção
amplitude_minima = 10

# Encontrando as ondas
ondas = []
for i in range(len(dados)):
    if i < amplitude_minima:
        continue
    if all(dados[i] > dados[i - amplitude_minima:i]) and all(dados[i-amplitude_minima:i] != 0):
        ondas.append((i - amplitude_minima + 1, i))

# Encontrando as correções
correcoes = []
for i in range(len(dados)):
    if i < amplitude_minima:
        continue
    if all(dados[i] < dados[i - amplitude_minima:i]) and all(dados[i-amplitude_minima:i] != 0):
        correcoes.append((i - amplitude_minima + 1, i))

# Verificando se a quantidade de ondas e correções está correta
if len(ondas) != len(correcoes):
    print("Erro: a quantidade de ondas não é igual à quantidade de correções")
else:
    total_movimentos = len(ondas) + len(correcoes)
    print("Total de ondas: ", len(ondas))
    print("Total de correções: ", len(correcoes))
    print("Total de movimentos: ", total_movimentos)

# Imprimindo os resultados
for i in range(len(ondas)):
    print("Onda ", i + 1, ": do número ", ondas[i][0], " ao número ", ondas[i][1])
    print("Correção ", i + 1, ": do número ", correcoes[i][0], " ao número ", correcoes[i][1])
