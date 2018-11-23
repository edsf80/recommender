from math import sqrt

def euclidiana(item1, item2):
    soma = sum(
        [pow(item1[i] - item2[i], 2) for i in range(len(item1)) if i > 0])
    return 1 / (1 + sqrt(soma))

def getRecomendacoes(item, dados, funcao):
    totais = {}
    totalSimilares = 0

    for outro in dados:
        similaridade = funcao(item, outro['features'])

        if similaridade < 0.5: continue

        totalSimilares += 1

        for valor in outro['testcases']:
            totais.setdefault(valor['desc'], 0)
            totais[valor['desc']] += similaridade

    rankings = [(total/totalSimilares, item) for item, total in totais.items() if total/totalSimilares > 0.5]
    rankings.sort()
    rankings.reverse()
    return rankings
