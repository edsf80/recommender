from math import sqrt

def euclidiana(item1, item2):
    soma = sum(
        [pow(item1[i] - item2[i], 2) for i in range(len(item1)) if i > 0])
    return 1 / (1 + sqrt(soma))

def getRecomendacoes(item, dados, funcao):
    totais = {}
    contaSimilaridade = {}

    for outro in dados:
        similaridade = funcao(item, outro['features'])

        if similaridade < 0.5: continue

        for valor in outro['testcases']:
            totais.setdefault(valor['desc'], 0)
            totais[valor['desc']] += similaridade
            contaSimilaridade.setdefault(valor['desc'], 0)
            contaSimilaridade[valor['desc']] += 1

    rankings = [(total/len(totais), item) for item, total in totais.items() if total/len(totais) > 0.5]
    rankings.sort()
    rankings.reverse()
    return rankings
