#!/usr/bin/python
from recommender import get_recommendations
# import psycopg2
# from userstorydao import getUserStories
# from similaridadeeuclidiana import euclidiana
# from similaridadeeuclidiana import getRecomendacoes
#
#
# def connect():
#     """ Connect to the PostgreSQL database server """
#     try:
#         user_stories = getUserStories()
#         # Feature matrix [cadastro, gerencial, autenticacao, recuperacao de dados, atualizar dados, inserir dados, modificar insercao de dados, remover dados]
#         novaUS = [1, 0, 0, 0, 1, 0, 0, 0]
#         recomendacoes = getRecomendacoes(novaUS, user_stories, euclidiana)
#         print(recomendacoes)
#         #cur.close()
#     except (Exception, psycopg2.DatabaseError) as error:
#         print(error)


if __name__ == '__main__':
    newUS = {'Módulo': 'Cadastro', 'Operação': 'Inserir_dados', 'Plataforma': 'Web', 'CAs': '1,2,3,4,5,6,47',
             'RNFs': '1,2'}
    recommendations = get_recommendations(newUS, 2)
    print(recommendations)