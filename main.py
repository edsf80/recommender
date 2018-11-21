#!/usr/bin/python
import psycopg2
from userstorydao import getUserStories
from similaridadeeuclidiana import euclidiana
from similaridadeeuclidiana import getRecomendacoes


def connect():
    """ Connect to the PostgreSQL database server """
    try:
        user_stories = getUserStories()
        # Feature matrix [cadastro, gerencial, autenticacao, recuperacao de dados, atualizar dados, inserir dados, modificar insercao de dados, remover dados]
        novaUS = [1, 0, 0, 0, 1, 0, 0, 0]
        recomendacoes = getRecomendacoes(novaUS, user_stories, euclidiana)
        print(recomendacoes)
        #cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


if __name__ == '__main__':
    connect()