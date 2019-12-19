#!/usr/bin/python
from recommender import get_recommendations
from datahandler import NewUSsDataHandler
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

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
    data_handler = NewUSsDataHandler()

    uss = data_handler.load_us_data()
    acs = pd.read_csv('data/USsCAs.csv')
    # Merge entre os dois dataframes com base no id da User Story
    ussacs = pd.merge(uss, acs, how='left', on='ID_US')
    # Criação de um aggregator padrão para juntar os ids dos casos de teste padrão em uma só coluna.
    foo = lambda a: ",".join(a)
    # Transformando a coluna de id do critério de aceitação padrão em string para poder ser agregado.
    ussacs['ID_STD_AC'] = ussacs['ID_STD_AC'].astype(str)
    # Aqui é feito o agrupamento de todos os critérios de uma user story em apenas uma coluna.
    filtered_uss = ussacs.groupby(by=['ID_US', 'Módulo', 'Operação', 'Plataforma', 'RNFs']).agg(
        CAs=('ID_STD_AC', foo)).reset_index()
    kf = KFold(n_splits=10)
    precisions = []
    recalls = []
    for train_index, test_index in kf.split(filtered_uss):
        train_set = filtered_uss.iloc[train_index, :]
        test_set = filtered_uss.iloc[test_index, :]
        precisions_fold = []
        recalls_fold = []
        for us_test in test_set.iterrows():
            recommendations = get_recommendations(us_test[1], train_set, 2)
            real = data_handler.load_us_test_data(us_test[1]['ID_US'])
            real = real.drop_duplicates()
            intersect_ds = real.loc[real['ID_STD_TC'].isin(recommendations['ID']), :]
            recall = len(intersect_ds.index)/len(real.index)
            precision = len(intersect_ds.index)/len(recommendations.index)
            precisions_fold.append(precision);
            recalls_fold.append(recall)
        precisions.append(precisions_fold)
        recalls.append(recalls_fold)
    print(precisions)
    print(recalls)





    # recommendations = get_recommendations(df, 2)
    # print(recommendations)