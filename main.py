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
    newUS = {'Módulo': 'Cadastro', 'Operação': 'Inserir_dados', 'Plataforma': 'Web', 'CAs': '1,2,3,4,5,6,47,51,56',
             'RNFs': '1,2'}
    data_handler = NewUSsDataHandler()

    uss = data_handler.load_us_data()
    # print(uss.head())
    # recomendations = get_recommendations(newUS, uss, 1)
    # print(recomendations)
    kf = KFold(n_splits=10)
    precisions = []
    recalls = []
    fmeasures = []
    uss_x = uss.drop(columns=['TCs'])
    uss_y = uss.loc[:, ['ID_US', 'TCs']]

    for train_index, test_index in kf.split(uss_x):
        train_set = uss_x.iloc[train_index, :]
        test_set = uss_x.iloc[test_index, :]
        precisions_fold = []
        recalls_fold = []
        fmeasures_fold = []
        for us_test in test_set.iterrows():
            recommendations = get_recommendations(us_test[1], train_set, 1)

            if recommendations.empty:
                fmeasures_fold.append(0)
                precisions_fold.append(0)
                recalls_fold.append(0)
                continue

            real = uss_y.loc[uss_y['ID_US'] == us_test[1]['ID_US'], 'TCs'].str.split(',')
            real = pd.Series(real.iloc[0])
            real = real.iloc[:].astype(float)

            rec_ids = pd.Series(recommendations['ID'])
            rec_ids = rec_ids.iloc[:].astype(float)

            intersect_ds = pd.Series(np.intersect1d(real.iloc[0], rec_ids))

            recall = len(intersect_ds.index)/len(real.index)
            precision = len(intersect_ds.index)/len(recommendations.index)
            fmeasure = 0
            if precision+recall > 0:
                fmeasure = 2 * ((precision*recall)/(precision+recall))
            fmeasures_fold.append(fmeasure)
            precisions_fold.append(precision)
            recalls_fold.append(recall)
        precisions.append(precisions_fold)
        recalls.append(recalls_fold)
        fmeasures.append(fmeasures_fold)
    print(precisions)
    print(recalls)
    print(fmeasures)
