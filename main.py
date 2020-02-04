#!/usr/bin/python
from validator import cross_validate
from datahandler import NewUSsDataHandler
from recommender import get_recommendations
import numpy as np

if __name__ == '__main__':

    precisions, recalls, f_measures = cross_validate(10, 1)

    # print(precisions)
    print('Precision: ', np.mean(precisions))
    # print(recalls)
    print('Recall: ', np.mean(recalls))
    # print(f_measures)
    print('F-measure: ', np.mean(f_measures))

    # newUS = {'Módulo': 'Cadastro', 'Operação': 'Inserir_dados', 'Plataforma': 'Web', 'CAs': '1,2,3,4,5,6,47,51,56',
    #          'RNFs': '1,2'}
    # data_handler = NewUSsDataHandler()
    #
    # uss = data_handler.load_us_data().drop(columns=['TCs'])
    # print(uss.head())
    # recommendations = get_recommendations(newUS, uss, 1)
    # print(recommendations)

