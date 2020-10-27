from datahandler import NewUSsDataHandler
from sklearn.model_selection import KFold
from recommender import get_recommendations, get_recommendations_heuristcs
import pandas as pd
import numpy as np
from sklearn.neighbors import DistanceMetric


def cross_validate(fold_length, k, metric='euclidean', heuristic=False):
    data_handler = NewUSsDataHandler()

    uss = data_handler.load_us_data()
    kf = KFold(n_splits=fold_length)
    precisions = []
    recalls = []
    f_measures = []
    uss_x = uss.drop(columns=['TCs'])
    uss_y = uss.loc[:, ['ID_US', 'TCs']]

    for train_index, test_index in kf.split(uss_x):
        train_set = uss_x.iloc[train_index, :]
        test_set = uss_x.iloc[test_index, :]

        for us_test in test_set.iterrows():
            recommendations = pd.DataFrame()
            if heuristic:
                recommendations = get_recommendations_heuristcs(us_test[1], train_set, k, distance_metric=metric)
            else:
                recommendations = get_recommendations(us_test[1], train_set, k, distance_metric=metric)

            if recommendations.empty:
                f_measures.append(0)
                precisions.append(0)
                recalls.append(0)
                continue

            real = uss_y.loc[uss_y['ID_US'] == us_test[1]['ID_US'], 'TCs'].str.split(',')
            real = pd.Series(real.iloc[0])
            real = real.iloc[:].astype(float)

            rec_ids = pd.Series(recommendations['ID'])
            rec_ids = rec_ids.iloc[:].astype(float)

            intersect_ds = pd.Series(np.intersect1d(real, rec_ids))

            recall = len(intersect_ds.index) / len(real.index)
            precision = len(intersect_ds.index) / len(recommendations.index)
            f_measure = 0
            if precision + recall > 0:
                f_measure = calculate_fbeta(precision, recall, 2)
            f_measures.append(f_measure)
            precisions.append(precision)
            recalls.append(recall)
    return precisions, recalls, f_measures


def calculate_fbeta(precision, recall, beta):
    return 2 * (((beta**2) * precision * recall) / (((beta**2) * precision) + recall))


if __name__ == '__main__':
    data_handler = NewUSsDataHandler()

    uss = data_handler.load_us_data().drop(columns=['TCs'])

    newUS = {'ID_US': '#263', 'Módulo': 'Cadastro', 'Operação': 'Atualizar_dados', 'Plataforma': 'Web', 'RNFs': '1,2',
             'CAs': '5,6,7,8'}

    recommendations = get_recommendations(newUS, uss, 3, distance_metric='jaccard')

    print(recommendations.to_string())
