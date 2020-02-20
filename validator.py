from datahandler import NewUSsDataHandler
from sklearn.model_selection import KFold
from recommender import get_recommendations
import pandas as pd
import numpy as np


def cross_validate(fold_length, k, metric='euclidean'):
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
                f_measure = 2 * ((precision * recall) / (precision + recall))
            f_measures.append(f_measure)
            precisions.append(precision)
            recalls.append(recall)
    return precisions, recalls, f_measures
