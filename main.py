#!/usr/bin/python
from validator import cross_validate
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import recommender as rec
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import stats
import statsmodels.formula.api as smf
import pandas as pd

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_style('italic')

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    # algorithms = ['canberra', 'jaccard', 'cosine', 'euclidean', 'chebyshev' ]
    withHeuristcs = [False, True]
    algorithms = [rec.METRIC_COSINE, rec.METRIC_CANBERRA, rec.METRIC_EUCLIDEAN, rec.METRIC_JACCARD]

    k = np.arange(1, 13, step=1)
    result_dataset = pd.DataFrame(columns=['Precision', 'Recall', 'F-Measure', 'K', 'Algoritmo', 'Heuristic'])

    for withHeuristc in withHeuristcs:
        for algorithm in algorithms:
            precision_list = []
            recall_list = []
            f_measures_list = []

            for i in k:
                precisions, recalls, f_measures = cross_validate(10, i, metric=algorithm, heuristic=withHeuristc)

                print('K = ', i)
                print('Precision: ', np.mean(precisions))
                precision_list.append(np.mean(precisions))
                stat, p = shapiro(precisions)
                print('Statistics shapiro=%.3f, p=%.3f' % (stat, p))
                stat, p = normaltest(precisions)
                print('Statistics Agostino=%.3f, p=%.3f' % (stat, p))

                print('Recall: ', np.mean(recalls))
                recall_list.append(np.mean(recalls))
                stat, p = shapiro(recalls)
                print('Statistics=%.3f, p=%.3f' % (stat, p))
                stat, p = normaltest(recalls)
                print('Statistics Agostino=%.3f, p=%.3f' % (stat, p))

                print('F-measure: ', np.mean(f_measures))
                f_measures_list.append(np.mean(f_measures))
                stat, p = shapiro(f_measures)
                print('Statistics=%.3f, p=%.3f' % (stat, p))
                stat, p = normaltest(f_measures)
                print('Statistics Agostino=%.3f, p=%.3f' % (stat, p))

                results = {
                    'Precision': precisions,
                    'Recall': recalls,
                    'F-Measure': f_measures
                }

                df = pd.DataFrame(results, columns=['Precision', 'Recall', 'F-Measure'])
                df['K'] = i
                df['Algoritmo'] = algorithm
                df['Heuristic'] = withHeuristc
                result_dataset = result_dataset.append(df, ignore_index=True)

            fig, ax = plt.subplots()
            ax.plot(k, precision_list, label='Precision', marker='*')
            ax.plot(k, recall_list, label='Recall', marker='o')
            ax.plot(k, f_measures_list, label='F-Measure', marker='d')
            ax.legend(loc='lower left', frameon=False)
            ax.set_ylabel('Valor', fontproperties=font)
            ax.set_xlabel('K', fontproperties=font)
            ax.plot()
            for x, y in zip(k, precision_list):
                label = "{:.3f}".format(y)

                plt.annotate(label,  # this is the text
                             (x, y),  # this is the point to label
                             textcoords="offset points",  # how to position the text
                             xytext=(0, 10),  # distance from text to points (x,y)
                             ha='center')  # horizontal alignment can be left, right or center

            for x, y in zip(k, recall_list):
                label = "{:.3f}".format(y)

                plt.annotate(label,  # this is the text
                             (x, y),  # this is the point to label
                             textcoords="offset points",  # how to position the text
                             xytext=(0, 10),  # distance from text to points (x,y)
                             ha='center')  # horizontal alignment can be left, right or center

            for x, y in zip(k, f_measures_list):
                label = "{:.3f}".format(y)

                plt.annotate(label,  # this is the text
                             (x, y),  # this is the point to label
                             textcoords="offset points",  # how to position the text
                             xytext=(0, 10),  # distance from text to points (x,y)
                             ha='center')  # horizontal alignment can be left, right or center
            plt.xticks(k)
            plt.title(algorithm)
            plt.show()

    test = smf.ols('Recall ~ K + Algoritmo + Heuristic', data=result_dataset).fit()
    print(test.summary())
    print(stats.f_oneway(result_dataset['Recall'][
                             (result_dataset['K'] == 1) & (result_dataset['Algoritmo'] == rec.METRIC_COSINE) & (
                                         result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 2) & (result_dataset['Algoritmo'] == rec.METRIC_COSINE) & (
                                         result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 3) & (result_dataset['Algoritmo'] == rec.METRIC_COSINE) & (
                                         result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 4) & (result_dataset['Algoritmo'] == rec.METRIC_COSINE) & (
                                         result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 5) & (result_dataset['Algoritmo'] == rec.METRIC_COSINE) & (
                                         result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 6) & (result_dataset['Algoritmo'] == rec.METRIC_COSINE) & (
                                         result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 7) & (result_dataset['Algoritmo'] == rec.METRIC_COSINE) & (
                                         result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 8) & (result_dataset['Algoritmo'] == rec.METRIC_COSINE) & (
                                         result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 9) & (result_dataset['Algoritmo'] == rec.METRIC_COSINE) & (
                                         result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 10) & (result_dataset['Algoritmo'] == rec.METRIC_COSINE) & (
                                         result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 11) & (result_dataset['Algoritmo'] == rec.METRIC_COSINE) & (
                                         result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 12) & (result_dataset['Algoritmo'] == rec.METRIC_COSINE) & (
                                         result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 1) & (result_dataset['Algoritmo'] == rec.METRIC_CANBERRA) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 2) & (result_dataset['Algoritmo'] == rec.METRIC_CANBERRA) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 3) & (result_dataset['Algoritmo'] == rec.METRIC_CANBERRA) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 4) & (result_dataset['Algoritmo'] == rec.METRIC_CANBERRA) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 5) & (result_dataset['Algoritmo'] == rec.METRIC_CANBERRA) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 6) & (result_dataset['Algoritmo'] == rec.METRIC_CANBERRA) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 7) & (result_dataset['Algoritmo'] == rec.METRIC_CANBERRA) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 8) & (result_dataset['Algoritmo'] == rec.METRIC_CANBERRA) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 9) & (result_dataset['Algoritmo'] == rec.METRIC_CANBERRA) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 10) & (result_dataset['Algoritmo'] == rec.METRIC_CANBERRA) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 11) & (result_dataset['Algoritmo'] == rec.METRIC_CANBERRA) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 12) & (result_dataset['Algoritmo'] == rec.METRIC_CANBERRA) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 1) & (result_dataset['Algoritmo'] == rec.METRIC_EUCLIDEAN) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 2) & (result_dataset['Algoritmo'] == rec.METRIC_EUCLIDEAN) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 3) & (result_dataset['Algoritmo'] == rec.METRIC_EUCLIDEAN) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 4) & (result_dataset['Algoritmo'] == rec.METRIC_EUCLIDEAN) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 5) & (result_dataset['Algoritmo'] == rec.METRIC_EUCLIDEAN) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 6) & (result_dataset['Algoritmo'] == rec.METRIC_EUCLIDEAN) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 7) & (result_dataset['Algoritmo'] == rec.METRIC_EUCLIDEAN) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 8) & (result_dataset['Algoritmo'] == rec.METRIC_EUCLIDEAN) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 9) & (result_dataset['Algoritmo'] == rec.METRIC_EUCLIDEAN) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 10) & (result_dataset['Algoritmo'] == rec.METRIC_EUCLIDEAN) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 11) & (result_dataset['Algoritmo'] == rec.METRIC_EUCLIDEAN) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 12) & (result_dataset['Algoritmo'] == rec.METRIC_EUCLIDEAN) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 1) & (result_dataset['Algoritmo'] == rec.METRIC_JACCARD) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 2) & (result_dataset['Algoritmo'] == rec.METRIC_JACCARD) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 3) & (result_dataset['Algoritmo'] == rec.METRIC_JACCARD) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 4) & (result_dataset['Algoritmo'] == rec.METRIC_JACCARD) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 5) & (result_dataset['Algoritmo'] == rec.METRIC_JACCARD) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 6) & (result_dataset['Algoritmo'] == rec.METRIC_JACCARD) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 7) & (result_dataset['Algoritmo'] == rec.METRIC_JACCARD) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 8) & (result_dataset['Algoritmo'] == rec.METRIC_JACCARD) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 9) & (result_dataset['Algoritmo'] == rec.METRIC_JACCARD) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 10) & (result_dataset['Algoritmo'] == rec.METRIC_JACCARD) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 11) & (result_dataset['Algoritmo'] == rec.METRIC_JACCARD) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 12) & (result_dataset['Algoritmo'] == rec.METRIC_JACCARD) & (
                                     result_dataset['Heuristic'] == False)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 1) & (result_dataset['Algoritmo'] == rec.METRIC_COSINE) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 2) & (result_dataset['Algoritmo'] == rec.METRIC_COSINE) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 3) & (result_dataset['Algoritmo'] == rec.METRIC_COSINE) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 4) & (result_dataset['Algoritmo'] == rec.METRIC_COSINE) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 5) & (result_dataset['Algoritmo'] == rec.METRIC_COSINE) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 6) & (result_dataset['Algoritmo'] == rec.METRIC_COSINE) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 7) & (result_dataset['Algoritmo'] == rec.METRIC_COSINE) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 8) & (result_dataset['Algoritmo'] == rec.METRIC_COSINE) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 9) & (result_dataset['Algoritmo'] == rec.METRIC_COSINE) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 10) & (result_dataset['Algoritmo'] == rec.METRIC_COSINE) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 11) & (result_dataset['Algoritmo'] == rec.METRIC_COSINE) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 12) & (result_dataset['Algoritmo'] == rec.METRIC_COSINE) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 1) & (result_dataset['Algoritmo'] == rec.METRIC_CANBERRA) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 2) & (result_dataset['Algoritmo'] == rec.METRIC_CANBERRA) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 3) & (result_dataset['Algoritmo'] == rec.METRIC_CANBERRA) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 4) & (result_dataset['Algoritmo'] == rec.METRIC_CANBERRA) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 5) & (result_dataset['Algoritmo'] == rec.METRIC_CANBERRA) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 6) & (result_dataset['Algoritmo'] == rec.METRIC_CANBERRA) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 7) & (result_dataset['Algoritmo'] == rec.METRIC_CANBERRA) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 8) & (result_dataset['Algoritmo'] == rec.METRIC_CANBERRA) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 9) & (result_dataset['Algoritmo'] == rec.METRIC_CANBERRA) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 10) & (result_dataset['Algoritmo'] == rec.METRIC_CANBERRA) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 11) & (result_dataset['Algoritmo'] == rec.METRIC_CANBERRA) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 12) & (result_dataset['Algoritmo'] == rec.METRIC_CANBERRA) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 1) & (result_dataset['Algoritmo'] == rec.METRIC_EUCLIDEAN) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 2) & (result_dataset['Algoritmo'] == rec.METRIC_EUCLIDEAN) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 3) & (result_dataset['Algoritmo'] == rec.METRIC_EUCLIDEAN) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 4) & (result_dataset['Algoritmo'] == rec.METRIC_EUCLIDEAN) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 5) & (result_dataset['Algoritmo'] == rec.METRIC_EUCLIDEAN) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 6) & (result_dataset['Algoritmo'] == rec.METRIC_EUCLIDEAN) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 7) & (result_dataset['Algoritmo'] == rec.METRIC_EUCLIDEAN) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 8) & (result_dataset['Algoritmo'] == rec.METRIC_EUCLIDEAN) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 9) & (result_dataset['Algoritmo'] == rec.METRIC_EUCLIDEAN) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 10) & (result_dataset['Algoritmo'] == rec.METRIC_EUCLIDEAN) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 11) & (result_dataset['Algoritmo'] == rec.METRIC_EUCLIDEAN) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 12) & (result_dataset['Algoritmo'] == rec.METRIC_EUCLIDEAN) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 1) & (result_dataset['Algoritmo'] == rec.METRIC_JACCARD) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 2) & (result_dataset['Algoritmo'] == rec.METRIC_JACCARD) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 3) & (result_dataset['Algoritmo'] == rec.METRIC_JACCARD) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 4) & (result_dataset['Algoritmo'] == rec.METRIC_JACCARD) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 5) & (result_dataset['Algoritmo'] == rec.METRIC_JACCARD) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 6) & (result_dataset['Algoritmo'] == rec.METRIC_JACCARD) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 7) & (result_dataset['Algoritmo'] == rec.METRIC_JACCARD) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 8) & (result_dataset['Algoritmo'] == rec.METRIC_JACCARD) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 9) & (result_dataset['Algoritmo'] == rec.METRIC_JACCARD) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 10) & (result_dataset['Algoritmo'] == rec.METRIC_JACCARD) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 11) & (result_dataset['Algoritmo'] == rec.METRIC_JACCARD) & (
                                     result_dataset['Heuristic'] == True)],
                         result_dataset['Recall'][
                             (result_dataset['K'] == 12) & (result_dataset['Algoritmo'] == rec.METRIC_JACCARD) & (
                                     result_dataset['Heuristic'] == True)]

                         ))
