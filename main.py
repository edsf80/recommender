#!/usr/bin/python
from validator import cross_validate
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    algorithms = ['jaccard', 'cosine', 'euclidean', 'chebyshev' ]
    k = np.arange(1, 13, step=1)

    for algorithm in algorithms:
        precision_list = []
        recall_list = []
        f_measures_list = []

        for i in k:
            precisions, recalls, f_measures = cross_validate(10, i, metric=algorithm)

            print('K = ', i)
            print('Precision: ', np.mean(precisions))
            precision_list.append(np.mean(precisions))

            print('Recall: ', np.mean(recalls))
            recall_list.append(np.mean(recalls))

            print('F-measure: ', np.mean(f_measures))
            f_measures_list.append(np.mean(f_measures))

        fig, ax = plt.subplots()
        ax.plot(k, precision_list, label='Precision', marker='*')
        ax.plot(k, recall_list, label='Recall', marker='o')
        ax.plot(k, f_measures_list, label='F-Measure', marker='d')
        ax.legend(loc='lower left', frameon=False)
        ax.set_ylabel('Valor')
        ax.set_xlabel('K')
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
