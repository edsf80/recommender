#!/usr/bin/python
from validator import cross_validate
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import recommender as rec
import statsmodels.formula.api as smf
import statsmodels.stats.anova as an
import statsmodels.stats.multicomp as multi
import pandas as pd
import matplotlib as mpl
from scipy.stats import friedmanchisquare


mpl.rc('font', family='Times New Roman', size=12)

warnings.filterwarnings("ignore")


def generate_results():
    with_heuristcs = [False, True]
    algorithms = [rec.METRIC_COSINE, rec.METRIC_EUCLIDEAN, rec.METRIC_CHEBYSHEV]

    k = np.arange(1, 11, step=1)
    result_dataset = pd.DataFrame(columns=['Precision', 'Recall', 'FMeasure', 'Configuracao', 'K', 'Algoritmo', 'Heuristica'])

    for withHeuristc in with_heuristcs:
        for algorithm in algorithms:
            precision_list = []
            recall_list = []
            f_measures_list = []

            for i in k:
                precisions, recalls, f_measures = cross_validate(10, i, metric=algorithm, heuristic=withHeuristc)

                print('Calculando K = ', i, ' para o algoritmo ', algorithm, ' Heurística: ', withHeuristc)
                precision_list.append(np.mean(precisions))
                recall_list.append(np.mean(recalls))
                f_measures_list.append(np.mean(f_measures))

                results = {
                    'Precision': precisions,
                    'Recall': recalls,
                    'FMeasure': f_measures
                }

                df = pd.DataFrame(results, columns=['Precision', 'Recall', 'FMeasure'])
                df['Configuracao'] = algorithm + '_' + str(withHeuristc) + '_' + str(i)
                df['K'] = i
                df['Algoritmo'] = algorithm
                df['Heuristica'] = 'sim' if withHeuristc else 'nao'
                result_dataset = result_dataset.append(df, ignore_index=True)

            # fig, ax = plt.subplots()
            # ax.plot(k, precision_list, label='Precision', marker='*')
            # ax.plot(k, recall_list, label='Recall', marker='o')
            # ax.plot(k, f_measures_list, label='F-Measure', marker='d')
            # ax.legend(loc='lower left', frameon=False)
            # ax.set_ylabel('Valor', fontproperties=font)
            # ax.set_xlabel('K', fontproperties=font)
            # ax.plot()
            # for x, y in zip(k, precision_list):
            #     label = "{:.3f}".format(y)
            #
            #     plt.annotate(label,  # this is the text
            #                  (x, y),  # this is the point to label
            #                  textcoords="offset points",  # how to position the text
            #                  xytext=(0, 10),  # distance from text to points (x,y)
            #                  ha='center')  # horizontal alignment can be left, right or center
            #
            # for x, y in zip(k, recall_list):
            #     label = "{:.3f}".format(y)
            #
            #     plt.annotate(label,  # this is the text
            #                  (x, y),  # this is the point to label
            #                  textcoords="offset points",  # how to position the text
            #                  xytext=(0, 10),  # distance from text to points (x,y)
            #                  ha='center')  # horizontal alignment can be left, right or center
            #
            # for x, y in zip(k, f_measures_list):
            #     label = "{:.3f}".format(y)
            #
            #     plt.annotate(label,  # this is the text
            #                  (x, y),  # this is the point to label
            #                  textcoords="offset points",  # how to position the text
            #                  xytext=(0, 10),  # distance from text to points (x,y)
            #                  ha='center')  # horizontal alignment can be left, right or center
            # plt.xticks(k)
            # plt.title(algorithm)
            # plt.show()

    result_dataset.to_csv(r'./data/results.csv', index=False, header=True)
    # test = smf.ols('FMeasure ~ Configuracao', data=result_dataset).fit()
    # print(test.summary())
    # aov_table = an.anova_lm(test, typ=2)
    # print(aov_table)
    # mc1 = multi.MultiComparison(result_dataset['FMeasure'], result_dataset['Configuracao'])
    # mc_results = mc1.tukeyhsd()
    # print(mc_results)


def analyze_data():
    data = pd.read_csv(r'./data/results_fmeasure_2-10Ks.csv')

    stat, p = friedmanchisquare(data[(data['Algoritmo'] == 'chebyshev') & (data['K'] == 1) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'chebyshev') & (data['K'] == 2) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'chebyshev') & (data['K'] == 3) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'chebyshev') & (data['K'] == 4) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'chebyshev') & (data['K'] == 5) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'chebyshev') & (data['K'] == 6) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'chebyshev') & (data['K'] == 7) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'chebyshev') & (data['K'] == 8) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'chebyshev') & (data['K'] == 9) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'chebyshev') & (data['K'] == 10) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'euclidean') & (data['K'] == 1) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'euclidean') & (data['K'] == 2) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'euclidean') & (data['K'] == 3) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'euclidean') & (data['K'] == 4) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'euclidean') & (data['K'] == 5) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'euclidean') & (data['K'] == 6) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'euclidean') & (data['K'] == 7) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'euclidean') & (data['K'] == 8) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'euclidean') & (data['K'] == 9) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'euclidean') & (data['K'] == 10) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'cosine') & (data['K'] == 1) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'cosine') & (data['K'] == 2) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'cosine') & (data['K'] == 3) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'cosine') & (data['K'] == 4) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'cosine') & (data['K'] == 5) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'cosine') & (data['K'] == 6) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'cosine') & (data['K'] == 7) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'cosine') & (data['K'] == 8) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'cosine') & (data['K'] == 9) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'cosine') & (data['K'] == 10) & (data['Heuristica'] == 'nao')]['FMeasure'],
                                data[(data['Algoritmo'] == 'chebyshev') & (data['K'] == 1) & (data['Heuristica'] == 'sim')]['FMeasure'],
                                data[(data['Algoritmo'] == 'chebyshev') & (data['K'] == 2) & (data['Heuristica'] == 'sim')]['FMeasure'],
                                data[(data['Algoritmo'] == 'chebyshev') & (data['K'] == 3) & (
                                            data['Heuristica'] == 'sim')]['FMeasure'],
                                data[(data['Algoritmo'] == 'chebyshev') & (data['K'] == 4) & (
                                            data['Heuristica'] == 'sim')]['FMeasure'],
                                data[(data['Algoritmo'] == 'chebyshev') & (data['K'] == 5) & (
                                            data['Heuristica'] == 'sim')]['FMeasure'],
                                data[(data['Algoritmo'] == 'chebyshev') & (data['K'] == 6) & (
                                            data['Heuristica'] == 'sim')]['FMeasure'],
                                data[(data['Algoritmo'] == 'chebyshev') & (data['K'] == 7) & (
                                            data['Heuristica'] == 'sim')]['FMeasure'],
                                data[(data['Algoritmo'] == 'chebyshev') & (data['K'] == 8) & (
                                            data['Heuristica'] == 'sim')]['FMeasure'],
                                data[(data['Algoritmo'] == 'chebyshev') & (data['K'] == 9) & (
                                            data['Heuristica'] == 'sim')]['FMeasure'],
                                data[(data['Algoritmo'] == 'chebyshev') & (data['K'] == 10) & (
                                            data['Heuristica'] == 'sim')]['FMeasure'],
                                data[(data['Algoritmo'] == 'euclidean') & (data['K'] == 1) & (
                                            data['Heuristica'] == 'sim')]['FMeasure'],
                                data[(data['Algoritmo'] == 'euclidean') & (data['K'] == 2) & (
                                            data['Heuristica'] == 'sim')]['FMeasure'],
                                data[(data['Algoritmo'] == 'euclidean') & (data['K'] == 3) & (
                                            data['Heuristica'] == 'sim')]['FMeasure'],
                                data[(data['Algoritmo'] == 'euclidean') & (data['K'] == 4) & (
                                            data['Heuristica'] == 'sim')]['FMeasure'],
                                data[(data['Algoritmo'] == 'euclidean') & (data['K'] == 5) & (
                                            data['Heuristica'] == 'sim')]['FMeasure'],
                                data[(data['Algoritmo'] == 'euclidean') & (data['K'] == 6) & (
                                            data['Heuristica'] == 'sim')]['FMeasure'],
                                data[(data['Algoritmo'] == 'euclidean') & (data['K'] == 7) & (
                                            data['Heuristica'] == 'sim')]['FMeasure'],
                                data[(data['Algoritmo'] == 'euclidean') & (data['K'] == 8) & (
                                            data['Heuristica'] == 'sim')]['FMeasure'],
                                data[(data['Algoritmo'] == 'euclidean') & (data['K'] == 9) & (
                                            data['Heuristica'] == 'sim')]['FMeasure'],
                                data[(data['Algoritmo'] == 'euclidean') & (data['K'] == 10) & (
                                            data['Heuristica'] == 'sim')]['FMeasure'],
                                data[
                                    (data['Algoritmo'] == 'cosine') & (data['K'] == 1) & (data['Heuristica'] == 'sim')][
                                    'FMeasure'],
                                data[
                                    (data['Algoritmo'] == 'cosine') & (data['K'] == 2) & (data['Heuristica'] == 'sim')][
                                    'FMeasure'],
                                data[
                                    (data['Algoritmo'] == 'cosine') & (data['K'] == 3) & (data['Heuristica'] == 'sim')][
                                    'FMeasure'],
                                data[
                                    (data['Algoritmo'] == 'cosine') & (data['K'] == 4) & (data['Heuristica'] == 'sim')][
                                    'FMeasure'],
                                data[
                                    (data['Algoritmo'] == 'cosine') & (data['K'] == 5) & (data['Heuristica'] == 'sim')][
                                    'FMeasure'],
                                data[
                                    (data['Algoritmo'] == 'cosine') & (data['K'] == 6) & (data['Heuristica'] == 'sim')][
                                    'FMeasure'],
                                data[
                                    (data['Algoritmo'] == 'cosine') & (data['K'] == 7) & (data['Heuristica'] == 'sim')][
                                    'FMeasure'],
                                data[
                                    (data['Algoritmo'] == 'cosine') & (data['K'] == 8) & (data['Heuristica'] == 'sim')][
                                    'FMeasure'],
                                data[
                                    (data['Algoritmo'] == 'cosine') & (data['K'] == 9) & (data['Heuristica'] == 'sim')][
                                    'FMeasure'],
                                data[(data['Algoritmo'] == 'cosine') & (data['K'] == 10) & (
                                            data['Heuristica'] == 'sim')]['FMeasure']
                                )
    print(stat)
    print(p)


def plot_graphs():
    data = pd.read_csv(r'./data/results.csv')
    csfont = {'fontname': 'Times New Roman', 'fontsize': 14}

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data[(data['Algoritmo'] == 'chebyshev') & (data['Heuristica'] == 'nao')].groupby('K').mean()[
                'Recall'].to_frame(), label='Chebyshev', marker='*')
    ax.plot(data[(data['Algoritmo'] == 'euclidean') & (data['Heuristica'] == 'nao')].groupby('K').mean()[
                'Recall'].to_frame(), label='Euclidean', marker='o')
    ax.plot(data[(data['Algoritmo'] == 'cosine') & (data['Heuristica'] == 'nao')].groupby('K').mean()[
                'Recall'].to_frame(), label='Cosine', marker='d')
    ax.plot(data[(data['Algoritmo'] == 'chebyshev') & (data['Heuristica'] == 'sim')].groupby('K').mean()[
                'Recall'].to_frame(), label='Chebyshev Heuristica',
            marker='*')
    ax.plot(data[(data['Algoritmo'] == 'euclidean') & (data['Heuristica'] == 'sim')].groupby('K').mean()[
                'Recall'].to_frame(), label='Euclidean Heuristica',
            marker='o')
    ax.plot(data[(data['Algoritmo'] == 'cosine') & (data['Heuristica'] == 'sim')].groupby('K').mean()[
                'Recall'].to_frame(), label='Cosine Heuristica', marker='d')
    ax.legend(loc='upper right', frameon=False)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_xlabel('K', fontsize=12)
    ax.yaxis.grid()
    ax.plot()
    plt.xticks(np.arange(1, 11, step=1))
    plt.title('Recalls by algorithm')
    plt.tight_layout()
    plt.savefig('recalls.png')
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data[(data['Algoritmo'] == 'chebyshev') & (data['Heuristica'] == 'nao')].groupby('K').mean()[
                'Precision'].to_frame(), label='Chebyshev', marker='*')
    ax.plot(data[(data['Algoritmo'] == 'euclidean') & (data['Heuristica'] == 'nao')].groupby('K').mean()[
                'Precision'].to_frame(), label='Euclidean', marker='o')
    ax.plot(data[(data['Algoritmo'] == 'cosine') & (data['Heuristica'] == 'nao')].groupby('K').mean()[
                'Precision'].to_frame(), label='Cosine', marker='d')
    ax.plot(data[(data['Algoritmo'] == 'chebyshev') & (data['Heuristica'] == 'sim')].groupby('K').mean()[
                'Precision'].to_frame(), label='Chebyshev Heuristic',
            marker='*')
    ax.plot(data[(data['Algoritmo'] == 'euclidean') & (data['Heuristica'] == 'sim')].groupby('K').mean()[
                'Precision'].to_frame(), label='Euclidean Heuristic',
            marker='o')
    ax.plot(data[(data['Algoritmo'] == 'cosine') & (data['Heuristica'] == 'sim')].groupby('K').mean()[
                'Precision'].to_frame(), label='Cosine Heuristic', marker='d')
    ax.legend(loc='upper right', frameon=False)
    ax.set_ylabel('Value')
    ax.set_xlabel('K')
    ax.yaxis.grid()
    ax.plot()
    plt.xticks(np.arange(1, 11, step=1))
    plt.title('Precisions by algorithm')
    plt.tight_layout()
    plt.savefig('precisoes.png')
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data[(data['Algoritmo'] == 'chebyshev') & (data['Heuristica'] == 'nao')].groupby('K').mean()['FMeasure'].to_frame(), label='Chebyshev', marker='*')
    ax.plot(data[(data['Algoritmo'] == 'euclidean') & (data['Heuristica'] == 'nao')].groupby('K').mean()['FMeasure'].to_frame(), label='Euclidean', marker='o')
    ax.plot(data[(data['Algoritmo'] == 'cosine') & (data['Heuristica'] == 'nao')].groupby('K').mean()['FMeasure'].to_frame(), label='Cosine', marker='d')
    ax.plot(data[(data['Algoritmo'] == 'chebyshev') & (data['Heuristica'] == 'sim')].groupby('K').mean()['FMeasure'].to_frame(), label='Chebyshev Heuristic',
            marker='*')
    ax.plot(data[(data['Algoritmo'] == 'euclidean') & (data['Heuristica'] == 'sim')].groupby('K').mean()['FMeasure'].to_frame(), label='Euclidean Heuristic',
            marker='o')
    ax.plot(data[(data['Algoritmo'] == 'cosine') & (data['Heuristica'] == 'sim')].groupby('K').mean()['FMeasure'].to_frame(), label='Cosine Heuristic', marker='d')
    ax.legend(loc='upper right', frameon=False)
    ax.set_ylabel('Value')
    ax.set_xlabel('K')
    ax.yaxis.grid()
    ax.plot()
    plt.xticks(np.arange(1, 11, step=1))
    plt.title('F-Measures by algorithm')
    plt.tight_layout()
    plt.savefig('f-measures.png')
    plt.show()


def plot_online_evolution():
    data = pd.read_csv('./data/online_registration.csv', delimiter=';')
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
    df_data = data.groupby(['Date', 'Status'])['Merge'].count().reset_index()
    table = pd.pivot_table(df_data, values='Merge', index='Date', columns='Status', fill_value=0)
    table['Total_Developed'] = table['Developed'] + table['Reuse']
    table['Total_Recomended'] = table['Accepted'] + table['Reuse'] + table['Rejected']
    table['Total_Accepted'] = table['Accepted'] + table['Reuse']
    table['Increase_Percent'] = (table['Accepted'] / table['Total_Developed']) * 100
    table['Precision'] = (table['Total_Accepted'] / table['Total_Recomended']) * 100
    table['Reuse_Percent'] = (table['Reuse'] / table['Total_Developed']) * 100
    table['Reject_Percent'] = (table['Rejected'] / table['Total_Recomended']) * 100

    table['Percentual de incremento'] = table['Increase_Percent']

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.yaxis.grid()
    ax.plot(table.loc[:, ['Increase_Percent']], marker='o', label='Percentual de incremento')
    ax.plot(table.loc[:, ['Precision']], marker='d', label='Precisão')
    ax.plot(table.loc[:, ['Reuse_Percent']], marker='x', label='Percentual de reuso')
    ax.plot(table.loc[:, ['Reject_Percent']], marker='p', label='Percentual de rejeição')
    ax.legend(loc='lower left', frameon=False)

    plt.title('Evolução das métricas no tempo')
    plt.tight_layout()
    plt.savefig('metrics_evolution.pdf')
    plt.show()


if __name__ == '__main__':
    plot_online_evolution()
