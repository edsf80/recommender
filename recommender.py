from datahandler import NewUSsDataHandler
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from scipy.spatial import distance

METRIC_EUCLIDEAN = 'euclidean'
METRIC_COSINE = 'cosine'
METRIC_CANBERRA = 'canberra'
METRIC_JACCARD = 'jaccard'


# def get_recommendations(new_us, base_uss, k, distance_metric='euclidean'):
#     """Decompose a nearest neighbors sparse graph into distances and indices
#
#         Parameters
#         ----------
#         new_us : {ID_US: string, Módulo: string, Operação: string, Plataforma: string, RNFs: string,
#                  CAs: string}
#             A dictionary representing the target User Story
#
#
#         base_uss : [{'ID_US', 'Módulo', 'Operação', 'Plataforma', 'RNFs', 'CAs'}]
#             The dataframe consisting of the recommender training data.
#
#         k : int > 0
#             The number of kneighbors to be considered.
#
#         distance_metric: str
#             The metric to calculate the distance among points.
#         Returns
#         -------
#         neigh_dist : array, shape (n_samples,) of arrays
#             Distances to nearest neighbors. Only present if return_distance=True.
#
#         neigh_ind :array, shape (n_samples,) of arrays
#             Indices of nearest neighbors.
#         """
#     data_handler = NewUSsDataHandler()
#
#     # Adicionando um filtro para trazer apenas uss com mesmo módulo e operação da nova US.
#     base_uss_filtrada = base_uss.loc[(base_uss['Módulo'] == new_us['Módulo']) & (base_uss['Operação'] == new_us['Operação']), :]
#
#     if base_uss_filtrada.empty:
#         return pd.DataFrame()
#
#     df = data_handler.load_filtered_us_data(new_us, base_uss_filtrada)
#
#     # Nesse ponto verifico que a base filtrada é menor que o K, pois não é possível calcular a distância com K maior
#     # que quantidade de itens.
#     if df.shape[0] < k:
#         return pd.DataFrame()
#
#     knn = NearestNeighbors(metric=distance_metric, algorithm='brute', n_neighbors=k).fit(df)
#
#     # Pega a quantidade de colunas para gerar uma lista de 1.
#     l = [1 for i in range(len(df.columns))]
#
#     distances, indices = knn.kneighbors([l])
#     candidate_uss = df.iloc[indices[0]]
#     candidate_uss.loc[:, 'similaridade'] = 1 / (1 + distances[0])
#
#     test_data = data_handler.load_test_data(candidate_uss.index.values)
#     test_data = test_data.merge(candidate_uss, left_on='ID_US', right_on='ID_US')
#
#     results = test_data.groupby("ID_STD_TC").sum()["similaridade"] / k
#     results = results.to_frame()
#
#     cts = data_handler.load_test_cases(results.index.values)
#
#     cts = cts.merge(results, left_on="ID", right_on="ID_STD_TC", how='inner')
#     return cts.sort_values('similaridade', ascending=False)


def get_recommendations(us_alvo, base_uss, k, distance_metric='euclidean'):
    """Decompose a nearest neighbors sparse graph into distances and indices

        Parameters
        ----------
        us_alvo : {ID_US: string, Módulo: string, Operação: string, Plataforma: string, RNFs: string,
                 CAs: string}
            A dictionary representing the target User Story


        base_uss : [{'ID_US', 'Módulo', 'Operação', 'Plataforma', 'RNFs', 'CAs'}]
            The dataframe consisting of the recommender training data.

        k : int > 0
            The number of kneighbors to be considered.

        distance_metric: str
            The metric to calculate the distance among points.
        Returns
        -------
        neigh_dist : array, shape (n_samples,) of arrays
            Distances to nearest neighbors. Only present if return_distance=True.

        neigh_ind :array, shape (n_samples,) of arrays
            Indices of nearest neighbors.
        """
    data_handler = NewUSsDataHandler()

    # Adicionando um filtro para trazer apenas uss com mesmo módulo e operação da nova US.
    base_uss_filtrada = base_uss.loc[
                        (base_uss['Módulo'] == us_alvo['Módulo']) & (base_uss['Operação'] == us_alvo['Operação']), :]

    if base_uss_filtrada.empty:
        return pd.DataFrame()

    # Retorna um conjunto de vetores de caracteristicas de acordo com a us alvo
    df = data_handler.load_filtered_us_data(us_alvo, base_uss_filtrada)

    # Nesse ponto verifico que a base filtrada é menor que o K, pois não é possível calcular a distância com K maior
    # que quantidade de itens.
    if df.shape[0] < k:
        return pd.DataFrame()

    # Pega a quantidade de colunas para gerar uma lista de 1.
    l = pd.Series([1 for i in range(len(df.columns))])

    df['similaridade'] = df.apply(lambda row: 1 / (1 + calculate_distance(row.array, l.array, distance_metric)), axis=1)

    tcs = pd.read_csv('data/USsTCs.csv', na_values='')

    test_count = tcs.groupby('ID_US')['ID_TC'].count()
    test_count = test_count.reset_index()
    df = df.merge(test_count, left_on='ID_US', right_on='ID_US')
    df.sort_values(by=['similaridade', 'ID_TC'], ascending=False, inplace=True)

    # Seleciona os k primeiros itens ordenados por similaridade e quantidade de casos de teste
    candidate_uss = df.iloc[:k][['ID_US', 'similaridade']]
    candidate_uss = candidate_uss.merge(tcs, left_on='ID_US', right_on='ID_US')

    result = candidate_uss.groupby('ID_STD_TC')['similaridade'].sum().to_frame()
    result = result.reset_index()

    cts = data_handler.load_test_cases(result['ID_STD_TC'].unique())

    cts = cts.merge(result, left_on='ID', right_on='ID_STD_TC')
    cts.sort_values(by=['similaridade'], ascending=False, inplace=True)

    return cts


def calculate_distance(X, Y, metric='euclidean'):
    if metric == METRIC_EUCLIDEAN: return distance.euclidean(X, Y)
    elif metric == METRIC_JACCARD: return distance.jaccard(X, Y)
    elif metric == METRIC_CANBERRA: return distance.canberra(X, Y)
    elif metric == METRIC_COSINE:
        dot_product = np.dot(X,Y)
        norm_a = np.linalg.norm(X)
        norm_b = np.linalg.norm(Y)
        return dot_product / (norm_a * norm_b)
