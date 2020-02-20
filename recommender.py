from datahandler import NewUSsDataHandler
from sklearn.neighbors import NearestNeighbors
import pandas as pd


def get_recommendations(new_us, base_uss, k, distance_metric='euclidean'):
    data_handler = NewUSsDataHandler()

    # Adicionando um filtro para trazer apenas uss com mesmo módulo e operação da nova US.
    base_uss_filtrada = base_uss.loc[(base_uss['Módulo'] == new_us['Módulo']) & (base_uss['Operação'] == new_us['Operação']), :] # base_uss.copy()

    if base_uss_filtrada.empty:
        return pd.DataFrame()

    df = data_handler.load_filtered_us_data(new_us, base_uss_filtrada)

    # Nesse ponto verifico que a base filtrada é menor que o K, pois não é possível calcular a distância com K maior
    # que quantidade de itens.
    if df.shape[0] < k:
        return pd.DataFrame()

    knn = NearestNeighbors(metric=distance_metric, algorithm='brute', n_neighbors=k).fit(df)

    # Pega a quantidade de colunas para gerar uma lista de 1.
    l = [1 for i in range(len(df.columns))]

    distances, indices = knn.kneighbors([l])
    candidate_uss = df.iloc[indices[0]]
    candidate_uss.loc[:, 'similaridade'] = 1 / (1 + distances[0])

    test_data = data_handler.load_test_data(candidate_uss.index.values)
    test_data = test_data.merge(candidate_uss, left_on='ID_US', right_on='ID_US')

    results = test_data.groupby("ID_STD_TC").sum()["similaridade"] / k
    results = results.to_frame()

    cts = data_handler.load_test_cases(results.index.values)

    cts = cts.merge(results, left_on="ID", right_on="ID_STD_TC", how='inner')
    return cts.sort_values("similaridade", ascending=False)
