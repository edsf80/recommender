from datahandler import NewUSsDataHandler
from recommender import get_recommendations_heuristcs
from recommender import METRIC_EUCLIDEAN


def recommend():
    data_handler = NewUSsDataHandler()
    uss = data_handler.load_us_data()
    uss_x = uss.drop(columns=['TCs'])

    us_alvo = {'ID_US': '123456', 'Módulo': 'Cadastro', 'Operação': 'Recuperação_de_dados', 'Plataforma': 'Web', 'RNFs': '1',
               'CAs': '15,16,17,18,19,20'}



    recommendations = get_recommendations_heuristcs(us_alvo, uss_x, 3, METRIC_EUCLIDEAN)
    print(recommendations.loc[:, ['Descrição', 'similaridade']].to_string())


if __name__ == '__main__':
    recommend()
