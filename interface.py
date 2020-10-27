from datahandler import NewUSsDataHandler
from recommender import get_recommendations_heuristcs


def recommend():
    data_handler = NewUSsDataHandler()
    uss = data_handler.load_us_data()
    uss_x = uss.drop(columns=['TCs'])

    us_alvo = {'ID_US': '123456', 'Módulo': 'Cadastro', 'Operação': 'Inserir_dados', 'Plataforma': 'Desktop', 'RNFs': '1,2',
               'CAs': '1,3,4,5,38,39'}

    recommendations = get_recommendations_heuristcs(us_alvo, uss_x, 3, 'euclidean')
    print(recommendations.loc[:, ['Descrição', 'similaridade']].to_string())


if __name__ == '__main__':
    recommend()
