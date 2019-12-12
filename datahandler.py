import pandas as pd
from abc import ABC, abstractmethod


class DataHandler(ABC):

    @abstractmethod
    def load_filtered_us_data(self, newUS):
        pass

    @abstractmethod
    def load_test_data(self, user_stories):
        pass

    @abstractmethod
    def load_test_cases(self, id_test_cases):
        pass


class NewUSsDataHandler(DataHandler):

    def load_filtered_us_data(self, newUS):
        # Arquivo que contém as user stories
        uss = pd.read_csv('data/newUSs.csv')
        tcs = pd.read_csv('data/USsTCs.csv')
        # Filtra apenas as User Stories que possuem casos de teste
        uss = uss.loc[uss['ID_US'].isin(tcs['ID_US']), ['ID_US', 'DESC_US', 'Módulo', 'Operação', 'Plataforma']]
        # Arquivo que contém os critérios de aceitação
        acs = pd.read_csv('data/USsCAs.csv')
        # Merge entre os dois dataframes com base no id da User Story
        ussacs = pd.merge(uss, acs, how='inner', on='ID_US')
        # Criação de um aggregator padrão para juntar os ids dos casos de teste padrão em uma só coluna.
        foo = lambda a: ",".join(a)
        # Transformando a coluna de id do critério de aceitação padrão em string para poder ser agregado.
        ussacs['ID_STD_AC'] = ussacs['ID_STD_AC'].astype(str)
        # Aqui é feito o agrupamento de todos os critérios de uma user story em apenas uma coluna.
        filtered_uss = ussacs.groupby(by=['ID_US', 'Módulo', 'Operação', 'Plataforma', 'RNFs']).agg(
            CAs=('ID_STD_AC', foo)).reset_index()
        # Separando os ids dos critérios de aceitação da us nova para cálculo do vetor de características.
        newUSACs = newUS['CAs'].split(',')
        # Separando os ids dos critérios de aceitação da us nova para cálculo do vetor de características.
        newUSRNFs = newUS['RNFs'].split(',')

        # Cria uma coluna para cada CA da nova US. Na matriz as que tiverem vão ficar com 1 e se não tiverem com 0.
        for i in range(len(newUSACs)):
            filtered_uss['CA_' + str(i)] = 0
        # Cria uma coluna para cada RNF da nova US. Na matriz as que tiverem vão ficar com 1 e se não tiverem com 0.
        for i in range(len(newUSRNFs)):
            filtered_uss['RNF_' + str(i)] = 0
        # Atualiza matriz com o módulo, operação, CAs e RNFs para 1 se forem iguais a da nova US ou 0 se forem
        # diferentes.
        filtered_uss.loc[filtered_uss['Módulo'] != newUS['Módulo'], 'Módulo'] = 0
        filtered_uss.loc[filtered_uss['Módulo'] == newUS['Módulo'], 'Módulo'] = 1
        filtered_uss.loc[filtered_uss['Operação'] != newUS['Operação'], 'Operação'] = 0
        filtered_uss.loc[filtered_uss['Operação'] == newUS['Operação'], 'Operação'] = 1
        filtered_uss.loc[filtered_uss['Plataforma'] != newUS['Plataforma'], 'Plataforma'] = 0
        filtered_uss.loc[filtered_uss['Plataforma'] == newUS['Plataforma'], 'Plataforma'] = 1

        for i, ca in enumerate(newUSACs):
            filtered_uss.loc[filtered_uss['CAs'].str.contains(ca), 'CA_' + str(i)] = 1
        for i, rnf in enumerate(newUSRNFs):
            filtered_uss.loc[filtered_uss['RNFs'].str.contains(rnf), 'RNF_' + str(i)] = 1

        filtered_uss.drop(columns=['RNFs', 'CAs'], inplace=True)
        filtered_uss.set_index('ID_US', append=False, inplace=True)

        return filtered_uss

    def load_test_data(self, user_stories):
        data = pd.read_csv('data/USsTCs.csv')

        new = pd.DataFrame(user_stories)

        # Retorna os IDs dos casos de teste de acordo com a user story passada.
        return data.loc[data['ID_US'].isin(new[0]) & data['ID_STD_TC'].notnull(), ['ID_US', 'ID_STD_TC']]

    def load_test_cases(self, id_test_cases):
        cts = pd.read_csv('tcs.csv')

        new = pd.DataFrame(id_test_cases)

        return cts.loc[cts['ID'].isin(new[0])]


class USsDataHandler(DataHandler):

    def load_filtered_us_data(self, newUS):
        uss = pd.read_csv('https://raw.githubusercontent.com/edsf80/recommender/master/uss.csv')
        data = pd.read_csv('https://raw.githubusercontent.com/edsf80/recommender/master/data.csv')

        filtered_uss = uss.loc[
            uss['Módulo'].notnull() & (uss['Módulo'] != 'X') & (uss['Tipo'] == 'Negocio') & uss['ID_US'].isin(
                data['ID_US']), ['ID_US', 'Módulo', 'Operação', 'Mapeado? (SIM ou NAO)']].drop_duplicates()

        # Das US's cadastradas, aqui são separadas apenas as que possuem dados de casos de teste.
        filtered_uss.loc[filtered_uss['Módulo'] != newUS['Módulo'], 'Módulo'] = 0
        filtered_uss.loc[filtered_uss['Módulo'] == newUS['Módulo'], 'Módulo'] = 1
        filtered_uss.loc[filtered_uss['Operação'] != newUS['Operação'], 'Operação'] = 0
        filtered_uss.loc[filtered_uss['Operação'] == newUS['Operação'], 'Operação'] = 1
        filtered_uss['Mapeado? (SIM ou NAO)'] = 1

        filtered_uss.set_index('ID_US', append=False, inplace=True)

        return filtered_uss

    def load_test_data(self, user_stories):
        data = pd.read_csv('https://raw.githubusercontent.com/edsf80/recommender/master/data.csv')

        new = pd.DataFrame(user_stories)

        # Retorna os IDs dos casos de teste de acordo com a user story passada.
        return data.loc[data['ID_US'].isin(new[0]) & data['ID_REC_TC'].notnull(), ['ID_US', 'ID_REC_TC']]

    def load_test_cases(self, id_test_cases):
        cts = pd.read_csv('https://raw.githubusercontent.com/edsf80/recommender/master/cts.csv')

        new = pd.DataFrame(id_test_cases)

        return cts.loc[cts['ID'].isin(new[0])]


class MovieLensDataHandler(DataHandler):

    def load_filtered_us_data(self, newUS):
        uss = pd.read_csv('https://raw.githubusercontent.com/khanhnamle1994/movielens/master/users.csv', sep='\t')

        filtered_uss = uss.loc[:, ['user_id', 'gender', 'occupation', 'zipcode', 'age_desc']]

        filtered_uss.rename(index=str, columns={"user_id": "ID_US"}, inplace=True)

        filtered_uss.loc[filtered_uss['gender'] != newUS['gender'], 'gender'] = 0
        filtered_uss.loc[filtered_uss['gender'] == newUS['gender'], 'gender'] = 1
        filtered_uss.loc[filtered_uss['occupation'] != newUS['occupation'], 'occupation'] = 0
        filtered_uss.loc[filtered_uss['occupation'] == newUS['occupation'], 'occupation'] = 1
        filtered_uss.loc[filtered_uss['zipcode'] != newUS['zipcode'], 'zipcode'] = 0
        filtered_uss.loc[filtered_uss['zipcode'] == newUS['zipcode'], 'zipcode'] = 1
        filtered_uss.loc[filtered_uss['age_desc'] != newUS['age_desc'], 'age_desc'] = 0
        filtered_uss.loc[filtered_uss['age_desc'] == newUS['age_desc'], 'age_desc'] = 1

        filtered_uss.set_index('ID_US', append=False, inplace=True)

        return filtered_uss

    def load_test_data(self, user_stories):
        data = pd.read_csv('https://raw.githubusercontent.com/khanhnamle1994/movielens/master/ratings.csv', sep='\t')

        new = pd.DataFrame(user_stories)

        data.rename(index=str, columns={'user_id': 'ID_US', 'movie_id': 'ID_REC_TC'}, inplace=True)

        # Retorna os IDs dos casos de teste de acordo com a user story passada.
        return data.loc[data['ID_US'].isin(new[0]) & data['ID_REC_TC'].notnull(), ['ID_US', 'ID_REC_TC']]

    def load_test_cases(self, id_test_cases):
        cts = pd.read_csv('https://raw.githubusercontent.com/khanhnamle1994/movielens/master/movies.csv', sep='\t',
                          encoding='latin-1')

        new = pd.DataFrame(id_test_cases)

        cts.rename(index=str, columns={'movie_id': 'ID'}, inplace=True)

        return cts.loc[cts['ID'].isin(new[0])]
