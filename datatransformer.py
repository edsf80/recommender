
def transform_one_hot_encoding(newUS, baseUSs):
    # Separando os ids dos critérios de aceitação da us nova para cálculo do vetor de características.
    filtered_uss = baseUSs.copy()
    newUSACs = newUS['CAs'].split(',')
    # Separando os ids dos critérios de aceitação da us nova para cálculo do vetor de características.
    newUSRNFs = newUS['RNFs'].split(',')
    # Cria uma coluna para cada CA da nova US. Na matriz as que tiverem vão ficar com 1 e se não tiverem com 0.
    for i in range(len(newUSACs)):
        filtered_uss.loc[:, 'CA_' + str(i)] = 0
    # Cria uma coluna para cada RNF da nova US. Na matriz as que tiverem vão ficar com 1 e se não tiverem com 0.
    for i in range(len(newUSRNFs)):
        filtered_uss.loc[:, 'RNF_' + str(i)] = 0
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