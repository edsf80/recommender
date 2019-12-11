import pandas as pd

newUS = {'Módulo': 'Cadastro', 'Operação': 'Recuperação de dados', 'Plataforma': 'Web', 'Bugs Low': 0, 'Bugs Medium': 0,
         'Bugs High': 0, 'CAs': '43,44,11,12', 'RNFs': '1,2'}
uss = pd.read_csv('newUSs.csv')
acs = pd.read_csv('CAs.csv')
ussacs = pd.merge(uss, acs, how='left', on='ID_US')
print(uss)
print(newUS)
print(ussacs)

# cas = newUS['CAs'].split(',')
# rnfs = newUS['RNFs'].split(',')
#
# filtered_uss = uss.loc[:,
#                ['ID_US', 'Módulo', 'Operação', 'Plataforma', 'Bugs Low', 'Bugs Medium', 'Bugs High', 'CAs', 'RNFs']]
# for i in range(len(cas)):
#     filtered_uss['CA_' + str(i)] = 0
# for i in range(len(rnfs)):
#     filtered_uss['RNF_' + str(i)] = 0
# filtered_uss.loc[filtered_uss['Módulo'] != newUS['Módulo'], 'Módulo'] = 0
# filtered_uss.loc[filtered_uss['Módulo'] == newUS['Módulo'], 'Módulo'] = 1
# filtered_uss.loc[filtered_uss['Operação'] != newUS['Operação'], 'Operação'] = 0
# filtered_uss.loc[filtered_uss['Operação'] == newUS['Operação'], 'Operação'] = 1
# filtered_uss.loc[filtered_uss['Plataforma'] != newUS['Plataforma'], 'Plataforma'] = 0
# filtered_uss.loc[filtered_uss['Plataforma'] == newUS['Plataforma'], 'Plataforma'] = 1
#
# for i, ca in enumerate(cas):
#     print(ca, i)
#     filtered_uss.loc[filtered_uss['CAs'].str.contains(ca), 'CA_' + str(i)] = 1
#
# print(filtered_uss)
