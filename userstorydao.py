from database import connect

def getUserStories():
    conexao = None

    try:
        conexao = connect()

        # create a cursor
        cur = conexao.cursor()

        # Feature matrix [cadastro, gerencial, autenticacao, recuperacao de dados, atualizar dados, inserir dados, modificar insercao de dados, remover dados]

        cur.execute(
            'select us.id_ust, us.dsc_ust, us.id_opr, us.id_mod, a.id_act, a.dsc_act ' +
            'from user_story us ' +
            '  inner join user_story_acceptance_test usat on usat.id_ust = us.id_ust ' +
            '  inner join acceptance_test a on a.id_act = usat.id_act')
        user_stories = []
        id_us_atual = 0
        user_story = {}
        for row in cur:
            if row[0] != id_us_atual:
                if len(user_story) > 0:
                    user_stories.append(user_story)
                id_us_atual = row[0]
                user_story = {}
                user_story['id'] = id_us_atual
                user_story['features'] = [1, 0, 0]
                if row[2] == 1:
                    user_story['features'].extend([1, 0, 0, 0, 0])
                elif row[2] == 2:
                    user_story['features'].extend([0, 1, 0, 0, 0])
                elif row[2] == 3:
                    user_story['features'].extend([0, 0, 1, 0, 0])
                user_story.setdefault('testcases', [])
                user_story['testcases'].append({'id': row[4], 'desc': row[5]})
            else:
                user_story['testcases'].append({'id': row[4], 'desc': row[5]})

        return user_stories
    except (Exception) as error:
        print(error)
    finally:
        if conexao is not None:
            conexao.close()
            print('Database connection closed.')