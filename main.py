import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def carregar_dados():
    filmes = pd.read_csv('movies_metadata.csv', low_memory=False)
    avaliacoes = pd.read_csv('ratings.csv')
    return filmes, avaliacoes


def preprocessar_dados(filmes, avaliacoes):
    # Seleção e renomeação de colunas
    film = filmes[['id', 'original_title', 'original_language', 'vote_count']].copy()
    film = film.rename(columns={
        'id': 'ID_FILME',
        'original_title': 'TITULO',
        'original_language': 'LINGUAGEM',
        'vote_count': 'QT_AVALIACOES'
    })

    av = avaliacoes[['userId', 'movieId', 'rating']].copy()
    av = av.rename(columns={
        'userId': 'ID_USUARIO',
        'movieId': 'ID_FILME',
        'rating': 'AVALIACAO'
    })

    # Limpeza
    film.dropna(inplace=True)

    # Conversão segura de ID_FILME
    film['ID_FILME'] = pd.to_numeric(film['ID_FILME'], errors='coerce')
    film.dropna(subset=['ID_FILME'], inplace=True)
    film['ID_FILME'] = film['ID_FILME'].astype(int)

    # Mantém apenas usuários muito ativos
    usuarios_ativos = av['ID_USUARIO'].value_counts()
    usuarios_ativos = usuarios_ativos[usuarios_ativos > 900].index
    av = av[av['ID_USUARIO'].isin(usuarios_ativos)]

    # Mantém filmes populares e em inglês
    film = film[film['QT_AVALIACOES'] > 900]
    film = film[film['LINGUAGEM'] == 'en']

    # Junta avaliações com metadados dos filmes
    avaliacoes_e_filmes = av.merge(film, on='ID_FILME')

    # Remove avaliações duplicadas do mesmo usuário para o mesmo filme
    avaliacoes_e_filmes.drop_duplicates(['ID_USUARIO', 'ID_FILME'], inplace=True)

    return avaliacoes_e_filmes


def criar_matriz(avaliacoes_e_filmes):
    filmes_pivot = avaliacoes_e_filmes.pivot_table(
        columns='ID_USUARIO',
        index='TITULO',
        values='AVALIACAO'
    )

    filmes_pivot.fillna(0, inplace=True)
    return filmes_pivot


def treinar_modelo(filmes_pivot):
    filmes_sparse = csr_matrix(filmes_pivot.values)
    modelo = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)
    modelo.fit(filmes_sparse)
    return modelo


def recomendar_filmes(titulo, filmes_pivot, modelo, n_recomendacoes=5):
    if titulo not in filmes_pivot.index:
        print(f'Filme "{titulo}" não encontrado na base.')
        return

    dados_filme = filmes_pivot.loc[titulo].values.reshape(1, -1)
    distances, indices = modelo.kneighbors(dados_filme, n_neighbors=n_recomendacoes + 1)

    print(f'\nRecomendações para: {titulo}\n')
    for i in range(1, len(indices[0])):
        indice = indices[0][i]
        distancia = distances[0][i]
        similaridade = 1 - distancia
        print(
            f'{filmes_pivot.index[indice]} | '
            f'Distância: {distancia:.4f} | '
            f'Similaridade: {similaridade:.4f}'
        )


def main():
    filmes, avaliacoes = carregar_dados()
    avaliacoes_e_filmes = preprocessar_dados(filmes, avaliacoes)
    filmes_pivot = criar_matriz(avaliacoes_e_filmes)
    modelo = treinar_modelo(filmes_pivot)

    recomendar_filmes('Toy Story', filmes_pivot, modelo)


if __name__ == '__main__':
    main()