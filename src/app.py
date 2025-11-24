import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import linear_kernel

# --- CONFIGURAÇÃO DA PÁGINA (DEVE SER O PRIMEIRO COMANDO STREAMLIT) ---
st.set_page_config(
    page_title="Sistema de Recomendação por Conteúdo",
    layout="centered",
    initial_sidebar_state="expanded"
)


# --- FUNÇÃO DE CARREGAMENTO ---
@st.cache_resource
def load_model_artifacts():
    """
    Loads all necessary models and data only once.
    Returns a tuple containing: movies_df, cosine_sim, indices.
    """

    artifact_path = './model_artifacts/'

    try:
        # 1. Load DataFrames and Matrices
        movies_df_loaded = pd.read_csv(artifact_path + 'movies_for_app.csv')
        cosine_sim_loaded = np.load(artifact_path + 'cosine_sim_matrix_compressed.npz')['matrix']
        indices_loaded = pd.read_pickle(artifact_path + 'indices.pickle')

        st.success("Model artifacts loaded successfully!")

        return movies_df_loaded, cosine_sim_loaded, indices_loaded

    except FileNotFoundError as e:
        st.error(f"Error: Artifact file not found: {e}. Check the folder '{artifact_path}'.")
        return None, None, None
    except Exception as e:
        st.error(f"Unknown error loading artifacts: {e}")
        return None, None, None


# --- FUNÇÃO DE RECOMENDAÇÃO BASEADA EM CONTEÚDO ---
def get_content_recommendations(
        user_id: int,
        n: int,
        content_weight: float,
        movies_df: pd.DataFrame,
        cosine_sim: np.ndarray,
        indices: pd.Series,
        movie_ids_rated: list,
        favorite_movie_ids: list
):
    """
    Generates recommendations based on content similarity only.
    """

    # --- 1. Get Average Content Similarity (Favorite Movies) ---
    sim_scores_favorite_avg = None

    if favorite_movie_ids:
        favorite_titles = movies_df[movies_df['movieId'].isin(favorite_movie_ids)]['title_clean']
        favorite_indices = [indices[title] for title in favorite_titles if title in indices]

        if favorite_indices:
            sim_vectors = cosine_sim[favorite_indices]
            sim_scores_favorite_avg = np.mean(sim_vectors, axis=0)

    # --- 2. Generate Recommendations based on Content Similarity ---
    if sim_scores_favorite_avg is not None:
        similar_indices = np.argsort(sim_scores_favorite_avg)[::-1]

        recommendations = []
        for idx in similar_indices:
            movie_id = movies_df.iloc[idx]['movieId']
            if movie_id not in movie_ids_rated and len(recommendations) < n:
                recommendations.append(movie_id)

        result_df = movies_df[movies_df['movieId'].isin(recommendations)][['movieId', 'title_clean', 'genres']]

        # Add similarity scores for display
        if len(result_df) > 0:
            score_map = {movie_id: sim_scores_favorite_avg[idx] for idx, movie_id in enumerate(movies_df['movieId'])
                         if movie_id in recommendations}
            result_df['similarity_score'] = result_df['movieId'].map(score_map)
            result_df['similarity_score'] = result_df['similarity_score'].round(4)
            result_df = result_df.sort_values('similarity_score', ascending=False)

        return result_df
    else:
        available_movies = movies_df[~movies_df['movieId'].isin(movie_ids_rated)]
        result_df = available_movies.sample(min(n, len(available_movies)))[['movieId', 'title_clean', 'genres']]
        result_df['similarity_score'] = "N/A"
        return result_df


# --- INTERFACE PRINCIPAL ---

# Load artifacts
artifacts = load_model_artifacts()

if artifacts[0] is not None:
    movies_df, cosine_sim, indices = artifacts

    st.title("Sistema de Recomendação por Conteúdo 🎬")
    st.markdown("Recomendações baseadas em similaridade de conteúdo dos seus filmes favoritos.")

    # SIDEBAR
    st.sidebar.header("Configurações de Recomendação")

    user_id = st.sidebar.number_input(
        'Digite seu ID de Usuário:',
        min_value=1,
        max_value=1000,
        value=50,
        step=1,
        help="ID do usuário para personalização"
    )

    # Mock data
    MOCKED_RATED_MOVIES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70, 80, 90, 100, 110, 120]
    MOCKED_FAVORITE_MOVIES = [1, 50, 110, 120]

    st.sidebar.markdown('---')
    st.sidebar.markdown('**Filmes de Exemplo:**')
    st.sidebar.markdown(f"• Assistidos: {len(MOCKED_RATED_MOVIES)} filmes")
    st.sidebar.markdown(f"• Favoritos: {len(MOCKED_FAVORITE_MOVIES)} filmes")

    # Show favorite movies
    if len(MOCKED_FAVORITE_MOVIES) > 0:
        favorite_movies_info = movies_df[movies_df['movieId'].isin(MOCKED_FAVORITE_MOVIES)]['title_clean'].tolist()
        st.sidebar.markdown("**Seus filmes favoritos:**")
        for movie in favorite_movies_info[:3]:
            st.sidebar.markdown(f"• {movie}")

    weight = st.sidebar.slider(
        'Intensidade da Similaridade:',
        0.0,
        1.0,
        0.7,
        0.05,
        help="Controla o quanto a similaridade com seus filmes favoritos influencia as recomendações"
    )

    n_recs = st.sidebar.slider(
        'Número de Recomendações:',
        1,
        30,
        10,
        1
    )

    st.sidebar.markdown('---')
    st.sidebar.markdown('**Informações do Sistema:**')
    st.sidebar.markdown(f"• Total de filmes: {len(movies_df)}")
    st.sidebar.markdown(f"• Matriz de similaridade: {cosine_sim.shape}")

    if st.button('Gerar Recomendações', type='primary'):
        with st.spinner(f"Gerando {n_recs} recomendações baseadas em seus filmes favoritos..."):
            recommendations = get_content_recommendations(
                user_id,
                n_recs,
                weight,
                movies_df,
                cosine_sim,
                indices,
                movie_ids_rated=MOCKED_RATED_MOVIES,
                favorite_movie_ids=MOCKED_FAVORITE_MOVIES
            )

            st.subheader(f"Top {n_recs} Recomendações para o Usuário {user_id}")

            if recommendations.empty:
                st.info("Nenhuma recomendação encontrada ou você já assistiu a todos os filmes disponíveis.")
            else:
                display_df = recommendations.rename(columns={
                    'movieId': 'ID do Filme',
                    'title_clean': 'Título',
                    'genres': 'Gêneros',
                    'similarity_score': 'Score de Similaridade'
                })

                st.dataframe(display_df, use_container_width=True)
                st.success(f"✅ {len(recommendations)} recomendações geradas com sucesso!")

else:
    st.error("Não foi possível carregar os artefatos do modelo.")

    st.info("""
    **Arquivos necessários na pasta model_artifacts/:**
    - movies_for_app.csv
    - cosine_sim_matrix.npy
    - indices.pickle

    **Para corrigir:**
    1. Verifique se o arquivo 'cosine_sim_matrix.ppy' foi renomeado para 'cosine_sim_matrix.npy'
    2. Certifique-se de que todos os arquivos estão na pasta correta
    """)