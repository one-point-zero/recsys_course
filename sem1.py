"""
Семинар 1. Простые рекомендации
Цель: построить базовые рекомендательные методы и оценить качество
на примере мобильной системы MovieLens.

Задачи:
1) Реализовать случайные рекомендации.
2) Реализовать рекомендации популярных фильмов на основе средних рейтингов.
3) Оценить системы по точности попадания в исторические оценки пользователя.

Для каждого метода требуется реализовать функцию, возвращающую
набор рекомендаций и метрику accuracy.
"""

import numpy as np

from utils import load_data


def random_recommend(n_recommendations: int = 10, seed: int = 42) -> list[int]:
    """
    Рекомендует случайные фильмы.
    """
    ratings_df, _ = load_data()
    np.random.seed(seed)
    
    all_movie_ids = ratings_df['movieId'].unique()
    recommendations = np.random.choice(all_movie_ids, size=n_recommendations, replace=False)
    
    return recommendations.tolist()



def top_n_recommend(
    n_recommendations: int = 10, min_ratings: int = 10
) -> list[tuple[int, float, int, str]]:
    """
    Рекомендует самые популярные фильмы на основе среднего рейтинга и количества оценок.
    """
    ratings_df, movies_df = load_data()
    
    # Группировка по movieId и вычисление статистик
    movie_stats = ratings_df.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        rating_count=('rating', 'count')
    ).reset_index()
    
    # Фильтрация по минимальному количеству оценок
    movie_stats = movie_stats[movie_stats['rating_count'] >= min_ratings]
    
    # Сортировка: сначала по рейтингу (убывание), затем по количеству оценок (убывание)
    movie_stats = movie_stats.sort_values(
        by=['avg_rating', 'rating_count'],
        ascending=[False, False]
    ).head(n_recommendations)
    
    # Формирование результата
    top_n_recs = []
    for _, row in movie_stats.iterrows():
        movie_id = int(row['movieId'])
        title = movies_df[movies_df['movieId'] == movie_id]['title'].iloc[0]
        top_n_recs.append((movie_id, row['avg_rating'], int(row['rating_count']), title))
    
    return top_n_recs



def evaluate_rec_systems(
    user_id: int = 610, n_recommendations: int = 10, random_state: int = 42
) -> dict:
    """
    Оценивает эффективность рекомендательной системы.
    """
    ratings_df, _ = load_data()
    
    # Фильмы, которые пользователь уже оценил
    user_rated_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    
    # Случайные рекомендации (БЕЗ исключения фильмов пользователя!)
    random_recs = random_recommend(n_recommendations=n_recommendations, seed=random_state)
    random_accuracy = len(set(random_recs) & set(user_rated_movies)) / n_recommendations
    
    # Популярные рекомендации
    popular_recs = top_n_recommend(n_recommendations=n_recommendations, min_ratings=10)
    popular_recs_ids = [rec[0] for rec in popular_recs]
    popular_accuracy = len(set(popular_recs_ids) & set(user_rated_movies)) / n_recommendations
    
    return {"random_accuracy": random_accuracy, "popular_accuracy": popular_accuracy}


if __name__ == "__main__":
    # 1. Случайные рекомендации
    print("\n1. СЛУЧАЙНЫЕ РЕКОМЕНДАЦИИ:")
    print("-" * 60)
    random_recs = random_recommend(n_recommendations=10)
    print(f"Рекомендованные ID фильмов: {random_recs}")

    # 2. Популярные фильмы
    print("\n2. ПОПУЛЯРНЫЕ ФИЛЬМЫ (рекомендации на основе популярности):")
    print("-" * 60)
    popular_recs = top_n_recommend(n_recommendations=10)
    print(
        f"{'Rank':<5} {'ID':<6} {'Ср рейтинг':<18} {'Кол-во оценок':<15} {'Название'}"
    )
    print("-" * 60)
    for i, (movie_id, avg_rating, rating_count, title) in enumerate(popular_recs, 1):
        print(
            f"{i:<5} {movie_id:<6} {avg_rating:<18.2f} {rating_count:<15} {title[:50]}"
        )

    # 3. Оценка системы
    print("\n3. ОЦЕНКА КАЧЕСТВА СИСТЕМЫ:")
    print("-" * 60)
    metrics = evaluate_rec_systems()
    print(f"Accuracy (случайные рекомендации): {metrics['random_accuracy']:.4f}")
    print(f"Accuracy (популярные фильмы): {metrics['popular_accuracy']:.4f}")
