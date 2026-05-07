"""
Семинар 3. Контентная фильтрация
Цель: Разработать методы контентной фильтрации по пользователям и по фильмам.
В качестве контента используем описание жанров для каждого фильма из movies.csv.
Для векторизации жанров используем CountVectorizer с разделителем "|".
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from utils import build_user_item_matrix, id_to_movie, load_data, print_user_rated_items


class ContentRecommender:
    """
    Класс для построения рекомендаций на основе контента - описания жанров.
    Матрица эмбеддингов размером (max_movie_id+1, n_genres), где строки
    соответствуют movieId, а столбцы — one-hot кодированию жанров.
    Матрица строится при инициализации экземпляра класса.
    """

    def __init__(self):
        self.embeddings = None
        self.ui_matrix = build_user_item_matrix()
        self._build_embeddings()

    def _build_embeddings(self):
        _, movies_df = load_data()
        self.movies_df = movies_df.copy()
        self.movies_df["genres"] = self.movies_df["genres"].fillna("")
        vectorizer = CountVectorizer(tokenizer=lambda s: s.split("|"), lowercase=False)
        genre_matrix = vectorizer.fit_transform(self.movies_df["genres"])
        max_movie_id = self.movies_df["movieId"].max()
        n_features = genre_matrix.shape[1]
        self.embeddings = np.zeros((max_movie_id + 1, n_features))
        for idx, row in self.movies_df.iterrows():
            movie_id = row["movieId"]
            self.embeddings[movie_id] = genre_matrix[idx].toarray().flatten()
        

    def predict_rating(self, user_id: int, item_id: int, k: int = 5) -> float:
        """
        Предсказывает рейтинг user_id для item_id на основе контентной фильтрации.

        Алгоритм:
        1) Берём вектор целевого фильма: target_vec.
        2) Находим все фильмы, оцененные пользователем.
        3) Считаем косинусное сходство target_vec с векторами оцененных фильмов.
        4) Отбираем топ-k похожих оцененных фильмов (k-параметр).
        5) Предсказываем рейтинг как взвешенное среднее оценок по сходствам.
        6) Если не удаётся предсказать (нет оценок или нулевые векторы), возвращаем 0.0.
        7) Клипируем результат в [0.0, 5.0].

        Args:
            user_id: индекс пользователя
            item_id: индекс фильма
            k: сколько наиболее похожих оцененных фильмов использовать

        Returns:
            float: предсказанный рейтинг
        """
        target_vec = self.embeddings[item_id]
        user_ratings = self.ui_matrix[user_id]
        rated_items = np.where(user_ratings > 0)[0]
        if len(rated_items) == 0:
            return 0.0
        similarities = []
        for movie_id in rated_items:
            vec = self.embeddings[movie_id]
            norm_target = np.linalg.norm(target_vec)
            norm_vec = np.linalg.norm(vec)
            if norm_target == 0 or norm_vec == 0:
                sim = 0.0
            else:
                sim = np.dot(target_vec, vec) / (norm_target * norm_vec)
            similarities.append(sim)
        similarities = np.array(similarities)
        rated_items = rated_items[np.argsort(similarities)[::-1]]
        similarities = np.sort(similarities)[::-1]
        k = min(k, len(rated_items))
        top_items = rated_items[:k]
        top_sims = similarities[:k]
        if top_sims.sum() == 0:
            return 0.0
        ratings = np.array([user_ratings[m] for m in top_items])
        pred = np.average(ratings, weights=top_sims)
        return float(np.clip(pred, 0.0, 5.0))

    def predict_items_for_user(
        self, user_id: int, k: int = 5, n_recommendations: int = 5
    ) -> list:
        """
        Рекомендует фильмы пользователю user_id на основе контента фильма.

        Алгоритм:
        1) Берем все фильмы, которые оценил пользователь.
        3) Строим профиль пользователя как взвешенное среднее жанров оцененных фильмов.
        4) Для всех фильмов, которые пользователь не оценил, считаем сходство с профилем.
        5) Сортируем по убыванию сходства и возвращаем top-n.
        """
        user_ratings = self.ui_matrix[user_id]
        rated_items = np.where(user_ratings > 0)[0]
        if len(rated_items) == 0:
            return []
        profile = np.zeros(self.embeddings.shape[1])
        total_weight = 0
        for movie_id in rated_items:
            vec = self.embeddings[movie_id]
            rating = user_ratings[movie_id]
            profile += vec * rating
            total_weight += rating
        if total_weight == 0:
            return []
        profile = profile / total_weight
        norm_profile = np.linalg.norm(profile)
        if norm_profile == 0:
            return []
        all_similarities = []
        for movie_id in range(1, self.embeddings.shape[0]):
            if user_ratings[movie_id] > 0:
                continue
            vec = self.embeddings[movie_id]
            norm_vec = np.linalg.norm(vec)
            if norm_vec == 0:
                sim = 0.0
            else:
                sim = np.dot(profile, vec) / (norm_profile * norm_vec)
            all_similarities.append((sim, movie_id))
        all_similarities.sort(key=lambda x: x[0], reverse=True)
        recommendations = [int(m) for _, m in all_similarities[:n_recommendations]]
        return recommendations


# Пример использования для дебага:
if __name__ == "__main__":
    user_id = 10
    item_id = 2
    k = 5
    content_recommender = ContentRecommender()
    print_user_rated_items(user_id, content_recommender.ui_matrix)

    pred_rating = content_recommender.predict_rating(user_id, item_id, k)
    print(f"Predicted rating for user {user_id} and item {item_id}: {pred_rating:.2f}")

    recommendations = content_recommender.predict_items_for_user(
        user_id, k=5, n_recommendations=10
    )
    for rec in recommendations:
        print(f"Recommended movie ID: {rec}, Title: {id_to_movie(rec)}")
