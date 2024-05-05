import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from src.utils import preprocess_ingredients

def prepare_data(recipes_df):
    # Combine ingredients and titles into a single text for each recipe
    recipes_df['processed_text'] = recipes_df['processed_ingredients'] + ' ' + recipes_df['title']

    # Create a TF-IDF Vectorizer with normalization
    vectorizer = make_pipeline(TfidfVectorizer(stop_words='english'), Normalizer())

    # Transform the processed text column into TF-IDF features
    tfidf_matrix = vectorizer.fit_transform(recipes_df['processed_text'])

    # Compute the cosine similarity between recipes based on ingredients and titles
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    return vectorizer, tfidf_matrix, cosine_sim

def get_ingredient_recommendations(ingredients, recipes_df, vectorizer, tfidf_matrix, num_recommendations=5):
    # Preprocess user input ingredients
    processed_user_ingredients = preprocess_ingredients(ingredients)

    # Transform user input ingredients into TF-IDF features
    user_tfidf = vectorizer.transform([processed_user_ingredients])

    # Calculate cosine similarities between user input and all recipes
    user_cosine_similarities = linear_kernel(user_tfidf, tfidf_matrix).flatten()

    # Get indices of recipes sorted by similarity
    recipe_indices = user_cosine_similarities.argsort()[::-1]

    # Get the top N similar recipes
    top_recipes = recipe_indices[1:num_recommendations + 1]

    return recipes_df['title'].iloc[top_recipes]

def get_user_and_ingredient_recommendations(ingredients, user_id, recipes_df, vectorizer, tfidf_matrix, cosine_sim, num_recommendations=5):
    # Preprocess user input ingredients
    processed_user_ingredients = preprocess_ingredients(ingredients)

    # Transform user input ingredients into TF-IDF features
    user_tfidf = vectorizer.transform([processed_user_ingredients])

    # Calculate cosine similarities between user input and all recipes
    user_cosine_similarities = linear_kernel(user_tfidf, tfidf_matrix).flatten()

    # Combine cosine similarities based on user preferences and user input
    combined_cosine_similarities = user_cosine_similarities + cosine_sim[user_id]

    # Get indices of recipes sorted by combined similarity
    recipe_indices = combined_cosine_similarities.argsort()[::-1]

    # Get the top N similar recipes
    top_recipes = recipe_indices[1:num_recommendations + 1]

    return recipes_df['title'].iloc[top_recipes]

def get_user_recommendations(user_id, recipes_df, cosine_sim, num_recommendations=5):
    # Get indices of recipes sorted by user preferences
    recipe_indices = cosine_sim[user_id].argsort()[::-1]

    # Get the top N similar recipes
    top_recipes = recipe_indices[1:num_recommendations + 1]

    return recipes_df['title'].iloc[top_recipes]