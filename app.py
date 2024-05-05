import streamlit as st
import pandas as pd
from src.data_processing import generate_user_profiles
from src.recommendations import get_ingredient_recommendations, get_user_and_ingredient_recommendations, get_user_recommendations, prepare_data

# Load the dataset
recipes_df = pd.read_csv('data/vegan_recipes.csv')

# Generate user profiles
user_profiles_df = generate_user_profiles(recipes_df, num_users=100)

# Prepare the data
vectorizer, tfidf_matrix, cosine_sim = prepare_data(recipes_df)

# Streamlit app
def main():
    st.title("Vegan Recipe Recommender")

    # Get user input
    user_id = st.number_input("Enter your user ID (optional)", value=0, min_value=0, max_value=len(user_profiles_df), step=1)
    ingredients = st.text_input("Enter ingredients (separated by commas)")

    if st.button("Get Recommendations"):
        # Display user profile
        if user_id:
            st.subheader(f"User Profile (ID: {user_id})")
            user_profile = user_profiles_df.loc[user_profiles_df['user_id'] == user_id, ['likes', 'past_searches']].squeeze()
            st.markdown(f"Liked Recipes: {', '.join(recipes_df.loc[recipes_df['sno'].isin(user_profile['likes']), 'title'].tolist())}")
            st.write(f"Past Searches: {', '.join(recipes_df.loc[recipes_df['sno'].isin(user_profile['past_searches']), 'title'].tolist())}")

        # Generate recommendations
        if ingredients:
            if user_id:
                recommendations = get_user_and_ingredient_recommendations(ingredients, user_id, recipes_df, vectorizer, tfidf_matrix, cosine_sim)
                st.subheader("Recommendations based on user preferences and input ingredients:")
            else:
                recommendations = get_ingredient_recommendations(ingredients, recipes_df, vectorizer, tfidf_matrix)
                st.subheader("Recommendations based on input ingredients:")
        elif user_id:
            recommendations = get_user_recommendations(user_id, recipes_df, cosine_sim)
            st.subheader("Recommendations based on user preferences:")
        else:
            st.warning("Please enter either a user ID or ingredients to get recommendations.")
            return

        if recommendations.size > 0:
            st.markdown("- " + "\n- ".join(recommendations))
        else:
            st.warning("No recommendations found.")

if __name__ == "__main__":
    main()