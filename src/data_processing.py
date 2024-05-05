import pandas as pd
import numpy as np
from src.utils import preprocess_ingredients

def generate_user_profiles(recipes_df, num_users=100):
    # Preprocess the ingredients column
    recipes_df['processed_ingredients'] = recipes_df['ingredients'].apply(preprocess_ingredients)

    # Generate synthetic user data
    user_ids = range(1, num_users + 1)
    user_data = []

    for user_id in user_ids:
        # Randomly select liked recipes (up to 5)
        num_likes = np.random.randint(1, 6)
        liked_recipes = np.random.choice(recipes_df['sno'], size=num_likes, replace=False)

        # Randomly select past searches (up to 5)
        num_searches = np.random.randint(1, 6)
        past_searches = np.random.choice(recipes_df['sno'], size=num_searches, replace=False)

        user_data.append({
            'user_id': user_id,
            'likes': liked_recipes,
            'past_searches': past_searches
        })

    # Create a DataFrame from the synthetic user data
    user_profiles_df = pd.DataFrame(user_data)

    # Extract liked ingredients and past search ingredients for each user
    user_profiles_df['liked_ingredients'] = user_profiles_df['likes'].apply(lambda likes: recipes_df.loc[recipes_df['sno'].isin(likes), 'processed_ingredients'].tolist())
    user_profiles_df['searched_ingredients'] = user_profiles_df['past_searches'].apply(lambda searches: recipes_df.loc[recipes_df['sno'].isin(searches), 'processed_ingredients'].tolist())

    # Merge ingredients into a single list for each user
    user_profiles_df['all_ingredients'] = user_profiles_df.apply(lambda row: row['liked_ingredients'] + row['searched_ingredients'], axis=1)

    return user_profiles_df