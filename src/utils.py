import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK stopwords and WordNet resources
nltk.download('stopwords')
nltk.download('wordnet')

# Function to clean ingredient text
def clean_ingredient(ingredient):
    # Remove quantities and measurements
    ingredient = re.sub(r'\b\d+\s*(tsp|tbsp|g|kg|oz|ml|cup|cups|grams|pounds)\b', '', ingredient, flags=re.IGNORECASE)
    # Remove common words
    common_words = ['ingredient', 'optional', 'vegan', 'non-dairy', 'free', 'gluten-free', 'organic']
    for word in common_words:
        ingredient = ingredient.replace(word, '')
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    # Remove numbers and words indicating quantities
    ingredient = ' '.join([word for word in ingredient.split() if not word.isdigit() and word.lower() not in stop_words])
    return ingredient.strip()

# Function to preprocess ingredients
def preprocess_ingredients(ingredients):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(ingredients.lower())
    cleaned_ingredients = [clean_ingredient(word) for word in tokens]
    processed_text = ' '.join([lemmatizer.lemmatize(word) for word in cleaned_ingredients])
    return processed_text