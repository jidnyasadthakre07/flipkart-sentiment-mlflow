import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from config import REVIEW_COLUMN, RATING_COLUMN

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english')) - {"not", "no", "nor"}
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

def preprocess_data(df):

    df = df.dropna(subset=[REVIEW_COLUMN])
    df = df[df[RATING_COLUMN] != 3]

    df["sentiment"] = df[RATING_COLUMN].apply(lambda x: 1 if x >= 4 else 0)
    df["clean_review"] = df[REVIEW_COLUMN].apply(clean_text)

    return df