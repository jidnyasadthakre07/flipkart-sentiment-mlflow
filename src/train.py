import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config import *
from src.evaluate import evaluate_model

def train_model(df):

    X = df["clean_review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    vectorizer = TfidfVectorizer(max_features=5000)

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    mlflow.set_experiment("Flipkart_Sentiment")

    best_score = 0
    best_model = None

    for C in [0.01, 0.1, 1, 10]:

        with mlflow.start_run(run_name=f"C_{C}"):

            model = LogisticRegression(max_iter=1000, C=C)
            model.fit(X_train, y_train)

            score = evaluate_model(model, X_test, y_test)

            print(f"C={C} → F1 Score: {score}")

            mlflow.log_param("C", C)
            mlflow.log_metric("f1_score", score)
            mlflow.sklearn.log_model(model, name="model")

            if score > best_score:
                best_score = score
                best_model = model

    os.makedirs("models", exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("Best Score:", best_score)