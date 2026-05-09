import streamlit as st
import joblib

# Load model
model = joblib.load("models/best_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

st.set_page_config(page_title="Flipkart Sentiment", layout="centered")

st.title("🛍️ Flipkart Sentiment Analysis")
st.write("Enter a product review to predict sentiment")

review = st.text_area("Enter your review:")

if st.button("Predict"):

    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        vec = vectorizer.transform([review])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.success("Positive 😊")
        else:
            st.error("Negative 😡")