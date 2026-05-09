# Flipkart_Sentiment_MLflow
# 🛍️ Flipkart Sentiment Analysis using MLflow

## 📌 Overview

This project performs **Sentiment Analysis on Flipkart product reviews** to classify them as **Positive** or **Negative**.
It follows a complete **Machine Learning pipeline** and integrates **MLflow** for experiment tracking and **Streamlit** for deployment.

The goal is to understand customer feedback and identify pain points in negative reviews.

---

## 🎯 Objective

* Classify customer reviews into:

  * ✅ Positive
  * ❌ Negative
* Analyze customer dissatisfaction patterns
* Track experiments using MLflow
* Deploy model for real-time predictions

---

## 📂 Dataset

* Dataset contains **8,518 Flipkart product reviews**
* Product: *YONEX MAVIS 350 Nylon Shuttle*
* Features include:

  * Reviewer Name
  * Review Title
  * Review Text
  * Ratings
  * Up Votes / Down Votes
  * Place & Date

---

## 🏗️ Project Structure

```
flipkart_sentiment_mlflow/
│
├── data/
│   └── data.csv
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│   ├── best_model.pkl
│   └── vectorizer.pkl
│
├── mlruns/              # MLflow experiment tracking
│
├── app/
│   └── app.py           # Streamlit app
│
├── requirements.txt
├── config.py
└── main.py
```

---

## ⚙️ Machine Learning Pipeline

### 1. Problem Statement

Classify Flipkart product reviews into sentiment categories.

### 2. Data Loading

* Data is loaded using `pandas` from CSV file.

### 3. Data Preprocessing

* Remove missing values
* Clean text (remove special characters, punctuation)
* Convert text to lowercase
* Remove stopwords (keeping words like *not*)
* Apply lemmatization

### 4. Feature Engineering

* TF-IDF Vectorization used to convert text into numerical features

### 5. Label Creation

* Rating ≥ 4 → Positive
* Rating ≤ 2 → Negative
* Rating = 3 → Removed

### 6. Model Selection

* Logistic Regression

### 7. Model Training

* Train multiple models with different hyperparameters

### 8. Hyperparameter Tuning

* Tuned parameter: `C = [0.01, 0.1, 1, 10]`

### 9. Model Evaluation

* Metric: **F1 Score**

### 10. Model Saving

* Best model saved using `joblib`

---

## 📊 MLflow Integration

MLflow is used for:

* 📌 Tracking experiments
* 📌 Logging parameters (`C`)
* 📌 Logging metrics (F1 Score)
* 📌 Saving trained models

### ▶️ Run MLflow UI

```
mlflow ui
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 🚀 How to Run the Project

### 1️⃣ Clone Repository

```
git clone <https://github.com/jidnyasadthakre07/flipkart-sentiment-mlflow>
cd flipkart_sentiment_mlflow
```

---

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

### 3️⃣ Train Model

```
python main.py
```

✔ This will:

* Train models
* Track experiments in MLflow
* Save best model

---

### 4️⃣ Run Streamlit App

```
streamlit run app/app.py
```

Open:

```
http://localhost:8501
```

---

## 🌐 Deployment (Hugging Face)

This project is deployed using **Streamlit on Hugging Face Spaces**.

### Steps:

1. Create a new Space (Streamlit)
2. Upload:

   * `app.py`
   * `models/`
   * `requirements.txt`
3. App will be live automatically

---

## 🧪 Example Predictions

| Input Review              | Output   |
| ------------------------- | -------- |
| "This product is amazing" | Positive |
| "Worst product ever"      | Negative |

---

## 🧠 Key Learnings

* End-to-end ML pipeline implementation
* Text preprocessing & NLP techniques
* Feature extraction using TF-IDF
* Model evaluation using F1-score
* Experiment tracking using MLflow
* Deployment using Streamlit

---

## 🔮 Future Improvements

* Use advanced models (BERT, LSTM)
* Add sentiment confidence score
* Perform error analysis dashboard
* Deploy using AWS EC2

---

## 📌 Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* NLTK
* MLflow
* Streamlit

---

## 👩‍💻 Author

**Jidnyasa Thakre**
- 🔗 LinkedIn: www.linkedin.com/in/kusuma-kumari-bodduluri
- 💻 GitHub: https://github.com/kusuma990/Flipkart_Sentiment_MLflow
- 🤗 Huggingface: https://huggingface.co/spaces/Kusuma990/Flipkart_Sentiment_Anlysis_MLflow

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share it!
