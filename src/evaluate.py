from sklearn.metrics import f1_score

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return f1_score(y_test, preds)