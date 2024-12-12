#evaluate.py
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

def evaluate_model(model_path, X_test_path, y_test_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)

if __name__ == "__main__":
    accuracy = evaluate_model("model/random_forest.pkl", "data/X_test.csv", "data/y_test.csv")
    print(f"Model Accuracy: {accuracy:.2f}")
