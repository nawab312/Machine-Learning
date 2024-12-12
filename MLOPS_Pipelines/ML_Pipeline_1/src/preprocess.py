#preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data.dyopna(inplace=True)  # Remove missing values
    X = data.drop("target", axis=1)
    y = data["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data("data/dataset.csv")
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)
