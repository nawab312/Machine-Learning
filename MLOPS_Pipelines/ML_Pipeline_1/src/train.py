#train.py
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_model(X_train_path, y_train_path):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    # Create Model Directory if it doesn't exists
    if not os.path.exists("model"):
        os.makedirs("model")
        
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    with open("model/random_forest.pkl", "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train_model("data/X_train.csv", "data/y_train.csv")

