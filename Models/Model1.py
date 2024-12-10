import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

#Load Datasets
data = load_iris(as_frame=True)
X, y = data.frame.iloc[:, :-1], data.frame.iloc[:, -1]

#Train Model
model = RandomForestClassifier()
model.fit(X, y)

print("Model Trained Successfully")
