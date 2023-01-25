from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle

dataset = pd.read_csv('./heart.csv')

features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'thalach']

X = dataset[features]
Y = dataset['target']

model = LogisticRegression()
model.fit(X, Y)


pickle.dump(model, open('heart_disease.pkl', 'wb'))