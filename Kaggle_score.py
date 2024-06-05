from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

# read in data
df = pd.read_csv('train.csv')

# Prétraitement des données
X = df.drop('isSold', axis=1)
y = df['isSold']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['timeStamp']
categorical_features = ['auctionId', 'placementId', 'websiteId', 'hashedRefererDeepThree', 'country', 'opeartingSystem', 'browser', 'browserVersion', 'device', 'environmentType', 'integrationType', 'articleSafenessCategorization']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Entraînement du modèle
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression())])

model.fit(X_train, y_train)

# Évaluation du modèle
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred)

print("F1-score:", f1)
