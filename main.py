import pandas as pd
import joblib
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

MODEL_PATH = "anxiety_model.joblib"
DATA_PATH = "enhanced_anxiety_dataset.csv"

app = FastAPI(title="Anxiety Level Prediction API")


class InputData(BaseModel):
    features: List[float]


def train_and_save_model():
    df = pd.read_csv(DATA_PATH)

    # Целевая переменная
    y = df["Anxiety Level (1-10)"]

    # Удаление цели из фич
    X = df.drop("Anxiety Level (1-10)", axis=1)

    # Обработка категориальных признаков
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
        remainder="passthrough"
    )

    # Модель
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=200))
    ])

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    print("Model trained and saved.")


def load_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()
    return joblib.load(MODEL_PATH)


model = load_model()


@app.post("/predict")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)[0]
    return {"prediction": prediction}
