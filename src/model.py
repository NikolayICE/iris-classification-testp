
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from data_preprocessing import load_data
from utils import logger


def train_model() -> None:
    logger.info("Загрузка данных...")
    df: pd.DataFrame = load_data()
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=200))
    ])

    param_grid = {
        "clf__C": [0.1, 1, 10]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy")
    logger.info("Начало подбора гиперпараметров...")
    grid_search.fit(X_train, y_train)

    logger.info(f"Лучшая комбинация параметров: {grid_search.best_params_}")

    y_pred = grid_search.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy на тестовой выборке: {acc:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    model_path = "../models/best_logistic_regression_model.pkl"
    joblib.dump(grid_search.best_estimator_, model_path)
    logger.info(f"Модель сохранена по адресу {model_path}")


if __name__ == "__main__":
    train_model()
