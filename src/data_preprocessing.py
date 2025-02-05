
import pandas as pd
from sklearn.datasets import load_iris
from typing import Union


def load_data() -> pd.DataFrame:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df


if __name__ == "__main__":
    data = load_data()
    print(data.head())
