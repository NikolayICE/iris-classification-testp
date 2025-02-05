import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data_preprocessing import load_data
from utils import save_figure, logger


def perform_eda() -> None:
    logger.info("Начало EDA...")
    df: pd.DataFrame = load_data()

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.histplot(df['sepal length (cm)'], kde=True, ax=ax1)
    ax1.set_title('Распределение длины чашелистика')
    ax1.set_xlabel('Длина чашелистика (см)')
    ax1.set_ylabel('Частота')
    save_figure(fig1, "sepal_length_distribution.png")
    plt.show()

    fig2 = sns.pairplot(df, hue='target', palette='Set2')
    fig2.fig.suptitle("Парная диаграмма признаков", y=1.02)
    fig2.savefig("../plots/pairplot.png")
    logger.info("EDA завершено.")


if __name__ == "__main__":
    perform_eda()
