import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import QuantileTransformer



def draw_boxplot(df: pd.DataFrame, x: str, ax) -> None:
    """Ref: https://seaborn.pydata.org/generated/seaborn.boxplot.html"""
    sns.boxplot(data=df, x=x, ax=ax)


def draw_violinplot(df: pd.DataFrame, x: str, ax) -> None:
    """Ref: https://seaborn.pydata.org/generated/seaborn.violinplot.html"""
    sns.violinplot(
        data=df,
        x=x,
        ax=ax,
        native_scale=True,
        linewidth=1,
        linecolor="k",
        inner_kws=dict(box_width=12, whis_width=2, color=".8"))


def draw_kdeplot(df: pd.DataFrame, x: str, ax, hue: str = "") -> None:
    """Ref: https://seaborn.pydata.org/generated/seaborn.kdeplot.html"""
    if hue:
        sns.kdeplot(data=df, x=x, ax=ax, fill=True, hue=hue, palette="crest", alpha=.5, linewidth=0)
    else:
        sns.kdeplot(data=df, x=x, ax=ax, fill=True, palette="crest", alpha=.5, linewidth=0)

    ax.axvline(df[x].mean(), color="red")
    ax.axvline(df[x].median(), color="green")


def draw_distribution_plots(df: pd.DataFrame, use_hue: bool = True) -> None:
    """Draw distribution plots for Sales and Customers columns"""

    if use_hue:
        fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(12,18))
    else:
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12,18))
    fig.tight_layout(pad=5.0)

    # Create plots
    # First Row
    axs[0, 0].set_title("Boxplot - Sales")
    axs[0, 1].set_title("Boxplot - Customers")

    # Second Row
    axs[1, 0].set_title("Violinplot - Sales")
    axs[1, 1].set_title("Violinplot - Customers")

    # Third Row
    axs[2, 0].set_title("KDE Plot - Sales")
    axs[2, 1].set_title("KDE Plot - Customers")

    if use_hue:
        # Fourth Row
        axs[3, 0].set_title("KDE Plot w/ DayOfWeek mapping - Sales")
        axs[3, 1].set_title("KDE Plot w/ DayOfWeek mapping - Customers")

    # Draw plots
    # First row
    draw_boxplot(df=df, x="Sales", ax=axs[0, 0])
    draw_boxplot(df=df, x="Customers", ax=axs[0, 1])

    # Second row
    draw_violinplot(df=df, x="Sales", ax=axs[1, 0])
    draw_violinplot(df=df, x="Customers", ax=axs[1, 1])

    # Third row
    draw_kdeplot(df=df, x="Sales", ax=axs[2, 0])
    draw_kdeplot(df=df, x="Customers", ax=axs[2, 1])

    if use_hue:
        # Fourth Row
        draw_kdeplot(df=df, x="Sales", ax=axs[3, 0], hue="DayOfWeek")
        draw_kdeplot(df=df, x="Customers", ax=axs[3, 1], hue="DayOfWeek")


class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bound = []
        self.upper_bound = []

    def outlier_detector(self, X, y=None):
        X = pd.Series(X).copy()
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1

        self.lower_bound.append(q1 - (self.factor * iqr))
        self.upper_bound.append(q3 + (self.factor * iqr))

    def fit(self, X, y=None):
        X.apply(self.outlier_detector)
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for i in range(X.shape[1]):
            x = X.iloc[:, i].copy()
            x[(x < self.lower_bound[i]) | (x > self.upper_bound[i])] = np.nan
            X.iloc[:, i] = x
        return X


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    X = df[['Customers', 'Sales']]

    outlier_remover = QuantileTransformer(output_distribution="normal").fit_transform(X)

    data_without_outliers = pd.DataFrame(outlier_remover, columns=X.columns)

    outlier_remover = OutlierRemover()

    ct = ColumnTransformer(transformers=[['outlier_remover', OutlierRemover(), list(range(data_without_outliers.shape[1]))]], remainder='passthrough')

    data_without_outliers = pd.DataFrame(ct.fit_transform(data_without_outliers), columns=data_without_outliers.columns)

    draw_distribution_plots(df=data_without_outliers, use_hue=False)

    return data_without_outliers
