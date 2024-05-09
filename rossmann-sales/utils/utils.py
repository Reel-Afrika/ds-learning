import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats



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


def draw_distribution_plots(df: pd.DataFrame, use_hue: bool = True, figsize: tuple = (12,18)) -> None:
    """Draw distribution plots for Sales and Customers columns"""

    if use_hue:
        fig, axs = plt.subplots(nrows=4, ncols=2, figsize=figsize)
    else:
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=figsize)
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


class OutlierHandler:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df_copy = df.copy()

    def generate_z_scores(self, column: str) -> pd.DataFrame:
        self.df_copy[f"{column}_z_score"] = stats.zscore(self.df_copy[column])

        return self.df_copy

    def drop_z_score_outliers(self, column: str, threshold: int = 3) -> pd.DataFrame:
        z_col = f"{column}_z_score"

        outliers = self.df_copy[np.abs(self.df_copy[z_col]) > threshold]

        return self.df_copy.drop(outliers.index)

    def drop_outliers_using_iqr(self, column: str, threshold: float = 1.5):
        q1 = self.df_copy[column].quantile(0.25)
        q3 = self.df_copy[column].quantile(0.75)
        iqr = q3 - q1

        outliers = self.df_copy[(self.df_copy[column] < q1 - threshold * iqr) | (self.df_copy[column] > q3 + threshold * iqr)]

        return self.df_copy.drop(outliers.index)

    def impute_z_score_outliers_with_median(self, column: str, threshold: int = 3):
        z_col = f"{column}_z_score"

        self.df_copy.loc[np.abs(self.df_copy[z_col]) > threshold, column] = self.df_copy[column].median()

        return self.df_copy

    def remove_outliers(self, columns: tuple = ("Customers", "Sales")) -> pd.DataFrame:
        for column in columns:
            self.generate_z_scores(column=column)

        for column in columns:
            self.df_copy = self.drop_z_score_outliers(column=column)

        for column in columns:
            self.df_copy = self.drop_outliers_using_iqr(column=column)

        draw_distribution_plots(df=self.df_copy)

        return self.df_copy
