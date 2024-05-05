import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


def draw_distribution_plots(df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(12,18))
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
    # Ref: https://seaborn.pydata.org/generated/seaborn.kdeplot.html
    sns.kdeplot(data=df, x="Sales", ax=axs[2, 0], fill=True, palette="crest", alpha=.5, linewidth=0)
    sns.kdeplot(data=df, x="Customers", ax=axs[2, 1], fill=True, palette="crest", alpha=.5, linewidth=0)

    axs[2, 0].axvline(df["Sales"].mean(), color="red")
    axs[2, 0].axvline(df["Sales"].median(), color="green")

    axs[2, 1].axvline(df["Customers"].mean(), color="red")
    axs[2, 1].axvline(df["Customers"].median(), color="green")

    # Fourth row
    # Ref: https://seaborn.pydata.org/generated/seaborn.kdeplot.html
    sns.kdeplot(data=df, x="Sales", ax=axs[3, 0], hue="DayOfWeek", fill=True, palette="crest", alpha=.5, linewidth=0)
    sns.kdeplot(data=df, x="Customers", ax=axs[3, 1], hue="DayOfWeek", fill=True, palette="crest", alpha=.5, linewidth=0)
