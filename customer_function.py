#
import warnings
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import norm, skew #for some statistics

# For DataFrame
def printing_distribution_skewness_kurtosis(df, column):
    # Distribution
    sns.set_style("white")
    sns.set_color_codes(palette='deep')
    f, ax = plt.subplots(figsize=(12, 8))

    # Fit a normal distribution
    mu, std = norm.fit(df[column])

    # Frequency
    sns.distplot(df[column], color="b", fit = stats.norm)
    ax.xaxis.grid(False)
    ax.set(ylabel="Frequency")
    ax.set(xlabel="Currencies")
    ax.set(title="%s distribution: mu = %.2f, std = %.2f" % (column, mu, std))
    sns.despine(trim=True, left=True)

    # Skewness and Kurtosis
    ax.text(x=1.1, y=1, transform=ax.transAxes, s="Skewness: %f" % df[column].skew(),\
    fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right',\
    backgroundcolor='white', color='xkcd:poo brown')
    ax.text(x=1.1, y=0.95, transform=ax.transAxes, s="Kurtosis: %f" % df[column].kurt(),\
    fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right',\
    backgroundcolor='white', color='xkcd:dried blood')

    plt.show()

 # For Array
 def printing_array_distribution(array):
    # Distribution
    sns.set_style("white")
    sns.set_color_codes(palette='deep')
    f, ax = plt.subplots(figsize=(12, 8))

    # Fit a normal distribution
    mu, std = norm.fit(array)

    # Frequency
    sns.distplot(array, color="b", fit = stats.norm)
    ax.xaxis.grid(False)
    ax.set(ylabel="Frequency")
    ax.set(xlabel="Currencies")
    ax.set(title="distribution: mu = %.2f, std = %.2f" % (mu, std))
    sns.despine(trim=True, left=True)

    plt.show()
