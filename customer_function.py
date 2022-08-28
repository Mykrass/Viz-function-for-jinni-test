#
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import norm, skew #for some statistics

# Printing parameters for Statistical distribution
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
    ax.set(xlabel=df[column])
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
    
    
# pivot table: weekdays in months
def printing_pivot_heatmap(df, values, index, columns): #printing_pivot_heatmap(df, "hire_salary", "month", "candidates_city")
    piv = pd.pivot_table(   df, 
                            values= values, 
                            index=index, 
                            columns=columns, 
                            aggfunc="mean", 
                            margins=True, margins_name="Avg", 
                            fill_value=0)
    pd.options.display.float_format = '{:,.0f}'.format

    plt.figure(figsize = (20, 10))
    sns.set(font_scale=1)
    sns.heatmap(piv.round(0), annot=True, square = True, \
                linewidths=.75, cmap="coolwarm", fmt = ".0f", annot_kws = {"size": 11})
    plt.title("hire_salary by candidates_city by month")
    plt.show()
    
    
    
# reduce data memory    
def convert_dtypes_with_reduce_memory(df):  #convert_dtypes_with_reduce_memory(df)
    # convert int and float64 columns to float32
    intcols = list(df.dtypes[df.dtypes == np.int64].index)
    df[intcols] = df[intcols].applymap(np.float32)

    f64cols = list(df.dtypes[df.dtypes == np.float64].index)
    df[f64cols] = df[f64cols].applymap(np.float32)

    f32cols = list(df.dtypes[df.dtypes == np.float32].index)
    
    
 # boxplots
def printing_boxplot(f32cols):
    for i, c in enumerate(f32cols):
        sns.boxplot(x=df1[c], palette="coolwarm")
        plt.show();   
    
