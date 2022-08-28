#
import os
import sys
import missingno as mno
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import norm, skew #for some statistics

# data quality   
def data_quality(df, column):  #convert_dtypes_with_reduce_memory(df)
    # datetime
    df[column] = pd.to_datetime(df[column], utc=True, infer_datetime_format=True)
    # any duplicate time periods?
    print("count of duplicates:",df.duplicated(subset=[column], keep="first").sum())
    # any non-numeric types?
    print("non-numeric columns:",list(df.dtypes[df.dtypes == "object"].index))
    
    
    
# any missing values?
def printing_missing_values(df):
    if df.isnull().values.any():
        print("MISSING values:\n")
        mno.matrix(df)
    else:
        print("no missing values\n")
    
    
    
# drop the NaN and zero columns, and also the 'forecast' columns
def data_cleaning(df):    
    df = df.drop(df.filter(regex="forecast").columns, axis=1, errors="ignore")
    df.dropna(axis=1, how="all", inplace=True)
    df = df.loc[:, (df!=0).any(axis=0)]  
    # handle missing values in rows of remaining columns
    df = df.interpolate(method ="bfill")
    if df.isnull().values.any():
        print("MISSING values:\n")
        mno.matrix(df)
    else:
        print("no missing values\n")



# cleaning names for column
def data_cleaning_with_vocabulary(df, en_level_candidate): # clearing 'en_level_candidate':  'no_english' and 'no english'
    df.dropna(axis=0, inplace=True) # my code for deleting last raw
    dict_days = {'upper':'upper', 'intermediate':'intermediate', 'fluent':'fluent','pre':'pre', 'basic':'basic', 'no english':'no english', 'no_english':'no english'}
    df['en_level_candidate'] = df['en_level_candidate'].apply(lambda x: dict_days[x])
    df['en_level_candidate'].unique()

        
        
# add time futures
def add_time_futures(df, column):  
    # datetime
    df[column] = pd.to_datetime(df[column], utc=True, infer_datetime_format=True)
    df.set_index(column, inplace=True)
    df["month"] = df.index.month
    df["wday"] = df.index.dayofweek
    dict_days = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"}
    df["weekday"] = df["wday"].apply(lambda x: dict_days[x])
    df["hour"] = df.index.hour
    df = df.astype({"hour":float, "wday":float, "month": float})
    df.iloc[[0, -1]]



 # boxplots
def printing_boxplot(f32cols):
    for i, c in enumerate(f32cols):
        sns.boxplot(x=df1[c], palette="coolwarm")
        plt.show();   
               
        

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
