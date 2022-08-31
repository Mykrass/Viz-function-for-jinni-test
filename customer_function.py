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

# Plots
from plotly.offline import iplot
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
    dict_days = {'upper':'upper', 'intermediate':'intermediate', 'fluent':'fluent','pre':'pre', 'basic':'basic', 'no_english':'no english'}
    df[en_level_candidate] = df[en_level_candidate].apply(lambda x: dict_days[x])
    print('Unique values:', df[en_level_candidate].unique())

        
        
# add time futures
def add_time_futures(df, column):  
    # datetime
    df[column] = pd.to_datetime(df[column], utc=True, infer_datetime_format=True)
    df.set_index(column, inplace=True)
    df.sort_index(inplace=True)
    df["month"] = df.index.month
    df["wday"] = df.index.dayofweek
    dict_days = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"}
    df["weekday"] = df["wday"].apply(lambda x: dict_days[x])
    df["hour"] = df.index.hour
    df = df.astype({"hour":float, "wday":float, "month": float})
    print("earliest time period:", df.index.min())
    print("latest time period:", df.index.max())
    

    
# convert int and float64 columns to float32
def convert_dtypes_with_reduce_memory(df): 
    intcols = list(df.dtypes[df.dtypes == np.int64].index)
    df[intcols] = df[intcols].applymap(np.float32)

    f64cols = list(df.dtypes[df.dtypes == np.float64].index)
    df[f64cols] = df[f64cols].applymap(np.float32)

    f32cols = list(df.dtypes[df.dtypes == np.float32].index)
    
    df.info()



 # boxplots
def printing_boxplot(df):
    f32cols = list(df.dtypes[df.dtypes == np.float32].index)
    for i, c in enumerate(f32cols):
        sns.boxplot(x=df[c], palette="coolwarm")
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
    ax.set(xlabel=column)
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
    
    

# Creating cohort    
def create_cohort(df, start_date, end_date):
    cohort = df[(df.index >=start_date) & (df.index <= end_date)].copy()
    #cohort.reset_index(inplace=True, drop=True)
    return(cohort)



# Helper functions for structured data
## Get info about the dataset
def dataset_info(dataset, dataset_name: str):
    print(f"Dataset Name: {dataset_name} \
        | Number of Samples: {dataset.shape[0]} \
        | Number of Columns: {dataset.shape[1]}")
    print(30*"=")
    print("Column             Data Type")
    print(dataset.dtypes)
    print(30*"=")
    missing_data = dataset.isnull().sum()
    if sum(missing_data) > 0:
        print(missing_data[missing_data.values > 0])
    else:
        print("No Missing Data on this Dataset!")
    print(30*"=")
    print("Memory Usage: {} MB".\
         format(np.round(
         dataset.memory_usage(index=True).sum() / 10e5, 3
         )))
## Dataset Sampling
def data_sampling(dataset, frac: float, random_seed: int):
    data_sampled_a = dataset.sample(frac=frac, random_state=random_seed)
    data_sampled_b =  dataset.drop(data_sampled_a.index).reset_index(drop=True)
    data_sampled_a.reset_index(drop=True, inplace=True)
    return data_sampled_a, data_sampled_b   
## Bar Plot
def bar_plot(data, plot_title: str, x_axis: str, y_axis: str):
    colors = ["#0080ff",] * len(data)
    colors[0] = "#ff8000"
    trace = go.Bar(y=data.values, x=data.index, text=data.values, 
                    marker_color=colors)
    layout = go.Layout(autosize=False, height=600,
                    title={"text" : plot_title,
                       "y" : 0.9,
                       "x" : 0.5,
                       "xanchor" : "center",
                       "yanchor" : "top"},  
                    xaxis={"title" : x_axis},
                    yaxis={"title" : y_axis},)
    fig = go.Figure(data=trace, layout=layout)
    fig.update_layout(template="simple_white")
    fig.update_traces(textposition="outside",
                    textfont_size=14,
                    marker=dict(line=dict(color="#000000", width=2)))                
    fig.update_yaxes(automargin=True)
    iplot(fig)
## Plot Pie Chart
def pie_plot(data, plot_title: str):
    trace = go.Pie(labels=data.index, values=data.values)
    layout = go.Layout(autosize=False,
                    title={"text" : plot_title,
                       "y" : 0.9,
                       "x" : 0.5,
                       "xanchor" : "center",
                       "yanchor" : "top"})
    fig = go.Figure(data=trace, layout=layout)
    fig.update_traces(textfont_size=14,
                    marker=dict(line=dict(color="#000000", width=2)))
    fig.update_yaxes(automargin=True)            
    iplot(fig)
## Histogram
def histogram_plot(data, plot_title: str, y_axis: str):
    trace = go.Histogram(x=data)
    layout = go.Layout(autosize=False,
                    title={"text" : plot_title,
                       "y" : 0.9,
                       "x" : 0.5,
                       "xanchor" : "center",
                       "yanchor" : "top"},  
                    yaxis={"title" : y_axis})
    fig = go.Figure(data=trace, layout=layout)
    fig.update_traces(marker=dict(line=dict(color="#000000", width=2)))
    fig.update_layout(template="simple_white")
    fig.update_yaxes(automargin=True)
    iplot(fig)
# Particular case: Histogram subplot (1, 2)
def histogram_subplot(dataset_a, dataset_b, feature_a: str,
                        feature_b: str, title: str, title_a: str, title_b: str):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
                        title_a,
                        title_b
                        )
                    )
    fig.add_trace(go.Histogram(x=dataset_a[feature_a],
                               showlegend=False),
                                row=1, col=1)
    fig.add_trace(go.Histogram(x=dataset_b[feature_b],
                               showlegend=False),
                              row=1, col=2)
    fig.update_layout(template="simple_white")
    fig.update_layout(autosize=False,
                        title={"text" : title,
                        "y" : 0.9,
                        "x" : 0.5,
                        "xanchor" : "center",
                        "yanchor" : "top"},  
                        yaxis={"title" : "<i>Frequency</i>"})
    fig.update_traces(marker=dict(line=dict(color="#000000", width=2)))
    fig.update_yaxes(automargin=True)
    iplot(fig)
# Calculate scores with Test/Unseen labeled data
def test_score_report(data_unseen, predict_unseen):
    le = LabelEncoder()
    data_unseen["Label"] = le.fit_transform(data_unseen.Churn.values)
    data_unseen["Label"] = data_unseen["Label"].astype(int)
    accuracy = accuracy_score(data_unseen["Label"], predict_unseen["Label"])
    roc_auc = roc_auc_score(data_unseen["Label"], predict_unseen["Label"])
    precision = precision_score(data_unseen["Label"], predict_unseen["Label"])
    recall = recall_score(data_unseen["Label"], predict_unseen["Label"])
    f1 = f1_score(data_unseen["Label"], predict_unseen["Label"])

    df_unseen = pd.DataFrame({
        "Accuracy" : [accuracy],
        "AUC" : [roc_auc],
        "Recall" : [recall],
        "Precision" : [precision],
        "F1 Score" : [f1]
    })
    return df_unseen

# Confusion Matrix
def conf_mat(data_unseen, predict_unseen):
    unique_label = data_unseen["Label"].unique()
    cmtx = pd.DataFrame(
        confusion_matrix(data_unseen["Label"],
                         predict_unseen["Label"],
                         labels=unique_label), 
        index=['{:}'.format(x) for x in unique_label], 
        columns=['{:}'.format(x) for x in unique_label]
    )
    ax = sns.heatmap(cmtx, annot=True, fmt="d", cmap="YlGnBu")
    ax.set_ylabel('Predicted')
    ax.set_xlabel('Target');
    ax.set_title("Predict Unseen Confusion Matrix", size=14);
