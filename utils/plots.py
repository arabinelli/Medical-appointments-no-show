import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.hypothesis_tests import hypothesis_test_binary_distribution

def plot_categorical_distribution(df: pd.DataFrame, column: str, target_var="noShow",
                                    dataset_key="appointmentId", ncols = 3, order = None) -> None:
    """
    Plot a set of graphs in a grid containing a value count and pie charts for the
    split based on the target variable for each value in the distribution

    ARGUMENTS:
    df (pd.DataFrame) - dataset to be analyzed
    column (str) - the name of the column to be analyzed
    target_var (str - defaults to "noShow") - column name of the target variable in the dataset
    dataset_key (str - defaults to "appointmentId") - column name of the unique key in the dataset
    ncols (int - defaults to 3) - number of columns    
    """
    # define the order in which the elements are plotted
    if order is None:
        values = list(df[column].unique())
        values.sort()
    else:
        values = order

    n_values = len(values)

    if n_values == 2:
        ncols = 2
    else:
        # delete?
        occurrences_count = df[[column,target_var]]\
                                .groupby(column)[target_var]\
                                .agg([np.mean])\
                                .rename(columns={"mean":"percentageNoShow"})\
                                .reset_index()
    
    nrows = 1 # math.ceil(n_values/ncols) + 1
    fig = plt.figure(constrained_layout=True,figsize=(17,5*nrows))
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols, left=0.05, right=0.48, wspace=0.05)
    
    
    occurrences_count = df[[column,target_var,dataset_key]]\
                            .groupby([column])[target_var].agg([np.mean])\
                            .rename(columns={"mean":"percentageNoShow"})\
                            .reset_index()

    # plotting variable distribution incidence
    fig_ax0 = fig.add_subplot(gs[0,0])
    fig_ax0.set_title("Distribution for the column " + column)
    g = sns.countplot(x=df[column], hue=df[column],ax=fig_ax0,order=values,dodge=False)
    sns.despine()
    fig_ax0.get_legend().remove()
    # add values labels
    for p in fig_ax0.patches:
        height = p.get_height()
        if not np.isnan(height):
            fig_ax0.text(p.get_x()+p.get_width()/2.,
                    height*0.5,
                    int(height),
                    ha="center",
                    fontsize=14)
    
    # plotting no-show incidence
    fig_ax = fig.add_subplot(gs[0,1])
    fig_ax.set_title("No-show incidence for " + column)
    g1 = sns.barplot(x=column, y="percentageNoShow",data=occurrences_count,
                       ax=fig_ax,dodge=False, order=values)
    sns.despine()

    # add values labels
    for p in fig_ax.patches:
        height = p.get_height()
        fig_ax.text(p.get_x()+p.get_width()/2.,
                height*0.5,
                format_percentage(height),
                ha="center",
                fontsize=14) 
    plt.show()

    # if only two values are possible, perform an hypothesis test on the two distributuons
    if n_values == 2:
        hypothesis_test_binary_distribution(a = df[target_var].loc[df[column] == values[0]].astype(int),
                                            b= df[target_var].loc[df[column] == values[1]].astype(int),
                                            column_name = column,
                                            values = values)



def format_percentage(num:float, n_digits:int = 2) -> str:
    """
    Takes a float and returns it as a string formatted as a percentage with 2 decimals
    """
    num *= (10**(2+n_digits))
    num = int(num)
    num /= 10**n_digits
    return str(num) + "%"