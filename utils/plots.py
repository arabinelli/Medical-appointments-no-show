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
    if order is None:
        values = list(df[column].unique())
        values.sort()
    else:
        values = order
    labels = [True,False]
    n_values = len(values)
    sns.set_palette("pastel")
    colors = sns.color_palette("pastel", 2)
    colors.reverse()

    if n_values == 2:
        ncols = 2
    else:
        occurrences_count = df[[column,target_var]]\
                                .groupby(column)[target_var]\
                                .agg([np.mean])\
                                .rename(columns={"mean":"percentageNoShow"})\
                                .reset_index()
        #values = occurrences_count\
        #            .sort_values(by="percentageNoShow",ascending=False)[column].tolist()
    
    nrows = math.ceil(n_values/ncols) + 1
    fig = plt.figure(constrained_layout=True,figsize=(14,5*nrows))
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols, left=0.05, right=0.48, wspace=0.05)
    
    
    occurrences_count = df[[column,target_var,dataset_key]]\
                        .groupby([column,target_var]).count() 

    
    fig_ax0 = fig.add_subplot(gs[0,:])
    fig_ax0.set_title("Distribution for the column " + column, fontsize=14, fontweight='bold')
    if order is None: 
        g = sns.countplot(x=df[column], hue=df[column],ax=fig_ax0,dodge=False)
    else:
        g = sns.countplot(x=df[column], hue=df[column],ax=fig_ax0,order=values,dodge=False)
    #g.set_xticklabels(g.get_xticklabels(),rotation=45)
    
    row_id = 0
    col_id = 0
    for i in range(n_values):
        size = [occurrences_count.loc[(values[i],labels[0])][0],
                occurrences_count.loc[(values[i],labels[1])][0]]
        if np.isnan(size).any():
            continue
        fig_ax = fig.add_subplot(gs[row_id+1,col_id])
        fig_ax.set_title("Show vs. No Show for " + column + " = " + str(values[i]))
        fig_ax.pie(size,
                    colors=colors, 
                    labels=labels, 
                    autopct='%1.1f%%', 
                    startangle=90)
        row_id, col_id = increment_row_col_count(row_id,col_id,ncols)
    
    plt.show()

    if n_values == 2:
        hypothesis_test_binary_distribution(a = df[target_var].loc[df[column] == values[0]].astype(int),
                                            b= df[target_var].loc[df[column] == values[1]].astype(int),
                                            column_name = column,
                                            values = values)


def increment_row_col_count(row_id,col_id,ncols):
    """
    Increments the row and col id, return a tuple with the updated values
    """
    if col_id == ncols - 1:
        return row_id +1, 0
    else:
        return row_id, col_id + 1