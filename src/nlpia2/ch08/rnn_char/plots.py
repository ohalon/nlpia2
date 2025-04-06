""" Confusion matrix heatmap
From Dennis Trimarchi
https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py
"""
import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def learning_curve(results, filepath='learning_curve.png', dpi=300, linewidth=2, grid='on'):
    """ plot a curve training and val loss or accuracy from log of losses recorded in results """

    loss_names, acc_names = [], []
    for k in results.keys():
        if 'loss' in k:
            loss_names.append(k)
        if 'acc' in k:
            acc_names.append(k)
    learning_df = pd.DataFrame({k: results[k] for k in acc_names})
    learning_df[acc_names].plot(linewidth=linewidth)
    plt.grid(grid)


def dataset_confusion(df,
                      normalize=True, fillna='0', text_col='surname', target='nationality',
                      ):
    """ Given a df with columns name & category, assume "truth" is most popular category for a name """
    confusion = {c: Counter() for c in sorted(df[target].unique())}
    for i, g in df.groupby(text_col):
        counts = Counter(g[target])
        confusion[counts.most_common()[0][0]] += counts
    confusion = pd.DataFrame(confusion)
    confusion = confusion[confusion.index]
    if normalize:
        confusion /= confusion.sum(axis=1)
    if fillna is not None:
        confusion.fillna(fillna, inplace=True)
    confusion.index.name = 'most_common'
    return confusion


def prediction_confusion(df, pred_col=1, truth_col=0, categories=None, normalize=True, fillna=None):
    """ Given a df with columns name & category, assume "truth" is most popular category for a name """
    truth_col = truth_col if isinstance(truth_col, str) else list(df.columns)[truth_col]
    pred_col = pred_col if isinstance(pred_col, str) else list(df.columns)[pred_col]
    categories = sorted(df[truth_col].unique()) if categories is None else categories
    confusion = {c: Counter() for c in categories}
    for i, g in df.groupby(truth_col):
        counts = Counter(g[pred_col])
        confusion[i] = counts
    confusion = pd.DataFrame(confusion)
    confusion = confusion[confusion.index]
    if normalize:
        confusion /= confusion.sum(axis=1)
    if fillna is not None:
        confusion.fillna(fillna, inplace=True)
    confusion.index.name = truth_col
    confusion.columns.name = pred_col
    return confusion


def plot_heatmap(df, filepath='heatmap.png', dpi=300):
    values = df.values.astype(float)
    values = [[x for x in row] for row in values]

    annot = [
        [
            f"{str(round(float(x), 2)) or ''}"
            for x in row
        ]
        for row in values
    ]
    fig = sns.heatmap(
        df, cmap='Blues', fmt="",
        annot=annot, xticklabels='auto', yticklabels='auto'
    )
    plt.savefig(filepath, dpi=dpi)
    return fig


def confusion_heatmap(df, group_names=None,
                      categories='auto',
                      count=True,
                      percent=True,
                      cbar=True,
                      xyticks=True,
                      xyplotlabels=True,
                      sum_stats=True,
                      figsize=None,
                      cmap='Blues',
                      title=None):
    """
    Heatmap of an sklearn Confusion Matrix

    Inputs:
        df:            confusion matrix to be passed in
        group_names:   List of strings that represent the labels row by row to be shown in each square.
        categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
        count:         If True, show the raw number in the confusion matrix. Default is True.
        normalize:     If True, show the proportions for each category. Default is True.
        cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                       Default is True.
        xyticks:       If True, show x and y ticks. Default is True.
        xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
        sum_stats:     If True, display summary statistics below the figure. Default is True.
        figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
        cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                       See http://matplotlib.org/examples/color/colormaps_reference.html

        title:         Title for the heatmap. Default is None.
    """

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(df.size)]

    if group_names and len(group_names) == df.size:
        group_labels = [f"{value}\n" for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = [f"{value:0.0f}\n" for value in df.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = [f"{x:.2%}".format(x) for x in df.flatten() / np.sum(df)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(df.shape[0], df.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(df) / float(np.sum(df))

        # if it is a binary confusion matrix, show some more stats
        if len(df) == 2:
            # Metrics for Binary Confusion Matrices
            precision = df[1, 1] / sum(df[:, 1])
            recall = df[1, 1] / sum(df[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize is None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if not xyticks:
        categories = False  # don't label xyticks if there are no ticks!

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(df, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)
