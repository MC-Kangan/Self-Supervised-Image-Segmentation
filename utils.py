import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import json

def preprocess_results(result: Dict[str, List[float]]) -> Dict[str, List[float]]:
    result['BCE'] = (np.array(result['batch_loss']).reshape(60, -1).mean(axis = 1), result['BCE'])
    result.pop('batch_loss')
    return result

def plot_metrics(eval_metrics: List[float], train_metrics: Optional[List[float]] = None, metric_name: str = "Loss", eval_epochs: Optional[List[int]] = None,
                 ax: Optional[Axes] = None, figsize: Tuple[int, int] = (10, 6), dpi: int = 100) -> Figure:
    """
    Plots training and evaluation metrics on a given or new matplotlib axis.

    Parameters:
    - train_metrics (List[float]): List of training metric values.
    - eval_metrics (List[float]): List of evaluation metric values.
    - metric_name (str): Name of the metric being plotted. Defaults to "Loss".
    - eval_epochs (List[int]): List of evaluation epochs like [1,3,..59]
    - ax (Optional[Axes]): Matplotlib Axes object to plot on. If None, creates a new figure and axes. Defaults to None.
    - figsize (Tuple[int, int]): Figure size for the new figure if ax is None. Defaults to (10, 6).
    - dpi (int): Dots per inch for the figure's resolution if ax is None. Defaults to 100.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
    
    if eval_epochs is None:
        eval_epochs = range(len(eval_metrics))
    ax.plot(eval_epochs, eval_metrics, label=f'Eval {metric_name}')
    if train_metrics is not None:
        ax.plot(train_metrics, label=f'Train {metric_name}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric_name)
    ax.set_title(f'Train vs. Eval {metric_name}')
    ax.legend()
    return fig


def plot_all_metrics(metrics_data: Dict[str, List[int]], fig_title: str, eval_epochs: Optional[List[int]] = None,
                     cols: int=2, figsize: Tuple[int]=(14, 7), dpi: int=100) -> Figure:
    """
    Creates a grid of subplots for multiple metrics using the plot_metrics function.

    Parameters:
    - metrics_data: Dictionary with metric names as keys and tuples of (train_metrics, eval_metrics) as values.
    - fig_title: global title of the figure
    - eval_epochs (List[int]): List of evaluation epochs like [1,3,..59]
    - cols: Number of columns in the subplot grid.
    - figsize: Figure size for the entire grid.
    - dpi: Resolution in dots per inch.
    """
    n = len(metrics_data)
    rows = n // cols + (n % cols > 0)
    fig, axs = plt.subplots(rows, cols, figsize=figsize, dpi=dpi, constrained_layout=True)

    # Flatten axs for easy iteration, in case of single row
    axs = axs.flatten()

    for ax, (metric_name, metrics) in zip(axs, metrics_data.items()):
        if isinstance(metrics, tuple):
            train_metrics, eval_metrics = metrics
        else:
            train_metrics = None
            eval_metrics = metrics
        plot_metrics(eval_metrics, train_metrics, eval_epochs=eval_epochs, metric_name=metric_name, ax=ax)
        
    # If there are any remaining subplots, hide them
    for i in range(len(metrics_data), len(axs)):
        fig.delaxes(axs[i])

    if fig_title:
        fig.suptitle(fig_title, fontsize = 16)
    return fig


def plot_confusion_matrix(tp: int, fp: int, tn: int, fn: int, name: str,
                          ax: Optional[Axes] = None, figsize: Tuple[int, int] = (10, 6), dpi: int = 100) -> Figure:
    """
    Plots a confusion matrix.

    Parameters:
    - tp: Number of true positives.
    - fp: Number of false positives.
    - tn: Number of true negatives.
    - fn: Number of false negatives.
    - figsize: Tuple representing figure size.
    - dpi: Dots per inch (image resolution).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
        
    conf_matrix = [[tp, fp], [fn, tn]]
    cax = ax.matshow(conf_matrix, cmap='Blues')
    ax.set_title(f'Confusion Matrix {name}')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + ['Positive', 'Negative'])
    ax.set_yticklabels([''] + ['Positive', 'Negative'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    # Annotate the matrix with text
    for (i, j), val in np.ndenumerate(conf_matrix):
        ax.text(j, i, f'{val}', ha='center', va='center', color='black')
    
    return fig
                            
# example usage
with open('baseline_BCE_50.json', 'r') as file:
    metrics_data = json.load(file)

metrics_data = preprocess_results(metrics_data)
eval_epochs = range(1,60,2)
# plotting all metrics using plot_all_metrics
fig_all = plot_all_metrics(metrics_data, eval_epochs=eval_epochs, fig_title="Training Evaluation Metrics")
plt.show()
