import math

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve


def convert_df_dtype(df ,target_dtype, target_cols):
    """Convert dtype of the selected column of entire pandas dataframe

        Args:
            df (pd.Dataframe): input dataframe to convert dtype.
            target_dtype (pd.dtype, Any): target dtype to convert
            target_cols (list[str]): list of target column to convert

        Returns:
            pd.Dataframe: return converted dataframe
        """
    for col in target_cols:
        df[col] = df[col].round(0).astype(target_dtype)
        
    return df

def cal_subplot_size(columns:list[str]):
    """Calculate the optimal number of rows and columns for subplots.

    This function determines the number of rows and columns required
    to plot multiple distributions by approximating a square layout.

    Args:
        columns (list[str]): List of column names to plot.

    Returns:
        tuple[int, int]: A tuple containing the number of rows and columns.
    """
    n = len(columns)
    nrows = math.ceil(math.sqrt(n))
    ncols = math.ceil(n/nrows)
    return nrows, ncols

def plot_distributions(df, columns_to_plot, plot_type='boxplot', figsize=(14, 12)):    
    """Plot distributions of multiple columns using either boxplot or histogram.

    This function creates subplots to visualize the distributions of specified columns
    from a DataFrame, allowing either boxplots or histograms with KDE.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data to plot.
        columns_to_plot (list[str]): List of column names to plot.
        plot_type (str, optional): Type of plot to create ('boxplot' or 'histplot'). Defaults: 'boxplot'.
        figsize (tuple[int, int], optional): Size of the figure. Defaults to (14, 12).

    Returns:
        None: This function displays the plots but does not return any value.
    """
    # Calculate the number of rows and columns needed
    nrows, ncols = cal_subplot_size(columns=columns_to_plot)
    
    # Set up the visual style
    sns.set_theme(style="whitegrid")

    # Create subplots
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

    # Plotting each column
    for ax, col in zip(axs.ravel(), columns_to_plot):
        if plot_type == 'boxplot':
            sns.boxplot(x=df[col], ax=ax, color='coral')
            ax.set_ylabel('Value')
        elif plot_type == 'histplot':
            sns.histplot(df[col], kde=True, ax=ax, bins=10, color='skyblue')
            ax.set_ylabel('Frequency')
        
        ax.set_title(f'Distribution of {col}', fontsize=12)
        ax.set_xlabel(col)

    # Remove any empty subplots
    for i in range(len(columns_to_plot), nrows * ncols):
        fig.delaxes(axs.ravel()[i])

    plt.tight_layout()
    plt.show()
    
    
def plot_corr(df, cols):
    """Plot a correlation matrix heatmap for selected columns.

    This function computes the correlation matrix for the specified columns 
    of a DataFrame and visualizes it using a heatmap with annotations.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        cols (list[str]): List of column names to include in the correlation matrix.

    Returns:
        None: This function displays the heatmap but does not return any value.
    """
    corr_matrix = df[cols].corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt='.2f', linewidths=0.5)

    plt.title('Correlation Matrix')
    plt.show()


def evaluate(model, X_test, y_test, threshold=0.5):
    """Evaluates a binary classification model's performance. only using for scikit-learn's model.
    
    Args:
        model (object): The trained classification model with a predict_proba method.
        X_test (array-like): The input features for testing.
        y_test (array-like): The true labels for testing.
        threshold (float): The probability threshold for classification. Default: 0.5
    
    Returns:
        dict: A dictionary containing evaluation metrics:
              - 'confusion_matrix': The confusion matrix
              - 'accuracy': The accuracy score (rounded to 2 decimal places)
              - 'precision': The precision score (rounded to 2 decimal places)
              - 'recall': The recall score (rounded to 2 decimal places)
    """
    y_prob = model.predict_proba(X_test)[:, 1] # probability of class 1
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    return {
        'confusion_matrix': cm,
        'accuracy': round(accuracy,2),
        'precision': round(precision,2),
        'recall': round(recall,2)
    }