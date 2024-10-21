import math
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, mean_absolute_error


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


def evaluate(model, X_test, y_test, threshold: float = None, 
             mode: Literal["classification", "regression"] = "classification") -> dict:
    """
    Evaluates the performance of a scikit-learn model (binary classification or regression).

    Args:
        model (object): The trained model (with a predict_proba method for classification).
        X_test (array-like): The input features for testing.
        y_test (array-like): The true labels or values for testing.
        threshold (float): For classification, the probability threshold. Default is 0.5.
                           For regression, the allowed error threshold. Default is 0.2 (20%).
        mode (Literal["classification", "regression"]): The evaluation mode. 
            - "classification": Confusion matrix, accuracy, precision, recall.
            - "regression": MAE, MSE, RMSE with optional tolerance check.

    Returns:
        dict: A dictionary with relevant evaluation metrics.

    Raises:
        ValueError: If the mode is invalid or threshold is not in the valid range.
    """
    if threshold is None:
        threshold = 0.5 if mode == "classification" else 0.2

    if not (0 <= threshold <= 1):
        raise ValueError("Threshold must be between 0 and 1.")

    if mode == "classification":
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability of class 1
        y_pred = (y_prob >= threshold).astype(int)

        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        return {
            'confusion_matrix': cm,
            'accuracy': round(accuracy, 2),
            'precision': round(precision, 2),
            'recall': round(recall, 2)
        }

    elif mode == "regression":
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = np.mean((y_test - y_pred) ** 2)  
        rmse = np.sqrt(mse) 

        # Optional: Check if errors are within the tolerance threshold
        tolerance_mae = threshold * np.mean(np.abs(y_test))
        tolerance_rmse = threshold * np.std(y_test)
        tolerance_mse = threshold * np.var(y_test)

        within_tolerance = {
            'mae_upperbound_tolerance': round(tolerance_mae - mae, 2),
            'rmse_upperbound_tolerance': round(tolerance_rmse - rmse, 2),
            'mse_upperbound_tolerance': round(tolerance_mse - mse, 2)
        }

        return {
            'mae': round(mae, 2),
            'mse': round(mse, 2),
            'rmse': round(rmse, 2),
            **within_tolerance
        }

    else:
        raise ValueError("Invalid mode. Mode must be 'classification' or 'regression'.")


def plot_confusion_matrix(cm):
    """Plots the confusion matrix using Seaborn heatmap.
    
    Args:
        cm (array-like): confusion matrix in array-like
    Returns:
        None: This function displays the matrix heatmap but does not return any value.
    
    """
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def plot_3d_scatter(df, columns):
    """Draw 3d scatter from df and columns inputed, the -1 index of the columns will be the Z in 3D scatter

    Args:
        df (pd.Dataframe): dataframe use to draw the 3D scatter
        columns (list[str]): list of columns to be draw the scatter plot. len(columns) should be 3

    Returns:
        None: This function displays the scatter but does not return any value.
        
    Raises:
        ValueError: ValueError if len(columns) != 3
    """
    if len(columns) != 3:
        raise ValueError("len(columns) are not equal to 3.")
    
    x_col, y_col, z_col = columns
    
    fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col,
                        title=f'Scatter plot for target {z_col}')
    
    fig.update_layout(scene = dict(
                        xaxis_title=x_col,
                        yaxis_title=y_col,
                        zaxis_title=z_col),
                      width=800,
                      height=800,
                      margin=dict(r=20, b=10, l=10, t=40))
    
    fig.show()