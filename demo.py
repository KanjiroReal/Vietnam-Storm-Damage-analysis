import warnings

warnings.filterwarnings(action="ignore" ,category=FutureWarning)
warnings.filterwarnings(action="ignore" ,category=UserWarning)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import load as j_load
from pytorch_tabnet.tab_model import TabNetRegressor

from src.utils import evaluate
from src.const import CATEGORICAL_TARGETS, ATTRIBUTES, DATA_HEAD, LINEAR_TARGETS


def get_scaler():
    filename = DATA_HEAD / "STORM_preprocessed_medianfill_1.csv"
    base_df = pd.read_csv(str(filename), index_col=0)

    X_categorical = base_df[ATTRIBUTES]
    y = base_df[CATEGORICAL_TARGETS[0]]

    X_train_categorical, X_test_categorical, _, y_test_categorical = train_test_split(X_categorical, y, test_size=0.1, random_state=1)

    categorical_scaler = StandardScaler()
    X_train_categorical_scaled = categorical_scaler.fit_transform(X_train_categorical)


    X_linear = base_df[ATTRIBUTES+CATEGORICAL_TARGETS]
    y = base_df[LINEAR_TARGETS[0]]

    X_train_linear, X_test_linear, _, y_test_linear = train_test_split(X_linear, y, test_size=0.1, random_state=1)

    linear_scaler = StandardScaler()
    X_train_linear_scaled = linear_scaler.fit_transform(X_train_linear)

    return categorical_scaler, linear_scaler

def load_model(path:str):
    """Auto load model with provided file's ext"""
    supported = ["joblib", "zip"]
    ext = path.split('.')[-1]
    
    if ext not in supported:
        raise ValueError(f"Not Support load .{ext} file. try with these file: {supported.join(',')}")
    
    # ML
    if ext == supported[0]:
        model = j_load(path)
    
    # DL
    elif ext == supported[1]:
        model = TabNetRegressor()
        model.load_model(path)
    
    return model

def normalize_landfall_location(df: pd.DataFrame) -> pd.DataFrame:
    """Convert MainLandfallLocation from number to name of region"""
    x_label = ['NorthEast', 'NorthWest', 'Red River Delta', 'North Central Coast', 
               'South Central Coast', 'Central Highlands', 'SouthEast', 'Mekong River Delta']
    map_dict = {i: name for i, name in enumerate(x_label, start=1)}
    
    df_plot = df.copy()
    df_plot['MainLandfallLocation'] = df_plot['MainLandfallLocation'].map(map_dict)
    return df_plot

# Slide
def slide_demo(model, test_df: pd.DataFrame, scaler: StandardScaler) -> None:
    y_col = CATEGORICAL_TARGETS[1]  # Slide column
    y_true = test_df[y_col]
    X = test_df[ATTRIBUTES]
    
    X = scaler.transform(X)
    
    threshold = 0.8
    eval_values = evaluate(model, X, y_true, threshold=threshold, mode="classification")
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=1.5)
    
    # 1. Main plot (Actual vs Predicted)
    ax1 = fig.add_subplot(gs[0, :])
    
    y_prob = model.predict_proba(X)[:, 1]
    
    df_plot = normalize_landfall_location(test_df)
    storm_labels = [f"{id} at {loc}" for id, loc in zip(df_plot['ID'], df_plot['MainLandfallLocation'])]
    
    ax1.axhspan(threshold, 1.0, color='coral', alpha=0.2, label='1')
    ax1.axhspan(0, threshold, color='blue', alpha=0.2, label='0')
    
    x = np.arange(len(storm_labels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, y_true, width, label='Actual', color='skyblue')
    bars2 = ax1.bar(x + width/2, y_prob, width, label='Predicted Probability', color='lightcoral')
    
    def autolabel(bars, values):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}',
                    ha='center', va='bottom')
    
    autolabel(bars1, y_true)
    autolabel(bars2, y_prob)
    
    ax1.set_xlabel('Storm ID and Location')
    ax1.set_ylabel('Landslide Probability')
    ax1.set_title('Actual vs Predicted Landslide Probability for Each Storm')
    ax1.set_xticks(x)
    ax1.set_xticklabels(storm_labels, rotation=45, ha='right')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label='Threshold (0.8)')
    
    # 2. Confusion Matrix
    ax2 = fig.add_subplot(gs[1, 0])
    cm = eval_values['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Confusion Matrix')
    
    # 3. Performance Metrics
    ax3 = fig.add_subplot(gs[1, 1])
    metrics = {k: v for k, v in eval_values.items() if k != 'confusion_matrix'}
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    
    bars = ax3.bar(range(len(metrics)), list(metrics.values()), color=colors)
    ax3.set_xticks(range(len(metrics)))
    ax3.set_xticklabels(metrics.keys(), rotation=45)
    ax3.set_ylim(0, 1)
    ax3.set_title('Model Performance Metrics')
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.show()    
    return

# FLood
def flood_demo(model, test_df: pd.DataFrame, scaler: StandardScaler) -> None:
    y_col = CATEGORICAL_TARGETS[0]  # Flood column
    y_true = test_df[y_col]
    X = test_df[ATTRIBUTES]
    
    X = scaler.transform(X)
    
    threshold = 0.3
    eval_values = evaluate(model, X, y_true, threshold=threshold, mode="classification")
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=1.5)
    
    # 1. Main plot (Actual vs Predicted)
    ax1 = fig.add_subplot(gs[0, :])
    
    y_prob = model.predict_proba(X)[:, 1]
    
    df_plot = normalize_landfall_location(test_df)
    storm_labels = [f"{id} at {loc}" for id, loc in zip(df_plot['ID'], df_plot['MainLandfallLocation'])]
    
    ax1.axhspan(threshold, 1.0, color='coral', alpha=0.2, label='1')
    ax1.axhspan(0, threshold, color='blue', alpha=0.2, label='0')
    
    x = np.arange(len(storm_labels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, y_true, width, label='Actual', color='skyblue')
    bars2 = ax1.bar(x + width/2, y_prob, width, label='Predicted Probability', color='lightcoral')
    
    def autolabel(bars, values):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}',
                    ha='center', va='bottom')
    
    autolabel(bars1, y_true)
    autolabel(bars2, y_prob)
    
    ax1.set_xlabel('Storm ID and Location')
    ax1.set_ylabel('Flood Probability')
    ax1.set_title('Actual vs Predicted Flood Probability for Each Storm')
    ax1.set_xticks(x)
    ax1.set_xticklabels(storm_labels, rotation=45, ha='right')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label='Threshold (0.3)')
    
    # 2. Confusion Matrix
    ax2 = fig.add_subplot(gs[1, 0])
    cm = eval_values['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Confusion Matrix')
    
    # 3. Performance Metrics
    ax3 = fig.add_subplot(gs[1, 1])
    metrics = {k: v for k, v in eval_values.items() if k != 'confusion_matrix'}
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    
    bars = ax3.bar(range(len(metrics)), list(metrics.values()), color=colors)
    ax3.set_xticks(range(len(metrics)))
    ax3.set_xticklabels(metrics.keys(), rotation=45)
    ax3.set_ylim(0, 1)
    ax3.set_title('Model Performance Metrics')
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.show()
    
    return

# Injured
def injured_demo(ML_model, DL_model, test_df: pd.DataFrame, scaler: StandardScaler) -> None:
    y_col = LINEAR_TARGETS[1]  # NoInjured column
    y_true = test_df[y_col]
    X = test_df[ATTRIBUTES + CATEGORICAL_TARGETS]  # Linear models use categorical targets as features
    
    X = scaler.transform(X)
    
    threshold = 0.3  
    ml_eval = evaluate(ML_model, X, y_true, threshold=threshold, mode="regression")
    
    y_pred_ml = ML_model.predict(X)
    y_pred_dl = DL_model.predict(X)[:, 0]
    
    dl_eval = evaluate(DL_model, X, y_true, threshold=threshold, mode="regression")
    
    tolerance = threshold * np.abs(y_true)
    upper_bounds = y_true + tolerance
    lower_bounds = y_true - tolerance
    
    df_plot = normalize_landfall_location(test_df)
    storm_labels = [f"{id}" for id in df_plot['ID']]
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=2) 
    
    # 1. Main plot (Actual vs Predicted)
    ax1 = fig.add_subplot(gs[0])
    
    x = np.arange(len(storm_labels))
    width = 0.2
    
    for i, (upper, lower) in enumerate(zip(upper_bounds, lower_bounds)):
        ax1.fill_between([i-0.3, i+0.3], [upper, upper], [lower, lower], 
                        color='gray', alpha=0.3)
        # Upperbound
        ax1.hlines(y=upper, xmin=i-0.3, xmax=i+0.3, colors='darkgray', linestyles='--', alpha=0.5)
        ax1.text(i, upper, f'↑{int(upper)}', 
                ha='center', va='bottom', color='darkgray', fontsize=10)
        # Lowerbound
        ax1.hlines(y=lower, xmin=i-0.3, xmax=i+0.3, colors='darkgray', linestyles='--', alpha=0.5)
        ax1.text(i, lower, f'↓{int(lower)}', 
                ha='center', va='top', color='darkgray', fontsize=10)
    
    bars1 = ax1.bar(x - width, y_true, width, label='Actual', color='skyblue')
    bars2 = ax1.bar(x, y_pred_ml, width, label='ML Prediction', color='lightcoral')
    bars3 = ax1.bar(x + width, y_pred_dl, width, label='DL Prediction', color='lightgreen')
    
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    ax1.set_xlabel('Storm ID')
    ax1.set_ylabel('Number of Injured People')
    ax1.set_title('Actual vs Predicted Number of Injured People for Each Storm')
    ax1.set_xticks(x)
    ax1.set_xticklabels(storm_labels, rotation=45, ha='right')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Performance Metrics Table
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    
    metrics_data = {
        'Metrics': ['MAE', 'MSE', 'RMSE', 'MAE Tolerance', 'MSE Tolerance', 'RMSE Tolerance'],
        'ML Model': [
            ml_eval['mae'],
            ml_eval['mse'],
            ml_eval['rmse'],
            ml_eval['mae_upperbound_tolerance'],
            ml_eval['mse_upperbound_tolerance'],
            ml_eval['rmse_upperbound_tolerance']
        ],
        'DL Model': [
            dl_eval['mae'],
            dl_eval['mse'],
            dl_eval['rmse'],
            dl_eval['mae_upperbound_tolerance'],
            dl_eval['mse_upperbound_tolerance'],
            dl_eval['rmse_upperbound_tolerance']
        ]
    }
    
    cell_colors = []
    for i in range(len(metrics_data['Metrics'])):
        ml_value = metrics_data['ML Model'][i]
        dl_value = metrics_data['DL Model'][i]
        
        row_colors = ['white', 'white', 'white']
        
        if i < 3:  
            better_value = min(ml_value, dl_value)
            if ml_value == better_value:
                row_colors[1] = '#ffcdd2'  # Màu đỏ nhạt
            if dl_value == better_value:
                row_colors[2] = '#ffcdd2'  # Màu đỏ nhạt
                
        else:  
            better_value = max(ml_value, dl_value)
            if ml_value == better_value:
                row_colors[1] = '#ffcdd2'  # Màu đỏ nhạt
            if dl_value == better_value:
                row_colors[2] = '#ffcdd2'  # Màu đỏ nhạt
                
        cell_colors.append(row_colors)
    
    table = ax2.table(
        cellText=[[metrics_data['Metrics'][i], 
                  f"{metrics_data['ML Model'][i]:.2f}", 
                  f"{metrics_data['DL Model'][i]:.2f}"] 
                 for i in range(len(metrics_data['Metrics']))],
        colLabels=['Metrics', 'ML Model', 'DL Model'],
        loc='center',
        cellLoc='center',
        colColours=['lightgray']*3,
        cellColours=cell_colors
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    ax2.set_title('Model Performance Comparison', pad=30, fontsize=12)
    plt.show()
    return

# Death
def death_demo(ML_model, DL_model, test_df: pd.DataFrame, scaler: StandardScaler) -> None:
    y_col = LINEAR_TARGETS[0]  # NoDeath column
    y_true = test_df[y_col]
    X = test_df[ATTRIBUTES + CATEGORICAL_TARGETS]
    
    X = scaler.transform(X)
    
    threshold = 0.3
    ml_eval = evaluate(ML_model, X, y_true, threshold=threshold, mode="regression")
    
    y_pred_ml = ML_model.predict(X)
    y_pred_dl = DL_model.predict(X)[:, 0]
    
    dl_eval = evaluate(DL_model, X, y_true, threshold=threshold, mode="regression")
    
    tolerance = threshold * np.abs(y_true)
    upper_bounds = y_true + tolerance
    lower_bounds = y_true - tolerance
    
    df_plot = normalize_landfall_location(test_df)
    storm_labels = [f"{id}" for id in df_plot['ID']]
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=2)
    
    # 1. Main plot (Actual vs Predicted)
    ax1 = fig.add_subplot(gs[0])
    
    x = np.arange(len(storm_labels))
    width = 0.2
    
    for i, (upper, lower) in enumerate(zip(upper_bounds, lower_bounds)):
        ax1.fill_between([i-0.3, i+0.3], [upper, upper], [lower, lower], 
                        color='gray', alpha=0.3)
        ax1.hlines(y=upper, xmin=i-0.3, xmax=i+0.3, colors='darkgray', linestyles='--', alpha=0.5)
        ax1.text(i, upper, f'↑{int(upper)}', 
                ha='center', va='bottom', color='darkgray', fontsize=10)
        ax1.hlines(y=lower, xmin=i-0.3, xmax=i+0.3, colors='darkgray', linestyles='--', alpha=0.5)
        ax1.text(i, lower, f'↓{int(lower)}', 
                ha='center', va='top', color='darkgray', fontsize=10)
    
    bars1 = ax1.bar(x - width, y_true, width, label='Actual', color='skyblue')
    bars2 = ax1.bar(x, y_pred_ml, width, label='ML Prediction', color='lightcoral')
    bars3 = ax1.bar(x + width, y_pred_dl, width, label='DL Prediction', color='lightgreen')
    
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    ax1.set_xlabel('Storm ID')
    ax1.set_ylabel('Number of Deaths')
    ax1.set_title('Actual vs Predicted Number of Deaths for Each Storm')
    ax1.set_xticks(x)
    ax1.set_xticklabels(storm_labels, rotation=45, ha='right')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Performance Metrics Table
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    
    metrics_data = {
        'Metrics': ['MAE', 'MSE', 'RMSE', 'MAE Tolerance', 'MSE Tolerance', 'RMSE Tolerance'],
        'ML Model': [ml_eval[k] for k in ['mae', 'mse', 'rmse', 'mae_upperbound_tolerance', 'mse_upperbound_tolerance', 'rmse_upperbound_tolerance']],
        'DL Model': [dl_eval[k] for k in ['mae', 'mse', 'rmse', 'mae_upperbound_tolerance', 'mse_upperbound_tolerance', 'rmse_upperbound_tolerance']]
    }
    
    cell_colors = []
    for i in range(len(metrics_data['Metrics'])):
        ml_value = metrics_data['ML Model'][i]
        dl_value = metrics_data['DL Model'][i]
        row_colors = ['white', 'white', 'white']
        if i < 3: 
            better_value = min(ml_value, dl_value)
            if ml_value == better_value: row_colors[1] = '#ffcdd2'
            if dl_value == better_value: row_colors[2] = '#ffcdd2'
        else: 
            better_value = max(ml_value, dl_value)
            if ml_value == better_value: row_colors[1] = '#ffcdd2'
            if dl_value == better_value: row_colors[2] = '#ffcdd2'
        cell_colors.append(row_colors)
    
    table = ax2.table(
        cellText=[[metrics_data['Metrics'][i], 
                  f"{metrics_data['ML Model'][i]:.2f}", 
                  f"{metrics_data['DL Model'][i]:.2f}"] 
                 for i in range(len(metrics_data['Metrics']))],
        colLabels=['Metrics', 'ML Model', 'DL Model'],
        loc='center',
        cellLoc='center',
        colColours=['lightgray']*3,
        cellColours=cell_colors
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    ax2.set_title('Model Performance Comparison', pad=30, fontsize=12)
    plt.show()
    return

# Damage
def damage_demo(ML_model, DL_model, test_df: pd.DataFrame, scaler: StandardScaler) -> None:
    y_col = "TotalDamage(000US$)"  # Damage column
    y_true = test_df[y_col]
    X = test_df[ATTRIBUTES + CATEGORICAL_TARGETS]
    
    X = scaler.transform(X)
    
    threshold = 0.3
    ml_eval = evaluate(ML_model, X, y_true, threshold=threshold, mode="regression")
    
    y_pred_ml = ML_model.predict(X)
    y_pred_dl = DL_model.predict(X)[:, 0]
    
    dl_eval = evaluate(DL_model, X, y_true, threshold=threshold, mode="regression")
    
    tolerance = threshold * np.abs(y_true)
    upper_bounds = y_true + tolerance
    lower_bounds = y_true - tolerance
    
    df_plot = normalize_landfall_location(test_df)
    storm_labels = [f"{id}" for id in df_plot['ID']]
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=2)
    
    # 1. Main plot (Actual vs Predicted)
    ax1 = fig.add_subplot(gs[0])
    
    x = np.arange(len(storm_labels))
    width = 0.2
    
    for i, (upper, lower) in enumerate(zip(upper_bounds, lower_bounds)):
        ax1.fill_between([i-0.3, i+0.3], [upper, upper], [lower, lower], 
                        color='gray', alpha=0.3)
        ax1.hlines(y=upper, xmin=i-0.3, xmax=i+0.3, colors='darkgray', linestyles='--', alpha=0.5)
        ax1.text(i, upper, f'↑{int(upper)}', 
                ha='center', va='bottom', color='darkgray', fontsize=10)
        ax1.hlines(y=lower, xmin=i-0.3, xmax=i+0.3, colors='darkgray', linestyles='--', alpha=0.5)
        ax1.text(i, lower, f'↓{int(lower)}', 
                ha='center', va='top', color='darkgray', fontsize=10)
    
    bars1 = ax1.bar(x - width, y_true, width, label='Actual', color='skyblue')
    bars2 = ax1.bar(x, y_pred_ml, width, label='ML Prediction', color='lightcoral')
    bars3 = ax1.bar(x + width, y_pred_dl, width, label='DL Prediction', color='lightgreen')
    
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    ax1.set_xlabel('Storm ID and Location')
    ax1.set_ylabel('Number of Damage')
    ax1.set_title('Actual vs Predicted Number of Damage for Each Storm')
    ax1.set_xticks(x)
    ax1.set_xticklabels(storm_labels, rotation=45, ha='right')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Performance Metrics Table
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    
    metrics_data = {
        'Metrics': ['MAE', 'MSE', 'RMSE', 'MAE Tolerance', 'MSE Tolerance', 'RMSE Tolerance'],
        'ML Model': [ml_eval[k] for k in ['mae', 'mse', 'rmse', 'mae_upperbound_tolerance', 'mse_upperbound_tolerance', 'rmse_upperbound_tolerance']],
        'DL Model': [dl_eval[k] for k in ['mae', 'mse', 'rmse', 'mae_upperbound_tolerance', 'mse_upperbound_tolerance', 'rmse_upperbound_tolerance']]
    }
    
    cell_colors = []
    for i in range(len(metrics_data['Metrics'])):
        ml_value = metrics_data['ML Model'][i]
        dl_value = metrics_data['DL Model'][i]
        row_colors = ['white', 'white', 'white']
        if i < 3:  
            better_value = min(ml_value, dl_value)
            if ml_value == better_value: row_colors[1] = '#ffcdd2'
            if dl_value == better_value: row_colors[2] = '#ffcdd2'
        else:  
            better_value = max(ml_value, dl_value)
            if ml_value == better_value: row_colors[1] = '#ffcdd2'
            if dl_value == better_value: row_colors[2] = '#ffcdd2'
        cell_colors.append(row_colors)
    
    table = ax2.table(
        cellText=[[metrics_data['Metrics'][i], 
                  f"{metrics_data['ML Model'][i]:.2f}", 
                  f"{metrics_data['DL Model'][i]:.2f}"] 
                 for i in range(len(metrics_data['Metrics']))],
        colLabels=['Metrics', 'ML Model', 'DL Model'],
        loc='center',
        cellLoc='center',
        colColours=['lightgray']*3,
        cellColours=cell_colors
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    ax2.set_title('Model Performance Comparison', pad=30, fontsize=12)
    plt.show()
    return

def main():
    model_paths = {
        "flood": {
            "ML": "weights/ML/flood_random_forest.joblib",
        },
        "slide": {
            "ML": {
                "svm": "weights/ML/slide_svm.joblib",
                "rf": "weights/ML/slide_random_forest.joblib"
            }
        },
        "death": {
            "ML": "weights/ML/death_svm.joblib",
            "DL": "weights/DL/death_tabnet.zip"
        },
        "injured": {
            "ML": "weights/ML/injured_svm.joblib",
            "DL": "weights/DL/injured_tabnet.zip"
        },
        "damage": {
            "ML": "weights/ML/damage_svm.joblib",
            "DL": "weights/DL/damage_tabnet.zip"
        }
    }
    
    categorical_scaler, linear_scaler = get_scaler()
    test_df = pd.read_csv(str(DATA_HEAD / "test_storm2224.csv"))
    
    # Categorical predictions
    # slide_demo(load_model(model_paths["slide"]["ML"]["svm"]), test_df, categorical_scaler)
    # flood_demo(load_model(model_paths["flood"]["ML"]), test_df, categorical_scaler)
    
    # Linear predictions
    # death_demo(load_model(model_paths["death"]["ML"]), load_model(model_paths["death"]["DL"]), test_df, linear_scaler)
    # injured_demo(load_model(model_paths["injured"]["ML"]), load_model(model_paths["injured"]["DL"]), test_df, linear_scaler)
    # damage_demo(load_model(model_paths["damage"]["ML"]), load_model(model_paths["damage"]["DL"]), test_df, linear_scaler)

if __name__ == "__main__":
    main()