
# !pip install -q xgboost opcuna statsmodels matplotlib scikit-learn pandas numpy

import pandas as pd
import numpy as np
import warnings
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.weightstats import ttest_1samp
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR

# Ignore warnings
warnings.filterwarnings("ignore")

# Optuna (for hyperparameter tuning)
try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

# 1. Data Loading
file_id = "19p212T1s9EbudQH4G4rxMgFP4vH1sVWQ4y7f_xWqXnM"
url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"
print("Loading dataset...")
df = pd.read_csv(url)
df = df.sort_values('Year').reset_index(drop=True)

# Define Features and Target
target_col = 'GDP_Current_USD'
feature_cols = [c for c in df.columns if c not in ['Year', target_col]]

X = df[feature_cols].values
y = df[target_col].values

# 2. Metric Calculation Functions
def mean_absolute_percentage_error(y_true, y_pred): 
    mask = y_true != 0
    if np.sum(mask) == 0: return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def pbias(y_true, y_pred):
    if np.sum(y_true) == 0: return np.nan
    return 100.0 * np.sum(y_pred - y_true) / np.sum(y_true)

def rel_error_mean(y_true, y_pred):
    mask = y_true != 0
    if np.sum(mask) == 0: return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def directional_accuracy(y_true, y_pred):
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    if len(y_true_diff) < 2: return np.nan
    correct_dir = np.sign(y_true_diff) == np.sign(y_pred_diff)
    return np.mean(correct_dir) * 100

def theils_u(y_true, y_pred):
    naive_forecast = y_true[:-1]
    actual_next = y_true[1:]
    if len(naive_forecast) == 0: return np.nan
    rmse_naive = np.sqrt(mean_squared_error(actual_next, naive_forecast))
    rmse_model = np.sqrt(mean_squared_error(y_true, y_pred))
    if rmse_naive == 0: return np.nan
    return rmse_model / rmse_naive

def prei_percent(y_true, y_pred):
    # Percentage Relative Error Index (PREI%)
    rel_err = rel_error_mean(y_true, y_pred)
    if np.isnan(rel_err):
        return np.nan
    return 100.0 - rel_err

def evaluate_model(model, X, y, model_name, val_type='step-forward'):
    y_true_all = []
    y_pred_all = []
    
    if val_type == 'step-forward':
        # Expanding window (Step Forward)
        min_train_size = 10
        for i in range(min_train_size, len(y)):
            X_train, X_test = X[:i], X[i:i+1]
            y_train, y_test = y[:i], y[i:i+1]
            
            # Scaling Features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Scaling TARGET (Y) specifically for SVR
            # SVR struggles with huge target values (e.g., Trillions)
            y_scaler = None
            if 'SVR' in model_name:
                y_scaler = StandardScaler()
                y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            else:
                y_train_s = y_train
            
            # Fit & Predict
            model.fit(X_train_scaled, y_train_s)
            y_pred_s = model.predict(X_test_scaled)
            
            # Inverse Transform Y prediction if we scaled it
            if y_scaler:
                y_pred = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
            else:
                y_pred = y_pred_s
            
            y_true_all.append(y_test[0])
            y_pred_all.append(y_pred[0])

    # Calculate Metrics on AGGREGATED predictions
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    
    rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    mae = mean_absolute_error(y_true_all, y_pred_all)
    mape = mean_absolute_percentage_error(y_true_all, y_pred_all)
    r2 = r2_score(y_true_all, y_pred_all)
    pb = pbias(y_true_all, y_pred_all)
    rel_err = rel_error_mean(y_true_all, y_pred_all)
    dir_acc = directional_accuracy(y_true_all, y_pred_all)
    theil_u = theils_u(y_true_all, y_pred_all)
    dw = durbin_watson(y_true_all - y_pred_all)
    prei = prei_percent(y_true_all, y_pred_all)
    _, p_value, _ = ttest_1samp(y_true_all - y_pred_all, 0.0)
    
    return [rmse, mae, mape, r2, pb, rel_err, dir_acc, theil_u, prei, p_value, dw]


def step_forward_predictions(model, X, y, model_name, scale_features=True):
    y_true_all = []
    y_pred_all = []

    min_train_size = 10
    for i in range(min_train_size, len(y)):
        X_train, X_test = X[:i], X[i:i+1]
        y_train, y_test = y[:i], y[i:i+1]

        if scale_features:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        y_scaler = None
        if 'SVR' in model_name:
            y_scaler = StandardScaler()
            y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        else:
            y_train_s = y_train

        model.fit(X_train_scaled, y_train_s)
        y_pred_s = model.predict(X_test_scaled)

        if y_scaler:
            y_pred = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
        else:
            y_pred = y_pred_s

        y_true_all.append(y_test[0])
        y_pred_all.append(y_pred[0])

    return np.array(y_true_all), np.array(y_pred_all)

def compute_metrics(y_true_all, y_pred_all):
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    if len(y_true_all) == 0:
        return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    mae = mean_absolute_error(y_true_all, y_pred_all)
    mape = mean_absolute_percentage_error(y_true_all, y_pred_all)
    r2 = r2_score(y_true_all, y_pred_all)
    pb = pbias(y_true_all, y_pred_all)
    rel_err = rel_error_mean(y_true_all, y_pred_all)
    dir_acc = directional_accuracy(y_true_all, y_pred_all)
    theil_u = theils_u(y_true_all, y_pred_all)
    dw = durbin_watson(y_true_all - y_pred_all)
    prei = prei_percent(y_true_all, y_pred_all)
    _, p_value, _ = ttest_1samp(y_true_all - y_pred_all, 0.0)

    return [rmse, mae, mape, r2, pb, rel_err, dir_acc, theil_u, prei, p_value, dw]

def evaluate_econometric_models(df, target_col, feature_cols):
    y = df[target_col].values
    n = len(y)
    min_train_size = 10

    results = []

    # ARIMA(3,1,2)
    y_true_arima, y_pred_arima = [], []
    for i in range(min_train_size, n):
        y_train = y[:i]
        try:
            model = ARIMA(y_train, order=(3, 1, 2))
            fitted = model.fit()
            pred = fitted.forecast(1)[0]
            y_true_arima.append(y[i])
            y_pred_arima.append(pred)
        except Exception:
            continue

    metrics = compute_metrics(y_true_arima, y_pred_arima)
    results.append(['ARIMA(3,1,2)'] + metrics)

    # OLS Trend
    y_true_ols, y_pred_ols = [], []
    for i in range(min_train_size, n):
        t = np.arange(i)
        X_train = sm.add_constant(t)
        try:
            model = sm.OLS(y[:i], X_train).fit()
            pred = model.predict(np.array([1, i]))[0]
            y_true_ols.append(y[i])
            y_pred_ols.append(pred)
        except Exception:
            continue

    metrics = compute_metrics(y_true_ols, y_pred_ols)
    results.append(['OLS_Trend'] + metrics)

    # VAR(2) using target + features
    y_true_var, y_pred_var = [], []
    var_df = df[[target_col] + feature_cols]
    for i in range(min_train_size, n):
        train = var_df.iloc[:i]
        if len(train) <= 2:
            continue
        try:
            model = VAR(train)
            fitted = model.fit(2)
            forecast = fitted.forecast(train.values[-2:], steps=1)[0]
            y_true_var.append(y[i])
            y_pred_var.append(forecast[0])
        except Exception:
            continue

    metrics = compute_metrics(y_true_var, y_pred_var)
    results.append(['VAR(2)'] + metrics)

    return results


def econometric_step_forward_predictions(df, target_col, feature_cols):
    y = df[target_col].values
    n = len(y)
    min_train_size = 10

    preds = {}

    # ARIMA(3,1,2)
    y_true_arima, y_pred_arima = [], []
    for i in range(min_train_size, n):
        y_train = y[:i]
        try:
            model = ARIMA(y_train, order=(3, 1, 2))
            fitted = model.fit()
            pred = fitted.forecast(1)[0]
            y_true_arima.append(y[i])
            y_pred_arima.append(pred)
        except Exception:
            continue
    preds['ARIMA(3,1,2)'] = (np.array(y_true_arima), np.array(y_pred_arima))

    # OLS Trend
    y_true_ols, y_pred_ols = [], []
    for i in range(min_train_size, n):
        t = np.arange(i)
        X_train = sm.add_constant(t)
        try:
            model = sm.OLS(y[:i], X_train).fit()
            pred = model.predict(np.array([1, i]))[0]
            y_true_ols.append(y[i])
            y_pred_ols.append(pred)
        except Exception:
            continue
    preds['OLS_Trend'] = (np.array(y_true_ols), np.array(y_pred_ols))

    # VAR(2)
    y_true_var, y_pred_var = [], []
    var_df = df[[target_col] + feature_cols]
    for i in range(min_train_size, n):
        train = var_df.iloc[:i]
        if len(train) <= 2:
            continue
        try:
            model = VAR(train)
            fitted = model.fit(2)
            forecast = fitted.forecast(train.values[-2:], steps=1)[0]
            y_true_var.append(y[i])
            y_pred_var.append(forecast[0])
        except Exception:
            continue
    preds['VAR(2)'] = (np.array(y_true_var), np.array(y_pred_var))

    return preds


def tune_with_optuna(model_name, X, y, n_trials=30):
    if not HAS_OPTUNA:
        print("Optuna not available. Skipping tuning.")
        return None

    def objective(trial):
        if model_name == 'Random Forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                'max_depth': trial.suggest_int('max_depth', 2, 12),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'random_state': 42
            }
            model = RandomForestRegressor(**params)
        elif model_name == 'Gradient Boosting':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 6),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_state': 42
            }
            model = GradientBoostingRegressor(**params)
        elif model_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 8),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42,
                'objective': 'reg:squarederror'
            }
            model = XGBRegressor(**params)
        elif model_name == 'SVR':
            params = {
                'C': trial.suggest_float('C', 1, 300, log=True),
                'gamma': trial.suggest_float('gamma', 1e-4, 1e-1, log=True),
                'kernel': 'rbf'
            }
            model = SVR(**params)
        elif model_name == 'Linear Regression':
            model = LinearRegression()
            params = {}
        else:
            return np.inf

        y_true, y_pred = step_forward_predictions(model, X, y, model_name)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def generate_diagnostic_figures(preds_by_model, metrics_by_model, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['figure.dpi'] = 300

    colors = {
        "Actual": "#000000",
        "Linear Regression": "#1f77b4",
        "Random Forest": "#8c564b",
        "Gradient Boosting": "#2ca02c",
        "XGBoost": "#d62728",
        "SVR": "#9467bd",
        "ARIMA(3,1,2)": "#ff7f0e",
        "VAR(2)": "#17becf",
        "OLS_Trend": "#7f7f7f"
    }

    # Figure 1: Actual vs Predicted GDP (Scatter)
    plt.figure(figsize=(7, 6))
    for model_name, (y_true, y_pred) in preds_by_model.items():
        plt.scatter(y_true, y_pred, s=18, alpha=0.7, label=model_name, color=colors.get(model_name, None))
    min_val = min([np.min(v[0]) for v in preds_by_model.values()])
    max_val = max([np.max(v[0]) for v in preds_by_model.values()])
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='black', label='Perfect Prediction')
    plt.xlabel("Actual GDP")
    plt.ylabel("Predicted GDP")
    plt.title("Actual vs Predicted GDP (Testing Phase)")
    plt.grid(True)
    plt.legend(fontsize=7)
    fig1_png = os.path.join(output_dir, "figure_1_actual_vs_predicted.png")
    fig1_svg = os.path.join(output_dir, "figure_1_actual_vs_predicted.svg")
    plt.tight_layout()
    plt.savefig(fig1_png, dpi=300)
    plt.savefig(fig1_svg)
    plt.show()
    print(f"Saved: {fig1_png}")
    print(f"Saved: {fig1_svg}")

    # Figure 2: Taylor Diagram (approximate using correlation/std ratio)
    def taylor_stats(y_true, y_pred):
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        std_ratio = np.std(y_pred) / np.std(y_true)
        crmsd = np.sqrt(np.mean(((y_pred - np.mean(y_pred)) - (y_true - np.mean(y_true)))**2))
        return corr, std_ratio, crmsd

    models_to_plot = [
        "ARIMA(3,1,2)", "VAR(2)", "XGBoost", "SVR", "Gradient Boosting", "Linear Regression"
    ]
    plt.figure(figsize=(7, 6))
    ax = plt.subplot(111, polar=True)

    for model_name in models_to_plot:
        if model_name not in preds_by_model:
            continue
        y_true, y_pred = preds_by_model[model_name]
        corr, std_ratio, _ = taylor_stats(y_true, y_pred)
        theta = np.arccos(np.clip(corr, -1, 1))
        r = std_ratio
        ax.plot(theta, r, 'o', label=model_name, color=colors.get(model_name, None))

    ax.set_thetalim(0, np.pi / 2)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(90)
    ax.set_title("Taylor Diagram – Model Performance Comparison", pad=20)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=7)

    fig2_png = os.path.join(output_dir, "figure_2_taylor_diagram.png")
    fig2_svg = os.path.join(output_dir, "figure_2_taylor_diagram.svg")
    plt.tight_layout()
    plt.savefig(fig2_png, dpi=300)
    plt.savefig(fig2_svg)
    plt.show()
    print(f"Saved: {fig2_png}")
    print(f"Saved: {fig2_svg}")

    # Figure 3: Relative Error Distribution (APE) - Violin
    ape_data = []
    ape_labels = []
    for model_name, (y_true, y_pred) in preds_by_model.items():
        ape = np.abs((y_true - y_pred) / y_true) * 100
        ape_data.append(ape)
        ape_labels.append(model_name)

    plt.figure(figsize=(9, 5))
    parts = plt.violinplot(ape_data, showmeans=False, showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors.get(ape_labels[i], '#cccccc'))
        pc.set_alpha(0.6)
    plt.xticks(np.arange(1, len(ape_labels) + 1), ape_labels, rotation=45, ha='right')
    plt.ylabel("Absolute Percentage Error (%)")
    plt.title("Relative Error Distribution (APE)")
    plt.grid(True)
    fig3_png = os.path.join(output_dir, "figure_3_ape_violin.png")
    fig3_svg = os.path.join(output_dir, "figure_3_ape_violin.svg")
    plt.tight_layout()
    plt.savefig(fig3_png, dpi=300)
    plt.savefig(fig3_svg)
    plt.show()
    print(f"Saved: {fig3_png}")
    print(f"Saved: {fig3_svg}")

    # Figure 4: Theil's U vs Durbin–Watson Diagnostic Map
    plt.figure(figsize=(7, 6))
    for model_name, metrics in metrics_by_model.items():
        theils_u = metrics.get('Theil\'s U', None)
        dw = metrics.get('Durbin-Watson', None)
        rmse = metrics.get('RMSE', None)
        if theils_u is None or dw is None or rmse is None:
            continue
        plt.scatter(dw, theils_u, s=max(30, rmse / np.nanmean([m['RMSE'] for m in metrics_by_model.values() if m.get('RMSE')]))*80,
                    label=model_name, color=colors.get(model_name, None), alpha=0.8)

    plt.axvspan(1.8, 2.2, color='green', alpha=0.1, label='Ideal DW Zone')
    plt.axhline(1.0, color='red', linestyle='--', label="Theil's U = 1")
    plt.xlabel("Durbin-Watson")
    plt.ylabel("Theil's U")
    plt.title("Theil’s U vs Durbin–Watson Diagnostic Map")
    plt.grid(True)
    plt.legend(fontsize=7)
    fig4_png = os.path.join(output_dir, "figure_4_theilsu_dw.png")
    fig4_svg = os.path.join(output_dir, "figure_4_theilsu_dw.svg")
    plt.tight_layout()
    plt.savefig(fig4_png, dpi=300)
    plt.savefig(fig4_svg)
    plt.show()
    print(f"Saved: {fig4_png}")
    print(f"Saved: {fig4_svg}")

# 3. Define Models (Included GBM)
OPTUNA_TUNE = True
N_TRIALS = 30

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror'),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf', C=100, gamma='scale')
}

if OPTUNA_TUNE:
    print("\nOptuna tuning enabled. Searching best parameters...")
    for name in list(models.keys()):
        if name not in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'SVR', 'Linear Regression']:
            continue
        best_params = tune_with_optuna(name, X, y, n_trials=N_TRIALS)
        if best_params is None:
            continue
        print(f"Best params for {name}: {best_params}")
        if name == 'Random Forest':
            models[name] = RandomForestRegressor(**best_params)
        elif name == 'Gradient Boosting':
            models[name] = GradientBoostingRegressor(**best_params)
        elif name == 'XGBoost':
            models[name] = XGBRegressor(**best_params)
        elif name == 'SVR':
            models[name] = SVR(**best_params)
        elif name == 'Linear Regression':
            models[name] = LinearRegression()

columns = [
    "Model", "RMSE", "MAE", "MAPE (%)", "R²", 
    "PBIAS (%)", "Rel_Error_Mean (%)", "Directional_Accuracy (%)", 
    "Theil's U", "PREI (%)", "p-value", "Durbin-Watson"
]

# 4. Run Evaluation
print("-" * 135)
print(f"{'Validation Strategy: Step-Forward (One-Year Ahead)':^135}")
print("-" * 135)
results_step = []
preds_by_model = {}
metrics_by_model = {}

for name, model in models.items():
    import copy
    metrics = evaluate_model(copy.deepcopy(model), X, y, name, val_type='step-forward')
    results_step.append([name] + list(metrics))

    y_true_sf, y_pred_sf = step_forward_predictions(copy.deepcopy(model), X, y, name)
    preds_by_model[name] = (y_true_sf, y_pred_sf)
    metrics_by_model[name] = {
        "RMSE": metrics[0],
        "MAE": metrics[1],
        "MAPE": metrics[2],
        "R2": metrics[3],
        "PBIAS": metrics[4],
        "Relative Error Mean": metrics[5],
        "Directional Accuracy": metrics[6],
        "Theil's U": metrics[7],
        "PREI": metrics[8],
        "p-value": metrics[9],
        "Durbin-Watson": metrics[10]
    }

df_step = pd.DataFrame(results_step, columns=columns)
df_step = df_step.fillna("N/A")
pd.set_option('display.float_format', lambda x: '%.4f' % x)
pd.set_option('display.max_columns', None)
print(df_step.to_string(index=False))

print("\n" + "-" * 135)
print(f"{'Econometric Models (Step-Forward)':^135}")
print("-" * 135)

results_econ = evaluate_econometric_models(df, target_col, feature_cols)
df_econ = pd.DataFrame(results_econ, columns=columns)
df_econ = df_econ.fillna("N/A")
print(df_econ.to_string(index=False))
print("-" * 135)

# Add econometric predictions for plotting
econ_preds = econometric_step_forward_predictions(df, target_col, feature_cols)
for name, (y_true, y_pred) in econ_preds.items():
    preds_by_model[name] = (y_true, y_pred)
    econ_metrics = compute_metrics(y_true, y_pred)
    metrics_by_model[name] = {
        "RMSE": econ_metrics[0],
        "MAE": econ_metrics[1],
        "MAPE": econ_metrics[2],
        "R2": econ_metrics[3],
        "PBIAS": econ_metrics[4],
        "Relative Error Mean": econ_metrics[5],
        "Directional Accuracy": econ_metrics[6],
        "Theil's U": econ_metrics[7],
        "PREI": econ_metrics[8],
        "p-value": econ_metrics[9],
        "Durbin-Watson": econ_metrics[10]
    }

# Generate diagnostic figures
generate_diagnostic_figures(preds_by_model, metrics_by_model, output_dir="figures")

