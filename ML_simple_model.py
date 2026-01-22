
!pip install -q xgboost

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.weightstats import ttest_1samp
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR

# Ignore warnings
warnings.filterwarnings("ignore")

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
            #use scaling for linear regression and svr
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

# 3. Define Models (Included GBM)
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror'),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf', C=100, gamma='scale') # C and gamma kept as requested
}

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

for name, model in models.items():
    import copy
    metrics = evaluate_model(copy.deepcopy(model), X, y, name, val_type='step-forward')
    results_step.append([name] + list(metrics))

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

