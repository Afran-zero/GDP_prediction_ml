import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.arima.model import ARIMA

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

warnings.filterwarnings("ignore")

# Dampening factor for ML residual correction
DAMPENING_ALPHA = 0.5

file_id = "19p212T1s9EbudQH4G4rxMgFP4vH1sVWQ4y7f_xWqXnM"
url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"
print("Loading dataset...")
df = pd.read_csv(url)
df = df.sort_values('Year').reset_index(drop=True)

target_col = 'GDP_Current_USD'
feature_cols = [c for c in df.columns if c not in ['Year', target_col]]

df_feat = df.copy()

X = df_feat[feature_cols].values
y = df_feat[target_col].values


def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def pbias(y_true, y_pred):
    if np.sum(y_true) == 0:
        return np.nan
    return 100.0 * np.sum(y_pred - y_true) / np.sum(y_true)


def rel_error_mean_signed(y_true, y_pred):
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100


def directional_accuracy(y_true, y_pred):
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    if len(y_true_diff) < 2:
        return np.nan
    correct_dir = np.sign(y_true_diff) == np.sign(y_pred_diff)
    return np.mean(correct_dir) * 100


def theils_u(y_true, y_pred):
    naive_forecast = y_true[:-1]
    actual_next = y_true[1:]
    if len(naive_forecast) == 0:
        return np.nan
    rmse_naive = np.sqrt(mean_squared_error(actual_next, naive_forecast))
    rmse_model = np.sqrt(mean_squared_error(y_true, y_pred))
    if rmse_naive == 0:
        return np.nan
    return rmse_model / rmse_naive


def compute_metrics(y_true_all, y_pred_all):
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    mask = ~np.isnan(y_pred_all) & ~np.isnan(y_true_all)
    y_true_all = y_true_all[mask]
    y_pred_all = y_pred_all[mask]

    if len(y_true_all) == 0:
        return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    mae = mean_absolute_error(y_true_all, y_pred_all)
    mape = mean_absolute_percentage_error(y_true_all, y_pred_all)
    r2 = r2_score(y_true_all, y_pred_all)
    pb = pbias(y_true_all, y_pred_all)
    rel_err = rel_error_mean_signed(y_true_all, y_pred_all)
    dir_acc = directional_accuracy(y_true_all, y_pred_all)
    theil_u = theils_u(y_true_all, y_pred_all)
    dw = durbin_watson(y_true_all - y_pred_all)

    return [rmse, mae, mape, r2, pb, rel_err, dir_acc, theil_u, dw]


def evaluate_hybrid_arima_ensemble(df_feat, target_col, feature_cols):
    y = df_feat[target_col].values
    X = df_feat[feature_cols].values
    n = len(y)
    min_train_size = 10

    y_true_all = []
    arima_preds, gbm_preds = [], []
    hybrid_gbm_preds, hybrid_rf_preds, hybrid_xgb_preds, hybrid_stack_preds = [], [], [], []

    for i in range(min_train_size, n):
        y_train = y[:i]
        X_train = X[:i]
        X_test = X[i:i+1]

        try:
            arima = ARIMA(y_train, order=(3, 1, 2))
            arima_fit = arima.fit()
            arima_forecast = arima_fit.forecast(1)[0]
        except Exception:
            continue

        try:
            arima_in_sample = arima_fit.predict(start=0, end=len(y_train) - 1)
            resid = y_train - arima_in_sample
        except Exception:
            continue

        if len(resid) != len(X_train):
            min_len = min(len(resid), len(X_train))
            resid = resid[-min_len:]
            X_train = X_train[-min_len:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        gbm_resid = GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=2,
            subsample=0.8, min_samples_leaf=4, random_state=42
        )
        try:
            gbm_resid.fit(X_train_scaled, resid)
            resid_pred_gbm = gbm_resid.predict(X_test_scaled)[0]
        except Exception:
            resid_pred_gbm = np.nan

        rf_resid = RandomForestRegressor(
            n_estimators=500, random_state=42, max_depth=3, min_samples_leaf=5
        )
        try:
            rf_resid.fit(X_train_scaled, resid)
            resid_pred_rf = rf_resid.predict(X_test_scaled)[0]
        except Exception:
            resid_pred_rf = np.nan

        if HAS_XGB:
            xgb_resid = XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=3,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=2.0,
                random_state=42, objective='reg:squarederror'
            )
            try:
                xgb_resid.fit(X_train_scaled, resid)
                resid_pred_xgb = xgb_resid.predict(X_test_scaled)[0]
            except Exception:
                resid_pred_xgb = np.nan
        else:
            resid_pred_xgb = np.nan

        hybrid_pred_gbm = arima_forecast + (DAMPENING_ALPHA * resid_pred_gbm)
        hybrid_pred_rf = arima_forecast + (DAMPENING_ALPHA * resid_pred_rf)
        hybrid_pred_xgb = arima_forecast + (DAMPENING_ALPHA * resid_pred_xgb)

        valid_resid_preds = [p for p in [resid_pred_gbm, resid_pred_rf, resid_pred_xgb] if not np.isnan(p)]
        if len(valid_resid_preds) > 0:
            resid_stack = float(np.mean(valid_resid_preds))
            hybrid_pred_stack = arima_forecast + (DAMPENING_ALPHA * resid_stack)
        else:
            hybrid_pred_stack = np.nan

        gbm_y = GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=3,
            subsample=0.8, random_state=42
        )
        try:
            gbm_y.fit(X_train_scaled, y_train)
            gbm_pred = gbm_y.predict(X_test_scaled)[0]
        except Exception:
            gbm_pred = np.nan

        y_true_all.append(y[i])
        arima_preds.append(arima_forecast)
        gbm_preds.append(gbm_pred)
        hybrid_gbm_preds.append(hybrid_pred_gbm)
        hybrid_rf_preds.append(hybrid_pred_rf)
        hybrid_xgb_preds.append(hybrid_pred_xgb)
        hybrid_stack_preds.append(hybrid_pred_stack)

    rows = []
    rows.append(['ARIMA'] + compute_metrics(y_true_all, arima_preds))
    rows.append(['Hybrid-GBM'] + compute_metrics(y_true_all, hybrid_gbm_preds))
    rows.append(['Hybrid-RF'] + compute_metrics(y_true_all, hybrid_rf_preds))
    if HAS_XGB:
        rows.append(['Hybrid-XGB'] + compute_metrics(y_true_all, hybrid_xgb_preds))
    rows.append(['Hybrid-Stack'] + compute_metrics(y_true_all, hybrid_stack_preds))
    rows.append(['GBM'] + compute_metrics(y_true_all, gbm_preds))
    return rows


columns = [
    "Model", "RMSE", "MAE", "MAPE (%)", "R²",
    "PBIAS (%)", "Rel_Error_Mean (%)", "Directional_Accuracy (%)",
    "Theil's U", "Durbin-Watson"
]

print("-" * 135)
print(f"{'Hybrid Models (Step-Forward)':^135}")
print("-" * 135)

results_hybrid = evaluate_hybrid_arima_ensemble(df_feat, target_col, feature_cols)
df_hybrid = pd.DataFrame(results_hybrid, columns=columns)
df_hybrid = df_hybrid.fillna("N/A")
pd.set_option('display.float_format', lambda x: '%.4f' % x)
pd.set_option('display.max_columns', None)
print(df_hybrid.to_string(index=False))
print("-" * 135)

#Sensitivity analysis of model performance 