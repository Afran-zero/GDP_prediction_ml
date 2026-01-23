import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
import shap
import lime
import lime.lime_tabular
from scipy import stats
from sklearn.metrics import mean_squared_error

# Create directory for high-res plots
os.makedirs("plots_output", exist_ok=True)

# Publication-grade styling
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
font_family = 'Times New Roman' if 'Times New Roman' in available_fonts else 'DejaVu Serif'

plt.rcParams.update({
    'font.size': 12,
    'font.family': font_family,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 300
})

COLORS = {
    "Actual": "#000000",
    "ARIMA": "#ff7f0e",
    "Hybrid-Stack": "#1f77b4",
    "Hybrid-RF": "#2ca02c",  # Added color for Random Forest
    "ResidualFill": "#9e9e9e"
}

# 1. DATA PREPARATION FOR PLOTTING
# Align years with the predictions (starts from min_train_size=10)
min_train_size = 10 # Defined in evaluate_hybrid_arima_ensemble
# forecast_years should match the number of predictions (len(y_true_arr))
forecast_years = df['Year'].values[min_train_size : min_train_size + len(y_true_arr)]
y_true_arr = np.array(y_true_all)
stack_pred_arr = np.array(hybrid_stack_preds)
arima_pred_arr = np.array(arima_preds)
rf_pred_arr = np.array(hybrid_rf_preds) # Added for Hybrid-RF

# CONVERT TO BILLIONS
y_true_arr_billion = y_true_arr / 1e9
stack_pred_arr_billion = stack_pred_arr / 1e9
arima_pred_arr_billion = arima_pred_arr / 1e9
rf_pred_arr_billion = rf_pred_arr / 1e9 # Added for Hybrid-RF

# --- FIGURE 1: TIME-SERIES COMPARISON ---
plt.figure(figsize=(14, 7))
plt.plot(forecast_years, y_true_arr_billion, color=COLORS["Actual"], marker='o', label='Actual GDP', linewidth=2.5, markersize=6)
plt.plot(forecast_years, arima_pred_arr_billion, color=COLORS["ARIMA"], linestyle='--', label='ARIMA Baseline', alpha=0.9, linewidth=2)
plt.plot(forecast_years, rf_pred_arr_billion, color=COLORS["Hybrid-RF"], marker='x', label='Hybrid-RF', linewidth=2.5, markersize=6) # Added Hybrid-RF plot
plt.plot(forecast_years, stack_pred_arr_billion, color=COLORS["Hybrid-Stack"], marker='^', label='Hybrid-Stack (Ours)', linewidth=2.5, markersize=6)
plt.fill_between(forecast_years, arima_pred_arr_billion, stack_pred_arr_billion, color=COLORS["ResidualFill"], alpha=0.2, label='ML Residual Correction')
plt.title("GDP Prediction: Actual vs. Models", fontsize=16, fontweight='bold')
plt.xlabel("Year", fontsize=13)
plt.ylabel("GDP (Billion USD)", fontsize=13)
plt.legend(loc='best', framealpha=0.95)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("plots_output/ts_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig("plots_output/ts_comparison.svg", bbox_inches='tight')
plt.show()

# --- FIGURE 2: RESIDUAL DIAGNOSTICS (HIST + Q-Q) ---
residuals_billion = y_true_arr_billion - stack_pred_arr_billion
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(residuals_billion, kde=True, ax=ax1, color=COLORS["Hybrid-Stack"], edgecolor='black', linewidth=1.2)
ax1.set_title("Residual Distribution (Hybrid-Stack)", fontsize=14, fontweight='bold')
ax1.set_xlabel("Residual (Billion USD)", fontsize=12)
ax1.set_ylabel("Frequency", fontsize=12)

stats.probplot(residuals_billion, dist="norm", plot=ax2)
ax2.get_lines()[0].set_markerfacecolor(COLORS["Hybrid-Stack"])
ax2.get_lines()[0].set_markersize(6)
ax2.set_title("Normal Q-Q Plot", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("plots_output/residual_diagnostics.png", dpi=300, bbox_inches='tight')
plt.savefig("plots_output/residual_diagnostics.svg", bbox_inches='tight')
plt.show()

# --- FIGURE 3: GLOBAL XAI - SHAP BEESWARM (AVERAGED ACROSS RF/GBM/XGB) ---
# Average SHAP values across residual learners to represent the ensemble stack
X_latest_scaled = X_train_for_lime  # Use the last X_train_scaled for SHAP

model_list = [("GBM", gbm_resid)]
if rf_resid is not None: # Check if rf_resid is available
    model_list.append(("RF", rf_resid))
try:
    if xgb_resid is not None: # Check if xgb_resid is available
        model_list.append(("XGB", xgb_resid))
except NameError:
    pass

shap_values_list = []
expected_values = []

for name, model in model_list:
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_latest_scaled)
    # shap.TreeExplainer.shap_values can return a list if there are multiple outputs
    # For regression, it's typically an array. Handle if it's a list for safety.
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0] # Take the first element if it's a list
    shap_values_list.append(shap_vals)
    
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = float(np.mean(ev))
    expected_values.append(ev)

# Only compute if shap_values_list is not empty
if shap_values_list:
    shap_values = np.mean(np.stack(shap_values_list, axis=0), axis=0)
else:
    shap_values = np.array([]) # Handle case where no models are in model_list

expected_value_avg = float(np.mean(expected_values)) if expected_values else 0.0

plt.figure(figsize=(12, 8))
# Only plot if shap_values is not empty
if shap_values.size > 0:
    shap.summary_plot(shap_values, X_latest_scaled, feature_names=feature_cols, show=False, plot_size=(12, 8))
    plt.title("Global Feature Impact on ARIMA Residual Correction", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig("plots_output/shap_summary.png", dpi=300, bbox_inches='tight')
    plt.savefig("plots_output/shap_summary.svg", bbox_inches='tight')
    plt.show()
else:
    print("No SHAP values to plot. Check if residual models were fitted.")

# Minimal SHAP importance table (no GDP correlation)
if shap_values.size > 0:
    mean_shap = np.abs(shap_values).mean(axis=0)
    mean_shap_billion = mean_shap / 1e9
    correlations = [df[col].corr(df[target_col]) for col in feature_cols]
    fi_df = pd.DataFrame({
        'Feature': feature_cols,
        'Mean_SHAP_Billion': mean_shap_billion,
        'Corr_with_GDP': correlations
    }).sort_values(by='Mean_SHAP_Billion', ascending=False)

    # --- FEATURE IMPORTANCE COMPARISON (SIDE-BY-SIDE) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Mean SHAP values (Billions)
    colors_shap = plt.cm.viridis(np.linspace(0.3, 0.9, len(fi_df)))
    bars1 = ax1.barh(fi_df['Feature'], fi_df['Mean_SHAP_Billion'], color=colors_shap, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Mean |SHAP| Value (Billion USD)', fontsize=12, fontweight='bold')
    ax1.set_title('Feature Importance by SHAP Values', fontsize=14, fontweight='bold')
    ax1.grid(True, axis='x', linestyle='--', alpha=0.4)
    ax1.invert_yaxis()

    for bar, val in zip(bars1, fi_df['Mean_SHAP_Billion']):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{val:.2f}B', va='center', fontsize=9, fontweight='bold')

    # Right: Correlation with GDP (green for positive)
    colors_corr = ['#2ecc71' if x >= 0 else '#e74c3c' for x in fi_df['Corr_with_GDP']]
    bars2 = ax2.barh(fi_df['Feature'], fi_df['Corr_with_GDP'], color=colors_corr, edgecolor='black', linewidth=1.2)
    ax2.set_xlabel('Correlation with GDP', fontsize=12, fontweight='bold')
    ax2.set_title('Feature Correlation with GDP', fontsize=14, fontweight='bold')
    ax2.axvline(0, color='black', linewidth=1)
    ax2.grid(True, axis='x', linestyle='--', alpha=0.4)
    ax2.invert_yaxis()

    for bar, val in zip(bars2, fi_df['Corr_with_GDP']):
        ax2.text(val + 0.02 if val >= 0 else val - 0.02,
                 bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right',
                 fontsize=9, fontweight='bold')

    plt.suptitle('Feature Importance Comparison', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig("plots_output/feature_importance_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig("plots_output/feature_importance_comparison.svg", bbox_inches='tight')
    plt.show()
else:
    fi_df = pd.DataFrame(columns=['Feature', 'Mean_SHAP_Billion', 'Corr_with_GDP'])


# --- FIGURE 4: LIME LOCAL EXPLANATION (2008, 2017, 2020, 2023) ---
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_for_lime,
    feature_names=feature_cols,
    mode='regression'
)

years_to_explain = [2008, 2017, 2020, 2023]
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for ax, year in zip(axes, years_to_explain):
    if year not in forecast_years:
        ax.axis('off')
        ax.set_title(f"Year {year} not in forecast window", fontsize=12)
        continue
    idx = np.where(forecast_years == year)[0][0]
    # Only explain if index is valid
    if idx < len(X_latest_scaled):
        exp = lime_explainer.explain_instance(X_latest_scaled[idx], gbm_resid.predict, num_features=10)
        # Manually plot LIME explanation onto the subplot
        exp_list = exp.as_list()
        labels = [e[0] for e in exp_list][::-1] # Reverse for plotting order
        values = [e[1] for e in exp_list][::-1]
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in values]
        ax.barh(labels, values, color=colors, edgecolor='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=1.2)
        ax.set_title(f"LIME Explanation for {year}", fontsize=13, fontweight='bold')
        ax.set_xlabel("Local Contribution", fontsize=11)
        ax.grid(True, axis='x', linestyle='--', alpha=0.4)
    else:
        ax.axis('off')
        ax.set_title(f"Not enough data to explain Year {year}", fontsize=12)

plt.suptitle("LIME Local Explanations Across Key Years", fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig("plots_output/lime_key_years.png", dpi=300, bbox_inches='tight')
plt.savefig("plots_output/lime_key_years.svg", bbox_inches='tight')
plt.show()

# --- SHAP TIME-SERIES FOR ALL 6 CO2 EMISSION FEATURES ---
# Identify CO2-related features
co2_features = [col for col in feature_cols if 'CO2' in col.upper() or 'EMISSION' in col.upper()]

if not co2_features:
    print("\nNo CO2 emission features found in feature_cols. Attempting pattern matching...")
    # Try alternative patterns if exact match doesn't work
    co2_features = [col for col in feature_cols if any(x in col.lower() for x in ['co2', 'emission', 'carbon'])]

if co2_features:
    print(f"\nFound {len(co2_features)} CO2-related features: {co2_features}")
    
    # Prepare SHAP values over time from collected test SHAP values
    all_test_shap_values_list = [np.array(all_test_shap_values_gbm_resid)]
    if all_test_shap_values_rf_resid and not np.all(np.isnan(all_test_shap_values_rf_resid)):
        all_test_shap_values_list.append(np.array(all_test_shap_values_rf_resid))
    if HAS_XGB and all_test_shap_values_xgb_resid and not np.all(np.isnan(all_test_shap_values_xgb_resid)):
        all_test_shap_values_list.append(np.array(all_test_shap_values_xgb_resid))

    # Ensure lists are not empty before stacking and averaging
    if all_test_shap_values_list:
        shap_values_over_time = np.mean(np.stack(all_test_shap_values_list, axis=0), axis=0)
        
        # Create subplot for all CO2 features
        n_co2_features = len(co2_features)
        fig, axes = plt.subplots(n_co2_features, 1, figsize=(14, 4*n_co2_features), sharex=True)
        
        # Handle single feature case
        if n_co2_features == 1:
            axes = [axes]
        
        for idx, co2_feat in enumerate(co2_features):
            feat_idx = feature_cols.index(co2_feat)
            co2_shap_series = shap_values_over_time[:, feat_idx]
            
            ax = axes[idx]
            ax.plot(forecast_years, co2_shap_series, color=COLORS["Hybrid-Stack"], linewidth=2.5, marker='o', markersize=4)
            ax.axhline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.7)
            ax.set_ylabel("SHAP Value", fontsize=11)
            ax.set_title(f"SHAP Time-Series: {co2_feat}", fontsize=12, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
        
        axes[-1].set_xlabel("Year", fontsize=12)
        plt.suptitle("SHAP Time-Series Decomposition for All CO2 Emission Features", fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig("plots_output/shap_timeseries_all_co2.png", dpi=300, bbox_inches='tight')
        plt.savefig("plots_output/shap_timeseries_all_co2.svg", bbox_inches='tight')
        plt.show()
    else:
        print("No SHAP values over time to plot.")
else:
    print("\nWarning: No CO2 emission features detected in your feature set.")
    print(f"Available features: {feature_cols}")
    print("\nPlotting top feature instead...")
    
    # Fallback to top feature if no CO2 features found
    top_feature = fi_df.iloc[0]["Feature"] if not fi_df.empty else None
    if top_feature is not None:
        top_idx = feature_cols.index(top_feature)
        
        # Prepare SHAP values over time
        all_test_shap_values_list = [np.array(all_test_shap_values_gbm_resid)]
        if all_test_shap_values_rf_resid and not np.all(np.isnan(all_test_shap_values_rf_resid)):
            all_test_shap_values_list.append(np.array(all_test_shap_values_rf_resid))
        if HAS_XGB and all_test_shap_values_xgb_resid and not np.all(np.isnan(all_test_shap_values_xgb_resid)):
            all_test_shap_values_list.append(np.array(all_test_shap_values_xgb_resid))

        if all_test_shap_values_list:
            shap_values_over_time = np.mean(np.stack(all_test_shap_values_list, axis=0), axis=0)
            top_shap_series = shap_values_over_time[:, top_idx]
            
            plt.figure(figsize=(14, 5))
            plt.plot(forecast_years, top_shap_series, color=COLORS["Hybrid-Stack"], linewidth=2.5, marker='o', markersize=5)
            plt.axhline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.7)
            plt.title(f"SHAP Time-Series – {top_feature}", fontsize=16, fontweight='bold')
            plt.xlabel("Year", fontsize=13)
            plt.ylabel("SHAP Value (Residual Impact)", fontsize=13)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig("plots_output/shap_timeseries_top_feature.png", dpi=300, bbox_inches='tight')
            plt.savefig("plots_output/shap_timeseries_top_feature.svg", bbox_inches='tight')
            plt.show()

# --- SENSITIVITY ANALYSIS (LEAVE-ONE-OUT) ---
print("\n" + "="*60)
print("SENSITIVITY ANALYSIS (MEAN SHAP IMPACT)")
print("="*60)

sensitivity_results = []
baseline_rmse = np.sqrt(mean_squared_error(y_true_arr_billion, stack_pred_arr_billion))

# Only proceed if fi_df is not empty
if not fi_df.empty:
    for col in feature_cols:
        # Using SHAP values as sensitivity metric (scaled to billions)
        impact = fi_df.loc[fi_df['Feature'] == col, 'Mean_SHAP_Billion'].values[0]
        sensitivity_results.append({'Feature': col, 'Sensitivity_Score_Billion': impact})

    sens_df = pd.DataFrame(sensitivity_results).sort_values(by='Sensitivity_Score_Billion', ascending=False)
    print(sens_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f"\nBaseline RMSE: {baseline_rmse:.4f} Billion USD")
    
    # --- VISUALIZATION: SENSITIVITY ANALYSIS ---
    plt.figure(figsize=(12, 8))
    
    # Create color gradient based on sensitivity
    colors_sens = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sens_df)))
    
    bars = plt.barh(sens_df['Feature'], sens_df['Sensitivity_Score_Billion'], 
                    color=colors_sens, edgecolor='black', linewidth=1.5, alpha=0.85)
    
    plt.xlabel('Sensitivity Score (Billion USD Impact)', fontsize=13, fontweight='bold')
    plt.ylabel('Features', fontsize=13, fontweight='bold')
    plt.title('Feature Sensitivity Analysis - Impact on GDP Predictions', fontsize=15, fontweight='bold', pad=20)
    plt.grid(True, axis='x', linestyle='--', alpha=0.4)
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, sens_df['Sensitivity_Score_Billion'])):
        plt.text(val + 0.03, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}B', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add baseline RMSE annotation
    plt.text(0.98, 0.02, f'Baseline RMSE: {baseline_rmse:.4f}B USD', 
            transform=plt.gca().transAxes, fontsize=11, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
            ha='right', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("plots_output/sensitivity_analysis.png", dpi=300, bbox_inches='tight')
    plt.savefig("plots_output/sensitivity_analysis.svg", bbox_inches='tight')
    plt.show()
    
    # --- VISUALIZATION: TOP 5 FEATURES RADAR CHART ---
    top_5_features = sens_df.head(5)
    
    # Normalize scores to 0-1 scale for radar chart
    max_score = top_5_features['Sensitivity_Score_Billion'].max()
    normalized_scores = top_5_features['Sensitivity_Score_Billion'] / max_score
    
    # Create radar chart
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    # Number of variables
    num_vars = len(top_5_features)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    scores = normalized_scores.tolist()
    
    # Close the plot
    angles += angles[:1]
    scores += scores[:1]
    
    # Plot
    ax.plot(angles, scores, 'o-', linewidth=3, color='#1f77b4', label='Sensitivity Score')
    ax.fill(angles, scores, alpha=0.25, color='#1f77b4')
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top_5_features['Feature'], fontsize=11, fontweight='bold')
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
    
    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.title('Top 5 Most Sensitive Features (Radar View)', 
             fontsize=15, fontweight='bold', pad=30)
    plt.tight_layout()
    plt.savefig("plots_output/sensitivity_radar_top5.png", dpi=300, bbox_inches='tight')
    plt.savefig("plots_output/sensitivity_radar_top5.svg", bbox_inches='tight')
    plt.show()
    
else:
    print("No Feature Importance DataFrame (fi_df) to perform sensitivity analysis.")