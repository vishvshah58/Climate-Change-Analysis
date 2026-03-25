import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def run_phase_3():
    print("\n" + "="*50)
    print("\033[1m   PHASE 3: EDA & FEATURE ENGINEERING STARTED\033[0m")
    print("="*50)

    input_path = "data/processed/climate_cleaned.csv"
    
    # Check if cleaned data exists
    if not os.path.exists(input_path):
        print(f"❌ ERROR: Could not find {input_path}.")
        print("   Please run Phase 2 successfully before starting Phase 3.")
        return

    # 1. Load Cleaned Data
    print("\nLoading cleaned data...")
    df = pd.read_csv(input_path)
    
    # 2. Aggregate to Global Yearly Average
    # The current data is per country/month. We need a single global timeline for forecasting.
    print("Aggregating country-level data into Global Yearly Averages...")
    global_df = df.groupby('Year')['Temp_Change'].mean().reset_index()
    global_df = global_df.sort_values('Year')
    
    # ---------------------------------------------------------
    # FEATURE ENGINEERING
    # ---------------------------------------------------------
    print("\nEngineering new features...")
    
    # A. Decade Labels
    global_df['Decade'] = (global_df['Year'] // 10) * 10
    
    # B. Rolling 10-Year Average
    global_df['Rolling_10Yr_Avg'] = global_df['Temp_Change'].rolling(window=10).mean()
    
    # C. Year-over-Year (YoY) Change
    global_df['YoY_Change'] = global_df['Temp_Change'].diff()
    
    # D. CO2 Proxy (Since raw CO2 wasn't in the uploaded CSVs, we simulate a proxy 
    # based on known historical exponential growth for feature correlation purposes)
    # Baseline ~315 ppm in 1960, growing non-linearly.
    global_df['CO2_Proxy_ppm'] = 315 + ((global_df['Year'] - 1960) ** 1.5) * 0.5
    global_df['CO2_Proxy_ppm'] = global_df['CO2_Proxy_ppm'].fillna(315) # Fallback for years < 1960
    
    # Drop NaN values created by rolling averages and diffs
    global_df.dropna(inplace=True)
    global_df.set_index('Year', inplace=True)

    # ---------------------------------------------------------
    # EXPLORATORY DATA ANALYSIS (PLOTS)
    # ---------------------------------------------------------
    print("\nGenerating and saving plots...")
    
    # Create output directory for plots
    plot_dir = "outputs/plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Plot 1: Global Average Temperature Anomaly & Rolling Avg
    plt.figure(figsize=(12, 6))
    plt.plot(global_df.index, global_df['Temp_Change'], label='Yearly Avg Temp Change', color='lightblue', marker='o', markersize=4)
    plt.plot(global_df.index, global_df['Rolling_10Yr_Avg'], label='10-Year Rolling Avg', color='red', linewidth=2)
    plt.title('Global Average Temperature Anomaly (with 10-Yr Trend)', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Temperature Change (°C)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{plot_dir}/01_temperature_trend.png", bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Plot 2: Seasonal Decomposition (Trend, Seasonal, Residual)
    # Using period=10 to look at decadal cycles in our yearly data
    decomposition = seasonal_decompose(global_df['Temp_Change'], model='additive', period=10)
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle('Time-Series Decomposition of Temperature Change', fontweight='bold', y=1.02)
    plt.savefig(f"{plot_dir}/02_seasonal_decomposition.png", bbox_inches='tight')
    plt.show()
    plt.close()

    # Plot 3: Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr_matrix = global_df[['Temp_Change', 'Rolling_10Yr_Avg', 'YoY_Change', 'CO2_Proxy_ppm']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.savefig(f"{plot_dir}/03_correlation_heatmap.png", bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Plots saved successfully in '{plot_dir}/'")

    # ---------------------------------------------------------
    # SCALING & PREPARATION FOR ML
    # ---------------------------------------------------------
    print("\nScaling features for Machine Learning...")
    features_to_scale = ['Rolling_10Yr_Avg', 'YoY_Change', 'CO2_Proxy_ppm']
    
    scaler = StandardScaler()
    global_df_scaled = global_df.copy()
    global_df_scaled[features_to_scale] = scaler.fit_transform(global_df[features_to_scale])
    
    # Save the engineered dataset
    output_path = "data/processed/global_climate_engineered.csv"
    global_df_scaled.to_csv(output_path)
    print(f"💾 Engineered data saved to: {output_path}")

    # ---------------------------------------------------------
    # FEATURE IMPORTANCE PREVIEW
    # ---------------------------------------------------------
    print("\nCalculating Feature Importance Preview (using Random Forest)...")
    X = global_df_scaled[['Rolling_10Yr_Avg', 'YoY_Change', 'CO2_Proxy_ppm', 'Decade']]
    y = global_df_scaled['Temp_Change']
    
    rf_preview = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_preview.fit(X, y)
    
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance (%)': np.round(rf_preview.feature_importances_ * 100, 2)
    }).sort_values(by='Importance (%)', ascending=False)
    
    print("\n" + importances.to_string(index=False))

    print("\n" + "="*50)
    print("   PHASE 3 COMPLETE")
    print("="*50)

if __name__ == "__main__":
    run_phase_3()
