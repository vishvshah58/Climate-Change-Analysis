import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

def run_phase_6():
    print("\n" + "="*50)
    print("\033[1m   PHASE 6: FUTURE PREDICTION (2025–2100)\033[0m")
    print("="*50)

    metrics_path = "data/processed/model_evaluation_metrics.csv"
    data_path = "data/processed/climate_cleaned.csv"
    
    if not os.path.exists(metrics_path) or not os.path.exists(data_path):
        print("❌ ERROR: Missing necessary files. Ensure Phases 2, 4, and 5 are complete.")
        return

    # 1. Identify the Best Model
    metrics_df = pd.read_csv(metrics_path, index_col='Model')
    best_model_name = metrics_df['RMSE'].idxmin()
    best_rmse = metrics_df.loc[best_model_name, 'RMSE']
    
    print(f"\n🏆 Using Best Model: \033[1m{best_model_name}\033[0m (RMSE: {best_rmse})")

    # 2. Rebuild Historical Unscaled Data (to seed the future predictions)
    print("Rebuilding historical baseline for future extrapolation...")
    clean_df = pd.read_csv(data_path)
    hist_df = clean_df.groupby('Year')['Temp_Change'].mean().reset_index().set_index('Year')
    
    hist_df['Rolling_10Yr_Avg'] = hist_df['Temp_Change'].rolling(window=10).mean()
    hist_df['YoY_Change'] = hist_df['Temp_Change'].diff()
    # Apply the same CO2 proxy logic used in Phase 3
    hist_df['CO2_Proxy_ppm'] = 315 + ((hist_df.index - 1960) ** 1.5) * 0.5
    hist_df['CO2_Proxy_ppm'] = hist_df['CO2_Proxy_ppm'].fillna(315)
    hist_df.dropna(inplace=True)

    # Fit our Scaler on historical data so we can scale future data
    scaler = StandardScaler()
    scaler.fit(hist_df[['Rolling_10Yr_Avg', 'YoY_Change', 'CO2_Proxy_ppm']])

    # 3. Predict the Future (2025 - 2100)
    print(f"Generating future timeline to 2100 using {best_model_name}...")
    future_years = list(range(2025, 2101))
    future_preds = []
    lower_bounds = []
    upper_bounds = []

    # Store a running list of temperatures to calculate future rolling averages
    running_temps = list(hist_df['Temp_Change'].values)

    if best_model_name == 'SARIMA':
        # SARIMA has built-in forecasting and confidence intervals
        sarima_model = joblib.load("models/sarima_model.pkl")
        forecast = sarima_model.get_forecast(steps=len(future_years))
        
        future_preds = forecast.predicted_mean.values
        conf_int = forecast.conf_int()
        lower_bounds = conf_int.iloc[:, 0].values
        upper_bounds = conf_int.iloc[:, 1].values
    else:
        # For ML models (Random Forest, XGBoost, LSTM), we do step-by-step autoregression
        if best_model_name == 'LSTM':
            from tensorflow.keras.models import load_model
            model = load_model("models/lstm_model.h5")
        else:
            file_map = {'RandomForest': 'random_forest_model.pkl', 'XGBoost': 'xgboost_model.pkl'}
            model = joblib.load(f"models/{file_map[best_model_name]}")

        for year in future_years:
            # 1. Calculate future features based on previous predictions
            rolling_avg = np.mean(running_temps[-10:])
            yoy_change = running_temps[-1] - running_temps[-2]
            co2_proxy = 315 + ((year - 1960) ** 1.5) * 0.5
            decade = (year // 10) * 10
            
            # 2. Scale the features
            scaled_feats = scaler.transform([[rolling_avg, yoy_change, co2_proxy]])
            
            # 3. Format input for the specific model
            X_future = pd.DataFrame({
                'Rolling_10Yr_Avg': [scaled_feats[0][0]],
                'YoY_Change': [scaled_feats[0][1]],
                'CO2_Proxy_ppm': [scaled_feats[0][2]],
                'Decade': [decade]
            })

            if best_model_name == 'LSTM':
                X_future_lstm = X_future.values.reshape((1, 1, X_future.shape[1]))
                pred = model.predict(X_future_lstm, verbose=0).flatten()[0]
            else:
                pred = model.predict(X_future)[0]
            
            future_preds.append(pred)
            running_temps.append(pred)
            
            # Approximate confidence intervals for ML using RMSE
            # 1.96 * RMSE gives approx 95% confidence interval assuming normal distribution of errors
            margin_of_error = 1.96 * best_rmse 
            lower_bounds.append(pred - margin_of_error)
            upper_bounds.append(pred + margin_of_error)

    # Create a DataFrame for future data
    future_df = pd.DataFrame({
        'Year': future_years,
        'Predicted_Temp_Change': future_preds,
        'Lower_CI': lower_bounds,
        'Upper_CI': upper_bounds
    }).set_index('Year')

    # 4. Save Future Predictions
    future_df.to_csv("data/processed/future_predictions_2100.csv")

    # 5. Print Decade-by-Decade Summary Table
    print("\n\033[1m--- PREDICTED TEMPERATURE CHANGE BY DECADE ---\033[0m")
    future_df['Decade'] = (future_df.index // 10) * 10
    decade_summary = future_df.groupby('Decade')['Predicted_Temp_Change'].mean().round(3)
    
    summary_table = pd.DataFrame({
        'Decade': decade_summary.index,
        'Avg Temp Change (°C)': decade_summary.values
    })
    print(summary_table.to_string(index=False))

    # 6. Plot the Final Master Chart
    print("\nGenerating final historical + future prediction chart...")
    plt.figure(figsize=(15, 8))
    
    # Plot historical data
    plt.plot(hist_df.index, hist_df['Temp_Change'], label='Historical Actuals (to 2024)', color='black', linewidth=2)
    
    # Plot future predictions
    plt.plot(future_df.index, future_df['Predicted_Temp_Change'], label=f'Future Prediction ({best_model_name})', color='red', linestyle='--', linewidth=2.5)
    
    # Plot uncertainty bands
    plt.fill_between(future_df.index, future_df['Lower_CI'], future_df['Upper_CI'], color='red', alpha=0.15, label='95% Confidence Interval')

    # Add milestone annotations
    milestones = [2030, 2050, 2075, 2100]
    for year in milestones:
        if year in future_df.index:
            val = future_df.loc[year, 'Predicted_Temp_Change']
            plt.axvline(x=year, color='gray', linestyle=':', alpha=0.5)
            plt.scatter(year, val, color='darkred', zorder=5)
            plt.annotate(f"{year}\n+{val:.2f}°C", 
                         (year, val), 
                         textcoords="offset points", 
                         xytext=(0, 15), 
                         ha='center', 
                         fontsize=10, 
                         fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.title(f'Global Temperature Change Prediction (Present to 2100) using {best_model_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Temperature Change Anomaly (°C)', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save and show
    os.makedirs("outputs/plots", exist_ok=True)
    plot_path = "outputs/plots/06_future_forecast_2100.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300) # dpi=300 for high quality report
    plt.show()

    print(f"\n✅ Future forecast chart saved to {plot_path}")
    print("✅ Future data saved to data/processed/future_predictions_2100.csv")

    print("\n" + "="*50)
    print("   PHASE 6 COMPLETE")
    print("="*50)

if __name__ == "__main__":
    run_phase_6()
