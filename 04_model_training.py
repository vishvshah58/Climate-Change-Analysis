import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import warnings

# Machine Learning & Time Series Libraries
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Deep Learning Libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

warnings.filterwarnings("ignore")

def run_phase_4():
    print("\n" + "="*50)
    print("\033[1m   PHASE 4: MODEL TRAINING STARTED\033[0m")
    print("="*50)

    input_path = "data/processed/global_climate_engineered.csv"
    
    if not os.path.exists(input_path):
        print(f"❌ ERROR: {input_path} not found. Run Phase 3 first.")
        return

    # 1. Load Data
    print("\nLoading engineered data...")
    df = pd.read_csv(input_path, index_col='Year')
    
    # Define Features (X) and Target (y)
    features = ['Rolling_10Yr_Avg', 'YoY_Change', 'CO2_Proxy_ppm', 'Decade']
    target = 'Temp_Change'

    # 2. Train/Test Split (Chronological: 80% Train, 20% Test)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    print(f"Data split: {len(train_df)} Train years, {len(test_df)} Test years.")

    # Create directories for models and plots
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    
    # Dictionary to store test predictions for Phase 5 evaluation
    predictions = {'Actual': y_test.values, 'Year': test_df.index}

    # ---------------------------------------------------------
    # MODEL 1: SARIMA (Time-Series)
    # ---------------------------------------------------------
    print("\n[1/4] Training SARIMA Model...")
    # SARIMA only uses the target variable (univariate forecasting for baseline)
    sarima_model = SARIMAX(train_df[target], order=(1, 1, 1), seasonal_order=(1, 1, 1, 10))
    sarima_fit = sarima_model.fit(disp=False)
    
    # Predict
    sarima_preds = sarima_fit.forecast(steps=len(test_df))
    predictions['SARIMA'] = sarima_preds.values
    
    # Save Model
    joblib.dump(sarima_fit, "models/sarima_model.pkl")
    print("✅ SARIMA trained and saved.")

    # ---------------------------------------------------------
    # MODEL 2: Random Forest Regressor
    # ---------------------------------------------------------
    print("\n[2/4] Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    rf_preds = rf_model.predict(X_test)
    predictions['RandomForest'] = rf_preds
    
    joblib.dump(rf_model, "models/random_forest_model.pkl")
    print("✅ Random Forest trained and saved.")

    # ---------------------------------------------------------
    # MODEL 3: XGBoost Regressor
    # ---------------------------------------------------------
    print("\n[3/4] Training XGBoost...")
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    xgb_preds = xgb_model.predict(X_test)
    predictions['XGBoost'] = xgb_preds
    
    joblib.dump(xgb_model, "models/xgboost_model.pkl")
    print("✅ XGBoost trained and saved.")

    # ---------------------------------------------------------
    # MODEL 4: LSTM Neural Network
    # ---------------------------------------------------------
    print("\n[4/4] Training LSTM Neural Network...")
    # LSTM requires input shape: [samples, time_steps, features]
    X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    lstm_model = Sequential([
        LSTM(50, activation='relu', input_shape=(1, X_train.shape[1])),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    
    # Fit the model quietly (verbose=0)
    lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=16, verbose=0)
    
    # Predict (flatten flattens the 2D output array to 1D)
    lstm_preds = lstm_model.predict(X_test_lstm, verbose=0).flatten()
    predictions['LSTM'] = lstm_preds
    
    lstm_model.save("models/lstm_model.h5")
    print("✅ LSTM trained and saved.")

    # ---------------------------------------------------------
    # SAVE PREDICTIONS & PLOT RESULTS
    # ---------------------------------------------------------
    print("\nGenerating and saving actual vs predicted plots...")
    
    # Save predictions to a CSV so Phase 5 can evaluate them easily
    preds_df = pd.DataFrame(predictions)
    preds_df.set_index('Year', inplace=True)
    preds_df.to_csv("data/processed/model_predictions.csv")
    
    # Plot Actual vs Predicted for EACH model separately
    models_to_plot = {
        'SARIMA': ('outputs/plots/04_model1_sarima.png', 'blue'),
        'RandomForest': ('outputs/plots/04_model2_rf.png', 'green'),
        'XGBoost': ('outputs/plots/04_model3_xgboost.png', 'red'),
        'LSTM': ('outputs/plots/04_model4_lstm.png', 'purple')
    }
    
    for model_name, (plot_path, color) in models_to_plot.items():
        plt.figure(figsize=(10, 6))
        
        # Plot historical training data for context
        plt.plot(train_df.index, y_train, label='Training Data (Actual)', color='gray', alpha=0.5)
        
        # Plot actual test data
        plt.plot(test_df.index, y_test, label='Test Data (Actual)', color='black', linewidth=2)
        
        # Plot model predictions
        plt.plot(test_df.index, preds_df[model_name], label=f'{model_name} Prediction', color=color, linestyle='--', linewidth=2)
        
        plt.title(f'{model_name}: Actual vs Predicted Temperature Change', fontsize=14, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Temperature Change (°C)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        
        plt.savefig(plot_path, bbox_inches='tight')
        plt.show()  # Displays the individual chart in Colab
        plt.close()
        
        print(f"✅ Prediction plot for {model_name} saved to {plot_path}")

    print(f"\n✅ Test predictions saved to data/processed/model_predictions.csv")
    
    print("\n" + "="*50)
    print("   PHASE 4 COMPLETE")
    print("="*50)

if __name__ == "__main__":
    run_phase_4()
