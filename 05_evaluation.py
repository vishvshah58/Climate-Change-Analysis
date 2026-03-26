import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import os

def run_phase_5():
    print("\n" + "="*50)
    print("\033[1m   PHASE 5: MODEL EVALUATION STARTED\033[0m")
    print("="*50)

    # 1. Load the predictions generated in Phase 4
    input_path = "data/processed/model_predictions.csv"
    
    if not os.path.exists(input_path):
        print(f"❌ ERROR: {input_path} not found. Please run Phase 4 first.")
        return

    print("\nLoading model predictions...")
    preds_df = pd.read_csv(input_path, index_col='Year')
    
    actual = preds_df['Actual']
    models = ['SARIMA', 'RandomForest', 'XGBoost', 'LSTM']
    
    # Dictionary to hold our metrics
    metrics_data = []

    # 2. Calculate Metrics for each model
    print("Calculating evaluation metrics...")
    for model in models:
        pred = preds_df[model]
        
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        r2 = r2_score(actual, pred)
        mape = mean_absolute_percentage_error(actual, pred) * 100 # Convert to percentage
        
        metrics_data.append({
            'Model': model,
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'R2_Score': round(r2, 4),
            'MAPE (%)': round(mape, 2)
        })

    # 3. Create Comparison Table (DataFrame)
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.set_index('Model', inplace=True)
    
    print("\n\033[1m--- FINAL MODEL COMPARISON ---\033[0m")
    print(metrics_df.to_string())
    
    # Save metrics to CSV for the final report
    metrics_df.to_csv("data/processed/model_evaluation_metrics.csv")
    
    # 4. Identify the Best Model (Based on Lowest RMSE)
    best_model_name = metrics_df['RMSE'].idxmin()
    best_rmse = metrics_df.loc[best_model_name, 'RMSE']
    
    print("\n" + "*"*50)
    print(f"\033[1m 🏆 BEST PERFORMING MODEL: {best_model_name} \033[0m")
    print(f"    (Achieved the lowest RMSE of {best_rmse})")
    print("*"*50)

    # 5. Plot Bar Chart comparing RMSE
    print("\nGenerating RMSE comparison chart...")
    plt.figure(figsize=(10, 6))
    
    # Create bars, highlight the best model in green
    colors = ['green' if model == best_model_name else 'royalblue' for model in metrics_df.index]
    bars = plt.bar(metrics_df.index, metrics_df['RMSE'], color=colors, edgecolor='black', alpha=0.8)
    
    # Add exact values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval}', 
                 ha='center', va='bottom', fontweight='bold')
        
    plt.title('Model Comparison: Root Mean Squared Error (Lower is Better)', fontsize=14, fontweight='bold')
    plt.ylabel('RMSE (Temperature Change °C)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save and show the plot
    plot_path = "outputs/plots/05_rmse_comparison.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Comparison chart saved to {plot_path}")

    print("\n" + "="*50)
    print("   PHASE 5 COMPLETE")
    print("="*50)

if __name__ == "__main__":
    run_phase_5()
