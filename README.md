
# Climate Change Analysis & Prediction System:
# Project OverviewThis project is an end-to-end Machine Learning and Time-Series forecasting pipeline designed to analyze historical global temperature changes and predict future climate   trends up to the year 2100. It processes raw climate datasets, engineers features (including CO2 proxies and rolling averages), and evaluates four distinct predictive models to find the most accurate forecaster.
# Datasets Used:
 The data is sourced from global environmental repositories, including:
# FAOSTAT Environment Temperature Change Data
# Global Land Temperatures By Country/City/State
# Global Weather Repository
# Project Pipeline (Phases)
# This project was built step-by-step in 6 main phases:
# Data Acquisition: Programmatic loading of multiple CSV datasets.
# Data Cleaning: Melting wide-format data, interpolating missing values, and formatting timelines.
# Exploratory Data Analysis (EDA): Visualizing historical trends, seasonal decomposition, and calculating feature correlations.
# Model Training: Training four distinct algorithms on an 80/20 chronological split:
   # SARIMA (Classical Time-Series)
   # Random Forest Regressor (Tree-based)
   # XGBoost Regressor (Gradient Boosting)
   # LSTM Neural Network (Deep Learning
# Model Evaluation: Comparing models using MAE, RMSE, R² Score, and MAPE to identify the highest performing algorithm.
# Future Forecasting: Using the optimal model to extrapolate temperature anomalies decade-by-decade until 2100, including 95% confidence intervals.

# Key ResultsEngineered a master timeline showing global temperature anomalies.Evaluated classical statistical models vs. modern deep learning approaches.Generated a continuous forecast to      2100 with marked milestones (2030, 2050, 2075, 2100).All visual outputs are automatically saved to the outputs/plots/ directory.Created as a Final Year Data Science & Machine Learning         Project.
