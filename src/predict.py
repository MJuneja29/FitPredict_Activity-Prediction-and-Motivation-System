# ==============================================================================
# Daily Forecast Model - Final Evaluation Script
#
# Project:      IIT Ropar Capstone Project
# Script:       evaluate_forecaster.py
#
# ------------------------------------------------------------------------------
#
# Purpose:
#
# This script is dedicated to the rigorous and systematic evaluation of the
# trained daily calorie forecast model. Its primary goal is to measure the
# model's real-world performance on a dedicated, unseen test dataset.
#
# ------------------------------------------------------------------------------
#
# Methodology:
#
# The script simulates the process of making a forecast at a fixed time of day
# (e.g., 3 PM) for every single day available in the test set. It then:
#
#   1. Loads the pre-trained LSTM model, scalers, and the exact `test_data.csv`
#      that was separated during the training phase.
#
#   2. For each unique day in the test set, it uses the historical data up to the
#      simulation time to generate a full-day calorie projection and a
#      YES/NO verdict on meeting the daily goal.
#
#   3. It compares this forecast against the actual, ground-truth outcome for that
#      day (which is known because we are using historical data).
#
#   4. Finally, it aggregates the results from all simulated days and calculates a
#      comprehensive suite of evaluation metrics, including:
#      - Regression Metrics (MAE, RMSE) to evaluate the accuracy of the
#        projected calorie number.
#      - Classification Metrics (Accuracy, Precision, Recall, F1-Score,
#        and a Confusion Matrix) to evaluate the reliability of the
#        YES/NO verdict.
#
# ==============================================================================

import pandas as pd
import numpy as np
from tensorflow.keras import models
from keras.metrics import mean_squared_error as mse
import joblib
import os
from datetime import time, datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

def run_daily_forecast(model, history_df, daily_target, feature_cols,
                       feature_scaler, target_scaler, timesteps=24):
    """
    A simplified prediction function for the evaluation loop.
    It takes the history up to a certain point and forecasts the rest of the day.
    """
    calories_burned_so_far = history_df[history_df['ActivityHour'].dt.date == history_df.iloc[-1].ActivityHour.date()]['Calories'].sum()
    
    # The history_window is the last `timesteps` of data provided
    history_window = history_df.tail(timesteps)[feature_cols].values
    predicted_future_calories = []
    
    last_known_hour = history_df.iloc[-1]['ActivityHour'].hour
    
    for hour in range(last_known_hour + 1, 24):
        scaled_window = feature_scaler.transform(history_window)
        X_pred = np.expand_dims(scaled_window, axis=0)
        prediction_scaled = model.predict(X_pred, verbose=0)
        prediction_descaled = target_scaler.inverse_transform(prediction_scaled)
        predicted_calories = max(0, prediction_descaled[0][0])
        predicted_future_calories.append(predicted_calories)
        
        new_row = history_window[-1, :].copy()
        try:
            calories_index = feature_cols.index('Calories')
            new_row[calories_index] = predicted_calories
        except ValueError: pass
        history_window = np.vstack([history_window[1:], new_row])

    projected_total_calories = calories_burned_so_far + sum(predicted_future_calories)
    verdict = projected_total_calories >= daily_target
    return projected_total_calories, verdict

# ==============================================================================
# MAIN EVALUATION SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    
    # --- Step 1: Load all our pre-trained assets and data ---
    print("--- Loading pre-trained model, assets, and all data... ---")
    load_dir = "src"

    custom_objects = {'mse': mse}

    try:
        model = models.load_model(os.path.join(load_dir, "calorie_prediction_model.h5"), custom_objects=custom_objects)
        feature_scaler = joblib.load(os.path.join(load_dir, "feature_scaler.joblib"))
        target_scaler = joblib.load(os.path.join(load_dir, "target_scaler.joblib"))
        feature_columns = joblib.load(os.path.join(load_dir, "feature_columns.joblib"))
        
        # THIS IS THE KEY: We load BOTH the full dataset and the specific test set.
        full_df = pd.read_csv("Data\\final_hourly.csv", parse_dates=['ActivityHour'])
        test_df = pd.read_csv(os.path.join(load_dir, "test_data.csv"), parse_dates=['ActivityHour'])

    except FileNotFoundError:
        print(f"ERROR: Asset files not found. Please run `train_model.py` first.")
        exit()
    print("Assets loaded successfully.")
    
    # --- Step 2: Set up and run the evaluation simulation ---
    daily_calorie_target = 2500
    simulation_time = time(15, 0) # 3 PM
    
    true_outcomes, predicted_outcomes = [], []
    true_totals, predicted_totals = [], []

    # The evaluation will be based on the unique days present in our test set
    test_df['Date'] = test_df['ActivityHour'].dt.date
    unique_test_days = test_df.groupby(['Id', 'Date'])

    print(f"\n--- Running evaluation simulation at {simulation_time.strftime('%I:%M %p')} for each day in the test set... ---")
    
    for (user_id, date), day_in_test_set in unique_test_days:
        # --- Ground Truth ---
        # Get the actual final calorie total for this day from the full dataset
        full_day_data = full_df[(full_df['Id'] == user_id) & (full_df['ActivityHour'].dt.date == date)]
        if full_day_data.empty:
            continue
        actual_total = full_day_data['Calories'].sum()
        true_outcome = actual_total >= daily_calorie_target
        
        # --- Prediction Setup ---
        # To make a prediction at 3 PM, we need the 24 hours of history BEFORE 3 PM on that day.
        # We get this from the FULL dataset to ensure we have enough history.
        simulation_datetime = datetime.combine(date, simulation_time)
        history_for_prediction = full_df[
            (full_df['Id'] == user_id) & 
            (full_df['ActivityHour'] < simulation_datetime)
        ]
        
        # We must have enough historical data to make a prediction
        if len(history_for_prediction) < 24:
            continue
            
        # Run our forecaster using the historical data
        projected_total, predicted_outcome = run_daily_forecast(
            model, history_for_prediction, daily_calorie_target, feature_columns,
            feature_scaler, target_scaler, 24
        )
        
        # Store the results for final calculation
        true_outcomes.append(true_outcome)
        predicted_outcomes.append(predicted_outcome)
        true_totals.append(actual_total)
        predicted_totals.append(projected_total)

    # --- Step 3: Calculate and display all evaluation metrics ---
    print("\n========================================")
    print("DAILY FORECAST EVALUATION REPORT")
    print("========================================")
    
    if not true_outcomes:
        print("Could not generate any test predictions. Ensure the test set contains days with sufficient history before the simulation time.")
    else:
        # --- Regression Metrics ---
        print("\n--- Regression Performance (Projected Daily Total) ---")
        mae = mean_absolute_error(true_totals, predicted_totals)
        rmse = np.sqrt(mean_squared_error(true_totals, predicted_totals))
        print(f"Mean Absolute Error (MAE): {mae:.2f} calories")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f} calories")

        # --- Classification Metrics ---
        print("\n--- Classification Performance (Will Meet Goal?) ---")
        accuracy = accuracy_score(true_outcomes, predicted_outcomes)
        precision = precision_score(true_outcomes, predicted_outcomes, zero_division=0)
        recall = recall_score(true_outcomes, predicted_outcomes, zero_division=0)
        f1 = f1_score(true_outcomes, predicted_outcomes, zero_division=0)
        
        print(f"Accuracy:  {accuracy:.2%} (Overall correct verdicts)")
        print(f"Precision: {precision:.2%} (When we predict YES, how often we're right)")
        print(f"Recall:    {recall:.2%} (Of all actual YES cases, how many we found)")
        print(f"F1-Score:  {f1:.2%} (Balanced score for precision and recall)")

        # --- Confusion Matrix ---
        print("\n--- Confusion Matrix ---")
        cm = confusion_matrix(true_outcomes, predicted_outcomes)
        
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted NO (Fail)', 'Predicted YES (Succeed)'], 
                    yticklabels=['Actual NO (Fail)', 'Actual YES (Succeed)'])
        plt.ylabel('Actual Outcome')
        plt.xlabel('Predicted Outcome')
        plt.title('Confusion Matrix for Daily Goal Forecast')
        plt.show()