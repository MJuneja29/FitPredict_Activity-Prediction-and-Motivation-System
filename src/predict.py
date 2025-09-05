# File: predict.py
import pandas as pd
import numpy as np
from tensorflow.keras import models
import joblib

# We only need the `predict_next_hour_calories` function here.
# It uses the loaded model and scalers to do its job.
def predict_next_hour_calories(model, full_data_df, user_id, feature_cols, feature_scaler, target_scaler, timesteps=24):
    """
    Predicts the next hour's calories for a specific user using loaded assets.
    """
    print(f"\n--- Generating prediction for user {user_id} ---")
    # Make sure Id column is a string for comparison
    full_data_df['Id'] = full_data_df['Id'].astype(str)
    user_data = full_data_df[full_data_df["Id"] == str(user_id)]
    
    if len(user_data) < timesteps:
        raise ValueError(f"User {user_id} does not have enough data ({len(user_data)} hours). Need at least {timesteps}.")
    
    last_sequence_df = user_data.tail(timesteps)[feature_cols]
    
    last_sequence_scaled = feature_scaler.transform(last_sequence_df)
    
    X_pred = np.expand_dims(last_sequence_scaled, axis=0)
    
    prediction_scaled = model.predict(X_pred, verbose=0)
    
    prediction_descaled = target_scaler.inverse_transform(prediction_scaled)
    
    return prediction_descaled[0][0]

# ==============================================================================
# MAIN PREDICTION SCRIPT
# ==============================================================================
if __name__ == "__main__":
    
    print("--- Loading pre-trained model and assets ---")
    
    try:
        # Load all the assets we saved from the training script
        model = models.load_model("calorie_prediction_model.h5")
        feature_scaler = joblib.load("feature_scaler.joblib")
        target_scaler = joblib.load("target_scaler.joblib")
        feature_columns = joblib.load("feature_columns.joblib")
        
        # Load the full dataset to get the user's history from
        full_df = pd.read_csv("your_data.csv") # I've changed the path for simplicity

    except FileNotFoundError:
        print("ERROR: Model, scaler, or feature list files not found.")
        print("Please run the `train_model.py` script first to create these files.")
        exit()
        
    print("Model and assets loaded successfully.")

    # --- Now you can make predictions instantly ---
    
    # Pick a user to predict for (e.g., the last user in the dataframe)
    user_to_predict = str(full_df['Id'].iloc[-1])

    try:
        predicted_calories = predict_next_hour_calories(
            model=model,
            full_data_df=full_df,
            user_id=user_to_predict,
            feature_cols=feature_columns,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            timesteps=24
        )
        print("\n========================================")
        print("FINAL PREDICTION")
        print("========================================")
        print(f"Predicted calories for the next hour for user '{user_to_predict}': {predicted_calories:.2f} calories")
    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")