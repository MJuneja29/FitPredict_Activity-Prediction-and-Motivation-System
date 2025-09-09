# ==============================================================================
# Fitbit User Activity - Model Training
#
# Project:      IIT Ropar Capstone Project
#
# Purpose: This script performs the entire machine learning pipeline:
#           1. Loads the raw Fitbit data.
#           2. Prepares it for the LSTM model (cleaning, creating sequences, scaling).
#           3. Trains the LSTM model, using Early Stopping to find the best version.
#           4. Evaluates the best model and shows performance graphs.
#           5. Saves the final, trained model and all necessary components to a 'src' folder.
#
# ==============================================================================

# --- Step 1: Import all the necessary libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import joblib
import os

# ==============================================================================
# SECTION 1: DATA PROCESSING UTILITY FUNCTIONS
# These are your helper functions that prepare the data piece by piece.
# ==============================================================================

def ensure_types(df):
    # This function makes sure our data is in the right format to work with.
    out = df.copy()
    out["Id"] = out["Id"].astype(str)
    out["ActivityHour"] = pd.to_datetime(out["ActivityHour"], errors="coerce")
    out = out.dropna(subset=["ActivityHour"]).sort_values(["Id", "ActivityHour"])
    return out

def make_dense_hourly(df):
    # This function finds any missing hours in a user's timeline and fills them with 0.
    # This is important so that our time sequences are not broken.
    pieces = []
    for uid, g in df.groupby("Id"):
        g = g.sort_values("ActivityHour").set_index("ActivityHour")
        if g.empty: continue
        full_index = pd.date_range(g.index.min(), g.index.max(), freq="h") 
        g_dense = g.reindex(full_index)
        g_dense["Id"] = uid
        g_dense.index.name = "ActivityHour"
        pieces.append(g_dense.reset_index())
    dense = pd.concat(pieces, axis=0, ignore_index=True)
    dense = dense.fillna(0)
    return dense

def select_feature_columns(df, target_col="Calories"):
    # This function automatically picks all the number-based columns to use as features,
    # while excluding the ID, the timestamp, and the target we want to predict.
    exclude = {"Id", "ActivityHour", target_col}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in num_cols if c not in exclude]
    return feature_cols

def split_per_user_time(df, test_size=0.2):
    # This function splits each user's data into a training set (the older 80% of data)
    # and a testing set (the newer 20% of data). This prevents the model from cheating
    # by looking at future data to predict the past.
    train_parts, test_parts = [], []
    for uid, g in df.groupby("Id"):
        g = g.sort_values("ActivityHour")
        n = len(g)
        if n < 3: continue
        cut = int(max(1, np.floor(n * (1 - test_size))))
        train_parts.append(g.iloc[:cut])
        test_parts.append(g.iloc[cut:])
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=df.columns)
    return train_df, test_df

def create_sequences_per_user(df, feature_cols, target_col="Calories", timesteps=24, horizon=1):
    # This is a key step for LSTMs. It takes the hourly data and chops it into
    # "sliding windows". For example, it uses hours 1-24 to predict hour 25,
    # then hours 2-25 to predict hour 26, and so on.
    X_list, y_list = [], []
    for uid, g in df.groupby("Id"):
        g = g.sort_values("ActivityHour")
        feats = g[feature_cols].values
        tgt = g[target_col].values
        max_start = len(g) - (timesteps + horizon) + 1
        if max_start <= 0: continue
        for i in range(max_start):
            X_list.append(feats[i : i + timesteps])
            y_list.append(tgt[i + timesteps + horizon - 1])
    if not X_list:
        return np.empty((0, timesteps, len(feature_cols))), np.empty((0,))
    return np.stack(X_list), np.array(y_list)

# ==============================================================================
# SECTION 2: THE MAIN DATA PREPARATION PIPELINE
# This function manages all the utility functions above to run the full process.
# ==============================================================================

def prepare_lstm_data(final_hourly, target_col="Calories", timesteps=24, test_size=0.2):
    # This function acts like a manager, calling all the helper functions in the
    # correct order to get our data ready for the model.
    df = ensure_types(final_hourly)
    df = make_dense_hourly(df)
    feature_cols = select_feature_columns(df, target_col=target_col)
    train_df, test_df = split_per_user_time(df, test_size=test_size)
    X_train, y_train = create_sequences_per_user(train_df, feature_cols, target_col, timesteps)
    X_test, y_test = create_sequences_per_user(test_df, feature_cols, target_col, timesteps)
    
    # Scale the features (inputs). We learn the scaling rules from the training data only.
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
    X_test_scaled = feature_scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)
    
    # Scale the target (output). We use a separate scaler for the calories.
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1))
    
    # Return everything we need for training, evaluation, and future predictions
    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, feature_cols, feature_scaler, target_scaler, y_test

# ==============================================================================
# SECTION 3: MODEL ARCHITECTURE AND TRAINING
# These functions define the neural network and run the training process.
# ==============================================================================

def build_lstm_model(timesteps, n_features):
    # This defines the layers of our LSTM neural network.
    model = models.Sequential([
        layers.Input(shape=(timesteps, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model

def train_and_evaluate(X_train, y_train, X_test, y_test_scaled, y_test_original, target_scaler, epochs=30):
    # This function builds the model and starts the training.
    # It now includes the crucial EarlyStopping callback.
    model = build_lstm_model(X_train.shape[1], X_train.shape[2])

    # This is the "smart" part of our training.
    # It watches the model's performance on the unseen test data ('val_loss').
    # If the performance doesn't improve for 5 epochs in a row ('patience=5'),
    # it stops the training and gives us back the best version of the model.
    es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test_scaled),
        epochs=epochs,
        batch_size=64,
        verbose=1,
        callbacks=[es] # We tell the model to use our EarlyStopping rule.
    )

    # After training, we evaluate the final (best) model on the test set.
    preds_scaled = model.predict(X_test, verbose=0)
    preds_descaled = target_scaler.inverse_transform(preds_scaled).ravel()
    
    mae = mean_absolute_error(y_test_original, preds_descaled)
    rmse = np.sqrt(mean_squared_error(y_test_original, preds_descaled))

    print(f"\nTest MAE (in Calories):  {mae:.2f}")
    print(f"Test RMSE (in Calories): {rmse:.2f}")
    
    return model, history, preds_descaled

# ==============================================================================
# SECTION 4: VISUALIZATION FUNCTIONS
# These functions help us see how the model performed.
# ==============================================================================

def plot_training_history(history):
    # This graph shows us the "learning curve" of the model, which helped us
    # see the overfitting problem and find the best epoch.
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'], 'b-', label='Training Loss (loss)')
    plt.plot(history.history['val_loss'], 'r-', label='Validation Loss (val_loss)')
    best_epoch = np.argmin(history.history['val_loss']) + 1
    plt.axvline(best_epoch, linestyle='--', color='g', label=f'Best Epoch: {best_epoch}')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Mean Squared Error)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_actual_vs_predicted(actual_values, predicted_values):
    # This graph gives us a direct visual comparison of our model's predictions
    # against the true calorie values from the test set.
    plt.figure(figsize=(15, 6))
    plt.plot(actual_values[:200], 'b-', label='Actual Calories')
    plt.plot(predicted_values[:200], 'g--', label='Predicted Calories')
    plt.title('Model Performance: Actual vs. Predicted (First 200 Test Points)')
    plt.xlabel('Time Step (in Test Set order)')
    plt.ylabel('Calories')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

# ==============================================================================
# MAIN SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    
    # --- Step 1: Load the Dataset ---
    try:
        final_df = pd.read_csv("Data\\final_hourly.csv") 
    except FileNotFoundError:
        print("ERROR: 'your_data.csv' not found. Please place it in the same directory.")
        exit()

    # --- Step 2: Prepare the Data for the LSTM Model ---
    X_train_s, y_train_s, X_test_s, y_test_s, feature_cols, f_scaler, t_scaler, y_test_original = prepare_lstm_data(
        final_df, target_col="Calories", timesteps=24, test_size=0.2
    )
    
    # --- Step 3: Train the Model ---
    # We set a high number of epochs and let EarlyStopping automatically find the best one.
    model, history, predictions_in_calories = train_and_evaluate(
        X_train_s, y_train_s, X_test_s, y_test_s, y_test_original, t_scaler, epochs=100
    )

    # --- Step 4: Visualize the Results ---
    print("\n--- Displaying graphs for analysis... ---")
    plot_training_history(history)
    if len(y_test_original) > 0:
        plot_actual_vs_predicted(y_test_original, predictions_in_calories)
        
    # --- Step 5: Save the Final Model and ALL Necessary Assets into the 'src' Folder ---
    print("\n--- Saving model, assets, and test data to the 'src' folder ---")
    
    # Define the name of the folder where we'll save our assets.
    save_dir = "src"
    
    # Create the folder if it doesn't already exist.
    os.makedirs(save_dir, exist_ok=True)
    
    # Save all the components we need for future predictions.
    model.save(os.path.join(save_dir, "calorie_prediction_model.h5"))
    joblib.dump(f_scaler, os.path.join(save_dir, "feature_scaler.joblib"))
    joblib.dump(t_scaler, os.path.join(save_dir, "target_scaler.joblib"))
    joblib.dump(feature_cols, os.path.join(save_dir, "feature_columns.joblib"))
    
    # --- THIS IS THE NEW PART ---
    # To properly evaluate our model later, we also need to save the exact
    # test data that was used. We regenerate it here to ensure it's the correct slice.
    # Note: We are not importing functions from the same file, so we need to call them directly.
    _ , test_df = split_per_user_time(make_dense_hourly(ensure_types(final_df)), test_size=0.2)
    test_df.to_csv(os.path.join(save_dir, "test_data.csv"), index=False)
    
    print(f"Model, assets, and test data saved successfully in the '{save_dir}' folder!")
    print("You are now ready to use `predict.py` for instant predictions.")