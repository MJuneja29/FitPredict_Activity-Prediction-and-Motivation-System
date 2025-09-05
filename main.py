import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def ensure_types(df):
    
    out = df.copy() # Make a copy so we don't change the original data accidentally.
    
    # Convert the 'Id' column to text (string). This is good practice because IDs are labels, not numbers to do math on.
    out["Id"] = out["Id"].astype(str) 
    
    # Convert the 'ActivityHour' column into actual datetime objects. This lets Python understand them as points in time.
    out["ActivityHour"] = pd.to_datetime(out["ActivityHour"], errors="coerce")
    
    # If any 'ActivityHour' values were invalid and couldn't be converted, they become 'NaT' (Not a Time). This line removes those rows.
    # Then, it sorts all the data first by user ID, and then by time. This is CRITICAL for time-series analysis.
    out = out.dropna(subset=["ActivityHour"]).sort_values(["Id", "ActivityHour"])
    
    return out # Return the cleaned and sorted dataframe.



def make_dense_hourly(df):
    
    pieces = [] # Start with an empty list to hold the processed data for each user.
    
    # Loop through the data, looking at one user at a time.
    for uid, g in df.groupby("Id"):
        
        # For the current user's data ('g'), sort it by time and set the 'ActivityHour' as the index.
        g = g.sort_values("ActivityHour").set_index("ActivityHour")
        if g.empty: continue # If a user has no data, skip them.
        
        # Create a complete, unbroken timeline of every single hour from their first recorded hour to their last.
        full_index = pd.date_range(g.index.min(), g.index.max(), freq="h")
        
        # Re-shape the user's data to fit this new, complete timeline.
        # Any missing hours in the original data will now exist as empty rows.
        g_dense = g.reindex(full_index)
        
        g_dense["Id"] = uid # Put the user's ID back in the 'Id' column.
        g_dense.index.name = "ActivityHour" # Name the index column.
        
        # Add this user's completed data to our list.
        pieces.append(g_dense.reset_index())
        
    # Combine the processed data from all users back into one single dataframe.
    dense = pd.concat(pieces, axis=0, ignore_index=True)
    
    # Find all the empty spots (NaN) created by filling in the timeline and replace them with 0.
    # This assumes that a missing hour means zero activity.
    dense = dense.fillna(0)
    
    return dense # Return the dataframe with no time gaps.



def select_feature_columns(df, target_col="Calories"):
    
    # Define columns we definitely DON'T want to use as features.
    exclude = {"Id", "ActivityHour", target_col} 
    
    # Get a list of all columns that contain numbers.
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create the final list of feature columns by taking all number columns and removing the excluded ones.
    feature_cols = [c for c in num_cols if c not in exclude]
    
    return feature_cols # Return the list of feature names.



def split_per_user_time(df, test_size=0.2):
    
    train_parts, test_parts = [], [] # Create empty lists for train and test data pieces.
    
    # Loop through each user's data.
    for uid, g in df.groupby("Id"):
        g = g.sort_values("ActivityHour") # Ensure the user's data is in chronological order.
        n = len(g) # Get the total number of hours recorded for this user.
        if n < 3: continue # If the user has too little data, we skip them.
            
        # Calculate the split point. The first 80% (1 - 0.2) of data will be for training.
        cut = int(max(1, np.floor(n * (1 - test_size))))
        
        # Add the first part of the data to the training list.
        train_parts.append(g.iloc[:cut])
        
        # Add the last part of the data to the testing list.
        test_parts.append(g.iloc[cut:])
        
    # Combine all the training pieces from all users into one big training dataframe.
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=df.columns)
    
    # Combine all the testing pieces into one big testing dataframe.
    test_df  = pd.concat(test_parts, ignore_index=True)  if test_parts else pd.DataFrame(columns=df.columns)
    
    return train_df, test_df # Return the two separate dataframes.



def create_sequences_per_user(df, feature_cols, target_col="Calories", timesteps=24, horizon=1):
    
    X_list, y_list = [], [] # Empty lists to store our input sequences (X) and target values (y).
    
    # Loop through each user's data.
    for uid, g in df.groupby("Id"):
        g = g.sort_values("ActivityHour") # Make sure data is in order.
        
        # Get the feature and target data as NumPy arrays, which is faster.
        feats = g[feature_cols].values
        tgt = g[target_col].values
        
        # Calculate how many sequences we can create for this user.
        max_start = len(g) - (timesteps + horizon) + 1
        if max_start <= 0: continue # If there isn't enough data to make even one sequence, skip.
        
        # The main loop that slides the window across the user's data.
        for i in range(max_start):
            
            # The input sequence (X) is a chunk of 'timesteps' hours (e.g., 24 hours).
            X_list.append(feats[i : i + timesteps])
            
            # The target (y) is the single 'Calories' value we want to predict, which is 'horizon' steps after the sequence ends.
            y_list.append(tgt[i + timesteps + horizon - 1])
            
    if not X_list: # If no sequences were created at all...
        # Return empty arrays with the correct shape.
        return np.empty((0, timesteps, len(feature_cols))), np.empty((0,))
        
    # Convert the lists of sequences into large 3D (for X) and 1D (for y) NumPy arrays.
    # This is the final format the neural network needs.
    return np.stack(X_list), np.array(y_list)






# Section 2: Main Preparation Function

# This function acts like a manager, calling all the utility functions from Section 1 in the correct order to run the entire data preparation pipeline from start to finish.

# prepare_lstm_data(...) function

def prepare_lstm_data(final_hourly,
                      target_col="Calories",
                      timesteps=24,
                      horizon=1,
                      test_size=0.2,
                      scale=True):
    """
    Full pipeline: type-fix -> dense hourly -> per-user split -> sequences -> scaling.
    Returns: X_train, y_train, X_test, y_test, feature_cols, scaler (or None)
    """

    # Step 1: Clean data types and sort.
    df = ensure_types(final_hourly)

    # Step 2: Fill in any missing hours for each user.
    df = make_dense_hourly(df)

    # Step 3: Automatically select the feature columns.
    feature_cols = select_feature_columns(df, target_col=target_col)
    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns found. Check your dataframe.")

    # Step 4: Split data into training and testing sets, carefully respecting time.
    train_df, test_df = split_per_user_time(df, test_size=test_size)

    # Step 5: Convert the flat dataframes into 3D sequences (sliding windows).
    X_train, y_train = create_sequences_per_user(train_df, feature_cols, target_col, timesteps, horizon)
    X_test, y_test   = create_sequences_per_user(test_df,  feature_cols, target_col, timesteps, horizon)

    # --- SCALING FEATURES (X) ---
    feature_scaler = StandardScaler()
    # Fit on train features flattened to 2D and transform
    X_train_scaled = feature_scaler.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
    X_test_scaled = feature_scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)

    # --- SCALING TARGET (y) ---
    # The target scaler expects a 2D array, so we reshape y
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1))

    # Print a summary of the final prepared data.
    print("Data ready for LSTM")
    print(f"Features: {feature_cols}")
    print("X_train:", X_train_scaled.shape, "y_train:", y_train_scaled.shape)
    print("X_test: ", X_test_scaled.shape,  "y_test: ", y_test_scaled.shape)
    
    # Return the scalers along with the data
    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, feature_cols, feature_scaler, target_scaler, y_test

    
    



final_df = pd.read_csv(r"Data\\final_hourly.csv")
X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, feature_cols, feature_scaler, target_scaler, y_test = prepare_lstm_data(
    final_df,
    target_col="Calories",
    timesteps=24,
    horizon=1,
    test_size=0.2,
    scale=True
)

# Show the first 3 timesteps of the first training sequence
print("X_train head (first sequence):")
print(X_train_scaled[0, :3, :])

print("\ny_train head:")
# Show the first 5 target values
print(y_train_scaled[:5])


# Show the first 3 timesteps of the first testing sequence
print("X_test head (first sequence):")
print(X_test_scaled[0, :3, :])

print("\ny_test head:")
# Show the first 5 target values
print(y_test[:5])


# Print the list of feature columns
print("\nFeature columns:")
print(feature_cols)

# Single value for mean of scaled data
print("Mean of scaled data (single value):")
print("X_train_scaled", np.mean(np.mean(X_train_scaled, axis=0)))
print("y_train_scaled", np.mean(np.mean(y_train_scaled, axis=0)))
print("X_test_scaled", np.mean(np.mean(X_test_scaled, axis=0)))
print("y_test_scaled", np.mean(np.mean(y_test_scaled, axis=0)))


# Single value for standard deviation of scaled data
print("\nStandard deviation of scaled data (single value):")
print("X_train_scaled", np.mean(np.std(X_train_scaled, axis=0)))
print("y_train_scaled", np.mean(np.std(y_train_scaled, axis=0)))
print("X_test_scaled", np.mean(np.std(X_test_scaled, axis=0)))
print("y_test_scaled", np.mean(np.std(y_test_scaled, axis=0)))
