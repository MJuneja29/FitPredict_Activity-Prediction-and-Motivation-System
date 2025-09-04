import pandas as pd
import numpy as np

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
        full_index = pd.date_range(g.index.min(), g.index.max(), freq="H")
        
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