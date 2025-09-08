import pandas as pd

# ==============================================================================
# DATA CONSISTENCY CHECK
#
# The first step is to analyze user overlap between different datasets. This helps
# justify which datasets to merge and which time periods to focus on.
# ==============================================================================

# A utility function to compare the user IDs present in two different dataframes.
def checking_consistency(first_file, second_file, label):
    """Analyzes and prints the overlap of user IDs between two dataframes.

    Args:
        first_file (pd.DataFrame): The first dataframe to compare.
        second_file (pd.DataFrame): The second dataframe to compare.
        label (str): A description for the comparison being printed.

    Returns:
        pd.DataFrame: A summary table showing the presence of each user ID in both files.
    """
    print("\n" + "="*50)
    print(f"{label} Comparison")
    print("="*50)

    # Get the unique set of user IDs from each dataframe.
    ids_first = set(first_file['Id'])
    ids_second = set(second_file['Id'])

    # Create a unified list of all user IDs from both files.
    all_ids = sorted(ids_first.union(ids_second))

    # Build a summary table for easy visual comparison.
    summary_table = pd.DataFrame({
        'Id': all_ids,
        'First_file': [id_ in ids_first for id_ in all_ids],
        'Second_file': [id_ in ids_second for id_ in all_ids]
    })

    print(summary_table)
    return summary_table

# --- Initial check across the two monthly data dumps ---
# This analysis helps justify the strategic decision to focus on the second month's data,
# which was found to be more complete, especially for sleep and heart rate.
wt_log_first = pd.read_csv(r"Data\\Fitabase Data 3.12.16-4.11.16\\weightLogInfo_merged.csv")
wt_log_second = pd.read_csv(r"Data\\Fitabase Data 4.12.16-5.12.16\\weightLogInfo_2_merged.csv")
wt_log_consistency = checking_consistency(wt_log_first, wt_log_second, "Weight Log Consistency across 2 months")

# --- Check user consistency within the second month's hourly data ---
# This is crucial for understanding the potential for missing data after merging.
# For instance, not all users who log calories also have heart rate data.
calories_hourly = pd.read_csv(r"Data\\Fitabase Data 4.12.16-5.12.16\\hourlyCalories_2_merged.csv")
intensities_hourly = pd.read_csv(r"Data\\Fitabase Data 4.12.16-5.12.16\\hourlyIntensities_2_merged.csv")
steps_hourly = pd.read_csv(r"Data\\Fitabase Data 4.12.16-5.12.16\\hourlySteps_2_merged.csv")
sleep_minutes = pd.read_csv(r"Data\\Fitabase Data 4.12.16-5.12.16\\minuteSleep_2_merged.csv")
hr_seconds = pd.read_csv(r"Data\\Fitabase Data 4.12.16-5.12.16\\heartrate_seconds_2_merged.csv")

cal_vs_int = checking_consistency(calories_hourly, intensities_hourly, "Calories vs Intensities - Ids")
cal_vs_steps = checking_consistency(calories_hourly, steps_hourly, "Calories vs Steps - Ids")
cal_vs_sleep = checking_consistency(calories_hourly, sleep_minutes, "Calories vs Sleep - Ids")
cal_vs_hr = checking_consistency(calories_hourly, hr_seconds, "Calories vs Heart Rate - Ids")


# ==============================================================================
# FEATURE ENGINEERING
#
# This section contains functions to process raw, granular data (minute-level sleep,
# second-level heart rate) into meaningful hourly features for our model.
# ==============================================================================

def preprocess_sleep_minute_to_hourly(sleep_minute):
    """Converts minute-level sleep data into aggregated hourly sleep features.

    This function calculates the proportion of time spent in each sleep state
    (asleep, restless, awake) within each hour, creating normalized features
    that are suitable for machine learning models.

    Args:
        sleep_minute (pd.DataFrame): The minute-level sleep data with 'Id', 'date', and 'value' columns.

    Returns:
        pd.DataFrame: An hourly dataframe with engineered sleep quality features.
    """
    # Ensure the 'date' column is treated as a proper datetime object.
    sleep_minute['date'] = pd.to_datetime(sleep_minute['date'])

    # Create a consistent 'ActivityHour' column by rounding down the timestamp.
    # This serves as the primary key for merging with other hourly datasets.
    sleep_minute['ActivityHour'] = sleep_minute['date'].dt.floor('h')

    # Group by user and hour, then aggregate to count each sleep state.
    # Sleep state values: 1 = asleep, 2 = restless, 3 = awake.
    sleep_hourly = sleep_minute.groupby(['Id', 'ActivityHour']).agg(
        total_minutes=('value', 'count'),
        asleep_count=('value', lambda x: (x == 1).sum()),
        restless_count=('value', lambda x: (x == 2).sum()),
        awake_count=('value', lambda x: (x == 3).sum())
    ).reset_index()

    # Convert the raw counts into ratios (proportions) for better model performance.
    sleep_hourly['asleep_ratio'] = (sleep_hourly['asleep_count'] / sleep_hourly['total_minutes']).round(2)
    sleep_hourly['restless_ratio'] = (sleep_hourly['restless_count'] / sleep_hourly['total_minutes']).round(2)
    sleep_hourly['awake_ratio'] = (sleep_hourly['awake_count'] / sleep_hourly['total_minutes']).round(2)

    # Define 'sleep_quality' as a primary feature, here represented by the asleep ratio.
    sleep_hourly['sleep_quality'] = sleep_hourly['asleep_ratio']

    # Select and return only the final, engineered features.
    sleep_hourly = sleep_hourly[['Id', 'ActivityHour', 'sleep_quality', 'restless_ratio', 'awake_ratio']]

    return sleep_hourly


def preprocess_heart_rate_with_stress(heartrate_df):
    """Processes second-level heart rate data into hourly features.

    This function calculates the average hourly heart rate and engineers a 'StressLevel'
    feature, defined as the deviation from a user's personalized resting heart rate.

    Args:
        heartrate_df (pd.DataFrame): The second-level heart rate data.

    Returns:
        pd.DataFrame: An hourly dataframe with heart rate and stress features.
    """
    # Ensure the 'Time' column is a proper datetime object.
    heartrate_df['Time'] = pd.to_datetime(heartrate_df['Time'])

    # Aggregate the second-level data to get the mean heart rate for each hour.
    hr_hourly = (
        heartrate_df.groupby(['Id', pd.Grouper(key='Time', freq='h')])['Value']
        .mean()
        .reset_index()
    )
    hr_hourly.rename(columns={'Value': 'HeartRate', 'Time': 'ActivityHour'}, inplace=True)

    # To create a personalized stress metric, we first calculate each user's resting heart rate.
    # The 25th percentile is used as a robust estimate for baseline heart rate,
    # as it's less affected by outlier high values from exercise.
    resting_hr = (
        hr_hourly.groupby('Id')['HeartRate']
        .quantile(0.25)
        .reset_index()
    )
    resting_hr.columns = ['Id', 'RestingHR']

    # Merge the personalized resting heart rate back to the hourly data.
    hr_hourly = pd.merge(hr_hourly, resting_hr, on='Id')

    # The 'StressLevel' feature is the difference between the current hour's HR and the user's baseline.
    hr_hourly['StressLevel'] = hr_hourly['HeartRate'] - hr_hourly['RestingHR']

    # Round the values for cleanliness.
    hr_hourly['HeartRate'] = hr_hourly['HeartRate'].round(2)
    hr_hourly['StressLevel'] = hr_hourly['StressLevel'].round(2)

    # The RestingHR column was a temporary calculation and is no longer needed.
    hr_hourly.drop('RestingHR', axis=1, inplace=True)

    return hr_hourly

# --- Run the feature engineering functions ---
sleep_hourly = preprocess_sleep_minute_to_hourly(sleep_minutes)
hr_hourly = preprocess_heart_rate_with_stress(hr_seconds)


# ==============================================================================
# SECTION 3: DATA CLEANING AND MERGING
# ==============================================================================

def clean_dataframe(df, name="DataFrame"):
    """A general utility function to clean a dataframe by filling nulls and removing duplicates."""
    df_clean = df.copy()

    # For this project, it's a reasonable assumption that missing sensor data
    # (NaN values) for activity implies zero activity, so we fill with 0.
    df_clean = df_clean.fillna(0)

    # Remove any fully duplicate rows.
    df_clean = df_clean.drop_duplicates()

    return df_clean

# --- Clean each of the hourly datasets before merging ---
calories_hourly_clean = clean_dataframe(calories_hourly, "Calories")
intensities_hourly_clean = clean_dataframe(intensities_hourly, "Intensities")
steps_hourly_clean = clean_dataframe(steps_hourly, "Steps")
sleep_hourly_clean = clean_dataframe(sleep_hourly, "Sleep")
hr_hourly_clean = clean_dataframe(hr_hourly, "Heart Rate")


def merge_all_data(calories, intensities, steps, sleep, heartrate, how="outer"):
    """Merges all the cleaned, hourly dataframes into one final dataset.

    An outer merge is used to ensure that all recorded data points are kept,
    even if a user is present in one dataset (e.g., calories) but not another
    (e.g., sleep) for a specific hour.

    Args:
        (All the cleaned hourly dataframes)
        how (str): The type of merge to perform (default is 'outer').

    Returns:
        pd.DataFrame: The final, merged dataframe ready for the ML model.
    """
    # A helper sub-function to ensure data types are consistent before merging.
    def ensure_types(df):
        df = df.copy()
        df["Id"] = df["Id"].astype(str)
        df["ActivityHour"] = pd.to_datetime(df["ActivityHour"])
        return df

    # Apply type consistency to all dataframes.
    calories = ensure_types(calories)
    intensities = ensure_types(intensities)
    steps = ensure_types(steps)
    sleep = ensure_types(sleep)
    heartrate = ensure_types(heartrate)

    # Chain the merges together, starting with the core activity data.
    # The `on` parameter ensures we merge correctly on both user and the specific hour.
    merged_df = calories.copy()
    merged_df = pd.merge(merged_df, intensities, on=["Id", "ActivityHour"], how=how)
    merged_df = pd.merge(merged_df, steps, on=["Id", "ActivityHour"], how=how)
    merged_df = pd.merge(merged_df, sleep, on=["Id", "ActivityHour"], how=how)
    merged_df = pd.merge(merged_df, heartrate, on=["Id", "ActivityHour"], how=how)

    # The outer merge can create NaN values where data is missing (e.g., no sleep data).
    # We fill these with 0 as per our earlier assumption.
    merged_df = merged_df.fillna(0)

    # Sorting by user and then by time is a critical step for preparing data
    # for a time-series model like an LSTM.
    merged_df = merged_df.sort_values(by=["Id", "ActivityHour"]).reset_index(drop=True)

    return merged_df

# --- Run the final merge process ---
final_hourly = merge_all_data(
    calories_hourly_clean,
    intensities_hourly_clean,
    steps_hourly_clean,
    sleep_hourly_clean,
    hr_hourly_clean
)

# --- Save the final, model-ready dataset to a new CSV file ---
# This file will be the input for the next phase: model training.
save_path = r"Data\\final_hourly.csv"
final_hourly.to_csv(save_path, index=False)