import pandas as pd


# checking consistency


def checking_consistency(first_file, second_file, label):
    print("\n" + "="*50)
    print(f"{label} Comparison")
    print("="*50)

    ids_first = set(first_file['Id'])
    ids_second = set(second_file['Id'])

    all_ids = sorted(ids_first.union(ids_second))

    summary_table = pd.DataFrame({
        'Id': all_ids,
        'First_file': [id_ in ids_first for id_ in all_ids],
        'Second_file': [id_ in ids_second for id_ in all_ids]
    })

    print(summary_table)
    return summary_table


wt_log_first = pd.read_csv(r"Data\\Fitabase Data 3.12.16-4.11.16\\weightLogInfo_merged.csv")
wt_log_second = pd.read_csv(r"Data\\Fitabase Data 4.12.16-5.12.16\\weightLogInfo_2_merged.csv")

wt_log_consistency = checking_consistency(wt_log_first, wt_log_second, "Weight Log Consistency across 2 months")

# Average weight per user
avg_wt_first = (
    wt_log_first.groupby('Id')['WeightKg']
    .mean().reset_index().rename(columns={'WeightKg': 'AvgWeight_First'})
)

avg_wt_second = (
    wt_log_second.groupby('Id')['WeightKg']
    .mean().reset_index().rename(columns={'WeightKg': 'AvgWeight_Second'})
)

# Merge for comparison
wt_comparison = pd.merge(avg_wt_first, avg_wt_second, how='outer', on='Id')

print("===== Average Weight Comparison (First vs Second Period) =====")
print(wt_comparison)


# avg wt is almost similar in both months, so we decided to predict calories.
# going with 2nd month only beacause it has more extensive sleep data, then first month.
# althogh Ids are not consistent in 2nd month also, but its more consistent then both months combined.
# if i had chosen to combine both months, there will be majority of missing values, which may confuse the model.
# so, adding first month data can be in future aspect. 
# now checking consistency within 2nd month

calories_hourly = pd.read_csv(r"Data\\Fitabase Data 4.12.16-5.12.16\\hourlyCalories_2_merged.csv")
intensities_hourly = pd.read_csv(r"Data\\Fitabase Data 4.12.16-5.12.16\\hourlyIntensities_2_merged.csv")
steps_hourly = pd.read_csv(r"Data\\Fitabase Data 4.12.16-5.12.16\\hourlySteps_2_merged.csv")
sleep_minutes = pd.read_csv(r"Data\\Fitabase Data 4.12.16-5.12.16\\minuteSleep_2_merged.csv")
hr_seconds = pd.read_csv(r"Data\\Fitabase Data 4.12.16-5.12.16\\heartrate_seconds_2_merged.csv")


cal_vs_int = checking_consistency(calories_hourly, intensities_hourly, "Calories vs Intensities - Ids")
cal_vs_steps = checking_consistency(calories_hourly, steps_hourly, "Calories vs Steps - Ids")
cal_vs_sleep = checking_consistency(calories_hourly, sleep_minutes, "Calories vs Sleep - Ids")
cal_vs_hr = checking_consistency(calories_hourly, hr_seconds, "Calories vs Heart Rate - Ids")





import pandas as pd

def preprocess_sleep_minute_to_hourly(sleep_minute):
    """
    Convert minute-level sleep data into hourly aggregated features.
    Input: sleep_minute dataframe with columns ['Id', 'date', 'value']
           - 'date' should contain both date & time
           - 'value' is sleep state (1 = asleep, 2 = restless, 3 = awake)
    Output: Hourly aggregated dataframe with normalized proportions
    """

    # Ensure datetime format
    sleep_minute['date'] = pd.to_datetime(sleep_minute['date'])

    # Extract hour from datetime
    sleep_minute['ActivityHour'] = sleep_minute['date'].dt.floor('h')

    # Aggregate hourly: count of each state
    sleep_hourly = sleep_minute.groupby(['Id', 'ActivityHour']).agg(
        total_minutes=('value', 'count'),
        asleep_count=('value', lambda x: (x == 1).sum()),
        restless_count=('value', lambda x: (x == 2).sum()),
        awake_count=('value', lambda x: (x == 3).sum())
    ).reset_index()

    # Convert to proportions
    sleep_hourly['asleep_ratio'] = (sleep_hourly['asleep_count'] / sleep_hourly['total_minutes']).round(2)
    sleep_hourly['restless_ratio'] = (sleep_hourly['restless_count'] / sleep_hourly['total_minutes']).round(2)
    sleep_hourly['awake_ratio'] = (sleep_hourly['awake_count'] / sleep_hourly['total_minutes']).round(2)

    # Sleep quality (example metric = asleep ratio)
    sleep_hourly['sleep_quality'] = sleep_hourly['asleep_ratio']

    # Keep only useful features
    sleep_hourly = sleep_hourly[['Id', 'ActivityHour', 'sleep_quality', 'restless_ratio', 'awake_ratio']]

    return sleep_hourly

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.colheader_justify", "left")
pd.set_option("display.precision", 2)

# Example usage:
sleep_hourly = preprocess_sleep_minute_to_hourly(sleep_minutes)
print(sleep_hourly.head(50))


# # Merge all files
# merged_all = calories.merge(steps, on=['Id','ActivityHour']) \
#                  .merge(intensities, on=['Id','ActivityHour']) \
#                  .merge(sleep_hourly, on=['Id','ActivityHour'], how='left')

# # For users without sleep data → sleep features = 0
# merged = merged_all.fillna(0)



import pandas as pd

def preprocess_heart_rate_with_stress(heartrate_df):
    """
    Process heart rate data:
    - Convert to datetime (single column, not separated)
    - Aggregate to hourly average HR
    - Calculate resting HR (25th percentile per user)
    - Add StressLevel = HR - RestingHR
    """

    # Ensure datetime
    heartrate_df['Time'] = pd.to_datetime(heartrate_df['Time'])

    # Aggregate heart rate hourly
    hr_hourly = (
        heartrate_df.groupby(['Id', pd.Grouper(key='Time', freq='h')])['Value']
        .mean()
        .reset_index()
    )
    hr_hourly.rename(columns={'Value': 'HeartRate', 'Time': 'ActivityHour'}, inplace=True)

    # Compute resting HR (25th percentile) for each user
    resting_hr = (
        hr_hourly.groupby('Id')['HeartRate']
        .quantile(0.25)
        .reset_index()
    )
    resting_hr.columns = ['Id', 'RestingHR']

    # Merge resting HR back to hourly HR
    hr_hourly = pd.merge(hr_hourly, resting_hr, on='Id')

    # Stress Level = HeartRate - RestingHR
    hr_hourly['StressLevel'] = hr_hourly['HeartRate'] - hr_hourly['RestingHR']

    hr_hourly['HeartRate'] = hr_hourly['HeartRate'].round(2)
    hr_hourly['StressLevel'] = hr_hourly['StressLevel'].round(2)

    # Drop RestingHR (if not needed in final output)
    hr_hourly.drop('RestingHR', axis=1, inplace=True)

    return hr_hourly


# Example usage
final_hr = preprocess_heart_rate_with_stress(hr_seconds)
print(final_hr.head(50))
