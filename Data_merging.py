import pandas as pd

# ============================================================
#                Fitbit Dataset Comparison
#    First Month (03-12-2016 to 04-11-2016) vs
#    Second Month (04-12-2016 to 05-12-2016)
# ============================================================

# ---------------------------
# Weight Log Comparison
# ---------------------------
wt_log_first = pd.read_csv(r"Data/Fitabase Data 3.12.16-4.11.16/weightLogInfo_merged.csv")
wt_log_second = pd.read_csv(r"Data/Fitabase Data 4.12.16-5.12.16/weightLogInfo_2_merged.csv")

print("\n===== Weight Log: Data Samples =====")
print("First Period:\n", wt_log_first.head(), "\n")
print("Second Period:\n", wt_log_second.head(), "\n")

# Extract unique user IDs
ids_first = set(wt_log_first['Id'])
ids_second = set(wt_log_second['Id'])

# ID comparison
common_ids = ids_first.intersection(ids_second)
only_in_first = ids_first - ids_second
only_in_second = ids_second - ids_first

print("\n===== Weight Log: User ID Summary =====")
print(f"Users in First Period   : {len(ids_first)}")
print(f"Users in Second Period  : {len(ids_second)}")
print(f"Common Users            : {len(common_ids)}")
print(f"Only in First Period    : {only_in_first}")
print(f"Only in Second Period   : {only_in_second}\n")

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

# ---------------------------
# dailyActivity Comparison
# ---------------------------

dailyActivity_first = pd.read_csv(r"Data/Fitabase Data 3.12.16-4.11.16/dailyActivity_merged.csv")
dailyActivity_second = pd.read_csv(r"Data/Fitabase Data 4.12.16-5.12.16/dailyActivity_2_merged.csv")

ids_activity_first = set(dailyActivity_first['Id'])
ids_activity_second = set(dailyActivity_second['Id'])

all_ids_activity = sorted(ids_activity_first.union(ids_activity_second))

activity_table = pd.DataFrame({
    'Id': all_ids_activity,
    'First_Period': [id_ in ids_activity_first for id_ in all_ids_activity],
    'Second_Period': [id_ in ids_activity_second for id_ in all_ids_activity]
})

print("\n===== dailyActivity Log: User Availability =====")
print(activity_table)



# ---------------------------
# Sleep Log Comparison
# ---------------------------
min_sleep_first = pd.read_csv(r"Data/Fitabase Data 3.12.16-4.11.16/minuteSleep_merged.csv")
min_sleep_second = pd.read_csv(r"Data/Fitabase Data 4.12.16-5.12.16/minuteSleep_2_merged.csv")

ids_sleep_first = set(min_sleep_first['Id'])
ids_sleep_second = set(min_sleep_second['Id'])

all_ids_sleep = sorted(ids_sleep_first.union(ids_sleep_second))

sleep_table = pd.DataFrame({
    'Id': all_ids_sleep,
    'First_Period': [id_ in ids_sleep_first for id_ in all_ids_sleep],
    'Second_Period': [id_ in ids_sleep_second for id_ in all_ids_sleep]
})

print("\n===== Sleep Log: User Availability =====")
print(sleep_table)


# ---------------------------
# Heart Rate Comparison
# ---------------------------
heartrate_first = pd.read_csv(r"Data/Fitabase Data 3.12.16-4.11.16/heartrate_seconds_merged.csv")
heartrate_second = pd.read_csv(r"Data/Fitabase Data 4.12.16-5.12.16/heartrate_seconds_2_merged.csv")

ids_heartrate_first = set(heartrate_first['Id'])
ids_heartrate_second = set(heartrate_second['Id'])

all_ids_heartrate = sorted(ids_heartrate_first.union(ids_heartrate_second))

heartrate_table = pd.DataFrame({
    'Id': all_ids_heartrate,
    'First_Period': [id_ in ids_heartrate_first for id_ in all_ids_heartrate],
    'Second_Period': [id_ in ids_heartrate_second for id_ in all_ids_heartrate]
})

print("\n===== heartrate: User Availability =====")
print(heartrate_table)


# ---------------------------
# Hourly Calories Comparison
# ---------------------------
hourlyCalories_first = pd.read_csv(r"Data/Fitabase Data 3.12.16-4.11.16/hourlyCalories_merged.csv")
hourlyCalories_second = pd.read_csv(r"Data/Fitabase Data 4.12.16-5.12.16/hourlyCalories_2_merged.csv")

ids_cal_first = set(hourlyCalories_first['Id'])
ids_cal_second = set(hourlyCalories_second['Id'])

all_ids_cal = sorted(ids_cal_first.union(ids_cal_second))

calorie_table = pd.DataFrame({
    'Id': all_ids_cal,
    'First_Period': [id_ in ids_cal_first for id_ in all_ids_cal],
    'Second_Period': [id_ in ids_cal_second for id_ in all_ids_cal]
})

print("\n===== Hourly Calories: User Availability =====")
print(calorie_table)


# ---------------------------
# Detailed Comparison: Calories, Intensities, Steps
# ---------------------------

# ---- First Period ----
hourlyIntensities_first = pd.read_csv(r"Data/Fitabase Data 3.12.16-4.11.16/hourlyIntensities_merged.csv")
hourlySteps_first = pd.read_csv(r"Data/Fitabase Data 3.12.16-4.11.16/hourlySteps_merged.csv")

cal_counts_1 = hourlyCalories_first['Id'].value_counts().rename("Calories_Count")
int_counts_1 = hourlyIntensities_first['Id'].value_counts().rename("Intensities_Count")
step_counts_1 = hourlySteps_first['Id'].value_counts().rename("Steps_Count")

comparison_first = pd.concat([cal_counts_1, int_counts_1, step_counts_1], axis=1).fillna(0).astype(int)
comparison_first["All_Counts_Equal"] = (
    (comparison_first["Calories_Count"] == comparison_first["Intensities_Count"]) &
    (comparison_first["Intensities_Count"] == comparison_first["Steps_Count"])
)

print("\n===== Comparison Across Files (First Period) =====")
print(comparison_first.to_string())


# ---- Second Period ----
hourlyIntensities_second = pd.read_csv(r"Data/Fitabase Data 4.12.16-5.12.16/hourlyIntensities_2_merged.csv")
hourlySteps_second = pd.read_csv(r"Data/Fitabase Data 4.12.16-5.12.16/hourlySteps_2_merged.csv")

cal_counts_2 = hourlyCalories_second['Id'].value_counts().rename("Calories_Count_2")
int_counts_2 = hourlyIntensities_second['Id'].value_counts().rename("Intensities_Count_2")
step_counts_2 = hourlySteps_second['Id'].value_counts().rename("Steps_Count_2")

comparison_second = pd.concat([cal_counts_2, int_counts_2, step_counts_2], axis=1).fillna(0).astype(int)
comparison_second["All_Counts_Equal"] = (
    (comparison_second["Calories_Count_2"] == comparison_second["Intensities_Count_2"]) &
    (comparison_second["Intensities_Count_2"] == comparison_second["Steps_Count_2"])
)

print("\n===== Comparison Across Files (Second Period) =====")
print(comparison_second.to_string())


# Merging data

# weightlog

# UNION ALL equivalent → concat
weight_data = pd.concat([wt_log_first, wt_log_second], ignore_index=True)

# Drop irrelevant columns
cols_to_drop = ['Fat', 'IsManualReport', 'LogId', 'WeightPounds']
weight_data = weight_data.drop(columns=cols_to_drop, errors='ignore')

weight_data['WeightKg'] = weight_data['WeightKg'].round(2)
weight_data['BMI'] = weight_data['BMI'].round(2)

# Convert 'Date' to timestamp and keep only date part
weight_data['Date'] = pd.to_datetime(weight_data['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce').dt.date

# Duplicate rows count
wt_duplicate_count = weight_data.duplicated().sum()
print(f"\n--- Duplicate Rows Found: {wt_duplicate_count} ---")

# Remove duplicate rows
weight_data = weight_data.drop_duplicates()

# ---- Data Quality Checks ----
# Column data types
wt_dtypes_info = weight_data.dtypes

# Null value counts
wt_null_counts = weight_data.isnull().sum()

# Save cleaned dataset
wt_output_path = "Data/Merged_data/weight_data.csv"
weight_data.to_csv(wt_output_path, index=False)

# ---- Display Report ----
print("weight_data table created successfully!")
print("Shape of merged dataset:", weight_data.shape)
print("\n--- Column Data Types ---")
print(wt_dtypes_info)

print("\n--- Null Values per Column ---")
print(wt_null_counts)

print(f"\n--- Duplicate Rows after cleaning: {wt_duplicate_count} ---")

print("\nSample rows from cleaned dataset:")
print(weight_data.head())


# dailyActivity

# UNION ALL equivalent → concat
activity_data = pd.concat([dailyActivity_first, dailyActivity_second], ignore_index=True)

# Drop irrelevant columns
cols_to_drop = ['TrackerDistance', 'LoggedActivitiesDistance', 'SedentaryActiveDistance']
activity_data = activity_data.drop(columns=cols_to_drop, errors='ignore')

cols_to_round = [
    'TotalDistance',
    'VeryActiveDistance',
    'ModeratelyActiveDistance',
    'LightActiveDistance',
    'VeryActiveMinutes',
    'FairlyActiveMinutes',
    'LightlyActiveMinutes',
    'SedentaryMinutes',
    'Calories'
]

activity_data[cols_to_round] = activity_data[cols_to_round].round(2)



# Convert 'Date' to timestamp and keep only date part
activity_data['ActivityDate'] = pd.to_datetime(activity_data['ActivityDate'], format='%m/%d/%Y', errors='coerce').dt.date

# Duplicate rows count
activity_duplicate_count = activity_data.duplicated().sum()
print(f"\n--- Duplicate Rows Found: {activity_duplicate_count} ---")

# Remove duplicate rows
activity_data = activity_data.drop_duplicates()

# ---- Data Quality Checks ----
# Column data types
activity_dtypes_info = activity_data.dtypes

# Null value counts
activity_null_counts = activity_data.isnull().sum()

# Save cleaned dataset
activity_output_path = "Data/Merged_data/activity_data.csv"
activity_data.to_csv(activity_output_path, index=False)

# ---- Display Report ----
print("activity_data table created successfully!")
print("Shape of merged dataset:", activity_data.shape)
print("\n--- Column Data Types ---")
print(activity_dtypes_info)
print("\n--- Null Values per Column ---")
print(activity_null_counts)

print("\nSample rows from cleaned dataset:")
print(activity_data.head())




# minuteSleep

# UNION ALL equivalent → concat
minSleep_data = pd.concat([min_sleep_first, min_sleep_second], ignore_index=True)

# Drop irrelevant columns
cols_to_drop = ['logId']
minSleep_data = minSleep_data.drop(columns=cols_to_drop, errors='ignore')

# Convert 'Date' to timestamp and keep date and time part
minSleep_data['date'] = pd.to_datetime(minSleep_data['date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
minSleep_data['Date'] = minSleep_data['date'].dt.date.astype(str)
minSleep_data['Time'] = minSleep_data['date'].dt.time.astype(str)
minSleep_data = minSleep_data.drop(columns=['date'])

# Duplicate rows count
sleep_duplicate_count = minSleep_data.duplicated().sum()
print(f"\n--- Duplicate Rows Found: {sleep_duplicate_count} ---")

# Remove duplicate rows
minSleep_data = minSleep_data.drop_duplicates()

# ---- Data Quality Checks ----
# Column data types
sleep_dtypes_info = minSleep_data.dtypes

# Null value counts
sleep_null_counts = minSleep_data.isnull().sum()

# Save cleaned dataset
sleep_output_path = "Data\Merged_data\minSleep_data.csv"
minSleep_data.to_csv(sleep_output_path, index=False)

# ---- Display Report ----
print("minSleep_data table created successfully!")
print("Shape of merged dataset:", minSleep_data.shape)
print("\n--- Column Data Types ---")
print(sleep_dtypes_info)
print("\n--- Null Values per Column ---")
print(sleep_null_counts)

print("\nSample rows from cleaned dataset:")
print(minSleep_data.head())




# hourlyCalories

# UNION ALL equivalent → concat
hourlyCalories_data = pd.concat([hourlyCalories_first, hourlyCalories_second], ignore_index=True)

# Convert 'Date' to timestamp and keep date and time part
hourlyCalories_data['ActivityHour'] = pd.to_datetime(hourlyCalories_data['ActivityHour'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
hourlyCalories_data['Date'] = hourlyCalories_data['ActivityHour'].dt.date.astype(str)        # YYYY-MM-DD as string
hourlyCalories_data['Time'] = hourlyCalories_data['ActivityHour'].dt.time.astype(str)
hourlyCalories_data = hourlyCalories_data.drop(columns=['ActivityHour'])

# Duplicate rows count
hourlyCalories_duplicate_count = hourlyCalories_data.duplicated().sum()
print(f"\n--- Duplicate Rows Found: {hourlyCalories_duplicate_count} ---")

# Remove duplicate rows
hourlyCalories_data = hourlyCalories_data.drop_duplicates()

# ---- Data Quality Checks ----
# Column data types
hourlyCalories_dtypes_info = hourlyCalories_data.dtypes

# Null value counts
hourlyCalories_null_counts = hourlyCalories_data.isnull().sum()

# Save cleaned dataset
hourlyCalories_output_path = "Data/Merged_data/hourlyCalories_data.csv"
hourlyCalories_data.to_csv(hourlyCalories_output_path, index=False)

# ---- Display Report ----
print("hourlyCalories_data table created successfully!")
print("Shape of merged dataset:", hourlyCalories_data.shape)
print("\n--- Column Data Types ---")
print(hourlyCalories_dtypes_info)
print("\n--- Null Values per Column ---")
print(hourlyCalories_null_counts)

print("\nSample rows from cleaned dataset:")
print(hourlyCalories_data.head())



# hourlyIntensities

# UNION ALL equivalent → concat
hourlyIntensities_data = pd.concat([hourlyIntensities_first, hourlyIntensities_second], ignore_index=True)

hourlyIntensities_data['AverageIntensity'] = hourlyIntensities_data['AverageIntensity'].round(3)

# Convert 'Date' to timestamp and keep date and time part
hourlyIntensities_data['ActivityHour'] = pd.to_datetime(hourlyIntensities_data['ActivityHour'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
hourlyIntensities_data['Date'] = hourlyIntensities_data['ActivityHour'].dt.date.astype(str)        # YYYY-MM-DD as string
hourlyIntensities_data['Time'] = hourlyIntensities_data['ActivityHour'].dt.time.astype(str)
hourlyIntensities_data = hourlyIntensities_data.drop(columns=['ActivityHour'])

# Duplicate rows count
hourlyIntensities_duplicate_count = hourlyIntensities_data.duplicated().sum()
print(f"\n--- Duplicate Rows Found: {hourlyIntensities_duplicate_count} ---")

# Remove duplicate rows
hourlyIntensities_data = hourlyIntensities_data.drop_duplicates()

# ---- Data Quality Checks ----
# Column data types
hourlyIntensities_dtypes_info = hourlyIntensities_data.dtypes

# Null value counts
hourlyIntensities_null_counts = hourlyIntensities_data.isnull().sum()

# Save cleaned dataset
hourlyIntensities_output_path = "Data/Merged_data/hourlyIntensities_data.csv"
hourlyIntensities_data.to_csv(hourlyIntensities_output_path, index=False)

# ---- Display Report ----
print("hourlyIntensities_data table created successfully!")
print("Shape of merged dataset:", hourlyIntensities_data.shape)
print("\n--- Column Data Types ---")
print(hourlyIntensities_dtypes_info)
print("\n--- Null Values per Column ---")
print(hourlyIntensities_null_counts)

print("\nSample rows from cleaned dataset:")
print(hourlyIntensities_data.head())





# hourlySteps

# UNION ALL equivalent → concat
hourlySteps_data = pd.concat([hourlySteps_first, hourlySteps_second], ignore_index=True)

# Convert 'Date' to timestamp and keep date and time part
hourlySteps_data['ActivityHour'] = pd.to_datetime(hourlySteps_data['ActivityHour'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
hourlySteps_data['Date'] = hourlySteps_data['ActivityHour'].dt.date.astype(str)        # YYYY-MM-DD as string
hourlySteps_data['Time'] = hourlySteps_data['ActivityHour'].dt.time.astype(str)
hourlySteps_data = hourlySteps_data.drop(columns=['ActivityHour'])

# Duplicate rows count
hourlySteps_duplicate_count = hourlySteps_data.duplicated().sum()
print(f"\n--- Duplicate Rows Found: {hourlySteps_duplicate_count} ---")

# Remove duplicate rows
hourlySteps_data = hourlySteps_data.drop_duplicates()

# ---- Data Quality Checks ----
# Column data types
hourlySteps_dtypes_info = hourlySteps_data.dtypes

# Null value counts
hourlySteps_null_counts = hourlySteps_data.isnull().sum()

# Save cleaned dataset
hourlySteps_output_path = "Data/Merged_data/hourlySteps_data.csv"
hourlySteps_data.to_csv(hourlySteps_output_path, index=False)

# ---- Display Report ----
print("hourlySteps_data table created successfully!")
print("Shape of merged dataset:", hourlySteps_data.shape)
print("\n--- Column Data Types ---")
print(hourlySteps_dtypes_info)
print("\n--- Null Values per Column ---")
print(hourlySteps_null_counts)

print("\nSample rows from cleaned dataset:")
print(hourlySteps_data.head())


