import pandas as pd

# ============================================================
#                Fitbit Dataset Comparison
#    First Month (03-12-2016 to 04-11-2016) vs
#    Second Month (04-12-2016 to 05-12-2016)
# ============================================================

# ---------------------------
# Weight Log Comparison
# ---------------------------
wt_log_first = pd.read_csv(r"Data\Fitabase Data 3.12.16-4.11.16\weightLogInfo_merged.csv")
wt_log_second = pd.read_csv(r"Data\Fitabase Data 4.12.16-5.12.16\weightLogInfo_2_merged.csv")

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
# Sleep Log Comparison
# ---------------------------
min_sleep_first = pd.read_csv(r"Data\Fitabase Data 3.12.16-4.11.16\minuteSleep_merged.csv")
min_sleep_second = pd.read_csv(r"Data\Fitabase Data 4.12.16-5.12.16\minuteSleep_2_merged.csv")

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
# Hourly Calories Comparison
# ---------------------------
hourlyCalories_first = pd.read_csv(r"Data\Fitabase Data 3.12.16-4.11.16\hourlyCalories_merged.csv")
hourlyCalories_second = pd.read_csv(r"Data\Fitabase Data 4.12.16-5.12.16\hourlyCalories_2_merged.csv")

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
hourlyIntensities_first = pd.read_csv(r"Data\Fitabase Data 3.12.16-4.11.16\hourlyIntensities_merged.csv")
hourlySteps_first = pd.read_csv(r"Data\Fitabase Data 3.12.16-4.11.16\hourlySteps_merged.csv")

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
hourlyIntensities_second = pd.read_csv(r"Data\Fitabase Data 4.12.16-5.12.16\hourlyIntensities_2_merged.csv")
hourlySteps_second = pd.read_csv(r"Data\Fitabase Data 4.12.16-5.12.16\hourlySteps_2_merged.csv")

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
