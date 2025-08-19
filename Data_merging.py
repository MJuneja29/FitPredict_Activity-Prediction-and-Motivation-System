import pandas as pd

# Load datasets for weight logs, for two time periods
wt_log_first = pd.read_csv(
    r"Data\Fitabase Data 3.12.16-4.11.16\weightLogInfo_merged.csv"
)
wt_log_second = pd.read_csv(
    r"Data\Fitabase Data 4.12.16-5.12.16\weightLogInfo_2_merged.csv"
)

# Display first few rows of each dataset to verify loading
print("First Period Data Sample:")
print(wt_log_first.head(), "\n")

print("Second Period Data Sample:")
print(wt_log_second.head(), "\n")

# Extract unique user IDs from both datasets
ids_first = set(wt_log_first['Id'])
ids_second = set(wt_log_second['Id'])

# Identify common and unique user IDs across the two periods
common_ids = ids_first.intersection(ids_second)
only_in_first = ids_first - ids_second
only_in_second = ids_second - ids_first

# Display summary of user ID comparison
print(f"Number of user IDs in first period: {len(ids_first)}")
print(f"Number of user IDs in second period: {len(ids_second)}")
print(f"Number of common user IDs: {len(common_ids)}")
print(f"User IDs only in first period: {only_in_first}")
print(f"User IDs only in second period: {only_in_second}\n")

# Calculate average weight per user for first period
avg_wt_first = (
    wt_log_first.groupby('Id')['WeightKg']
    .mean()
    .reset_index()
    .rename(columns={'WeightKg': 'avg_wt_first'})
)

# Calculate average weight per user for second period
avg_wt_second = (
    wt_log_second.groupby('Id')['WeightKg']
    .mean()
    .reset_index()
    .rename(columns={'WeightKg': 'avg_wt_second'})
)

# Merge average weights on user ID to compare across periods
result = pd.merge(avg_wt_first, avg_wt_second, how='outer', on='Id')

# Display the resulting comparison table
print("Average Weight Comparison by User (First vs Second Period):")
print(result)
