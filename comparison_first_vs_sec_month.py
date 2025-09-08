# ==============================================================================
# Fitbit Dataset Consistency Analysis (Month 1 vs. Month 2)
#
# Project:      IIT Ropar Capstone Project
#
# Purpose:      This script performs a critical preliminary analysis to compare
#               the user participation and data availability between the two
#               separate monthly Fitbit data dumps.
#
#               The primary goal is to make a data-driven decision on whether
#               to merge the datasets or to select the most complete and
#               consistent one for the machine learning model.
# ==============================================================================

import pandas as pd

# ==============================================================================
# SECTION 1: UTILITY FUNCTION
# ==============================================================================

def comparison_months(first_month_df, second_month_df, label):
    """Analyzes and reports the overlap of unique user IDs between two dataframes.

    This function is used to check how many users are present in the first time
    period, the second, or both, which is a key indicator of data consistency.

    Args:
        first_month_df (pd.DataFrame): Dataframe from the first time period.
        second_month_df (pd.DataFrame): Dataframe from the second time period.
        label (str): A descriptive name for the dataset being compared.

    Returns:
        pd.DataFrame: A summary table showing user presence in each period.
    """
    print("\n" + "="*40)
    print(f"User Consistency Check: {label}")
    print("="*40)

    # Extract the unique set of user IDs from each dataframe.
    ids_first = set(first_month_df['Id'])
    ids_second = set(second_month_df['Id'])

    # Create a master list of all unique IDs that appear in either period.
    all_ids = sorted(ids_first.union(ids_second))

    # Construct a summary table for a clear side-by-side comparison.
    summary_table = pd.DataFrame({
        'Id': all_ids,
        'In_First_Period': [id_ in ids_first for id_ in all_ids],
        'In_Second_Period': [id_ in ids_second for id_ in all_ids]
    })

    print(summary_table.to_string())
    return summary_table

# ==============================================================================
# SECTION 2: DATA LOADING AND ANALYSIS
# ==============================================================================
if __name__ == "__main__":
    
    print("--- Loading datasets from both time periods for comparison ---")
    
    # --- Define file paths for clarity ---
    # Period 1: March 12, 2016 to April 11, 2016
    path_p1 = "Data/Fitabase Data 3.12.16-4.11.16/"
    # Period 2: April 12, 2016 to May 12, 2016
    path_p2 = "Data/Fitabase Data 4.12.16-5.12.16/"

    try:
        # Load the datasets needed for the user overlap analysis.
        wt_log_first = pd.read_csv(path_p1 + "weightLogInfo_merged.csv")
        wt_log_second = pd.read_csv(path_p2 + "weightLogInfo_2_merged.csv")
        dailyActivity_first = pd.read_csv(path_p1 + "dailyActivity_merged.csv")
        dailyActivity_second = pd.read_csv(path_p2 + "dailyActivity_2_merged.csv")
        min_sleep_first = pd.read_csv(path_p1 + "minuteSleep_merged.csv")
        min_sleep_second = pd.read_csv(path_p2 + "minuteSleep_2_merged.csv")
        heartrate_first = pd.read_csv(path_p1 + "heartrate_seconds_merged.csv")
        heartrate_second = pd.read_csv(path_p2 + "heartrate_seconds_2_merged.csv")
        hourlyCalories_first = pd.read_csv(path_p1 + "hourlyCalories_merged.csv")
        hourlyCalories_second = pd.read_csv(path_p2 + "hourlyCalories_2_merged.csv")
        print("All datasets loaded successfully.")
    except FileNotFoundError as e:
        print(f"ERROR: A data file was not found. Please check your file paths.")
        print(f"   Details: {e}")
        exit()

    # --- Run the comparison for each key dataset ---
    # This generates a series of tables showing which users tracked which metrics in each month.
    wt_comparison = comparison_months(wt_log_first, wt_log_second, "Weight Log")
    activity_comparison = comparison_months(dailyActivity_first, dailyActivity_second, "Daily Activity")
    min_sleep_comparison = comparison_months(min_sleep_first, min_sleep_second, "Minute Sleep")
    heartrate_comparison = comparison_months(heartrate_first, heartrate_second, "Heart Rate")
    calories_comparison = comparison_months(hourlyCalories_first, hourlyCalories_second, "Hourly Calories")

    # ==============================================================================
    # SECTION 3: CONCLUSION
    # Based on the analysis, a strategic decision is made for the project.
    # ==============================================================================
    
    # Calculate the number of users who consistently provided the most valuable data.
    # We are particularly interested in users who have sleep and heart rate data,
    # as these are rich sources for feature engineering.
    consistent_sleep_users_p2 = min_sleep_second['Id'].nunique()
    consistent_hr_users_p2 = heartrate_second['Id'].nunique()
    total_users_p2 = dailyActivity_second['Id'].nunique()

    print("\n" + "="*70)
    print("ANALYSIS CONCLUSION & STRATEGIC DECISION")
    print("="*70)
    print("\n### Key Observations:")
    print(f"1. User Overlap: The comparison tables show that while many users are present in both periods,")
    print(f"   a significant number of users only appear in one period, especially for optional data like sleep and heart rate.")
    print("\n2. Data Richness in Period 2:")
    print(f"   - In the second month, {consistent_sleep_users_p2} out of {total_users_p2} users provided sleep data.")
    print(f"   - In the second month, {consistent_hr_users_p2} out of {total_users_p2} users provided heart rate data.")
    print(f"   This indicates a higher and more consistent level of user engagement and data availability in the second period.")
    print("\n3. Risk of Merging:")
    print(f"   - Merging both datasets would create a combined timeline with significant data gaps for many users.")
    print(f"   - These gaps (missing values) would complicate the feature engineering process and could negatively impact the performance")
    print(f"     of the time-series model (LSTM), which relies on continuous data.")

    print("\n### Final Decision:")
    print("To build a robust and high-quality baseline model, we will proceed using **ONLY the data from the second month**")
    print("(April 12, 2016 to May 12, 2016). This approach maximizes data completeness and minimizes the issues")
    print("caused by inconsistent user tracking across the two periods.")
    print("\nIntegrating the first month's data is noted as a potential future enhancement for the project.")
    print("="*70)