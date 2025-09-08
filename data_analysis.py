# ==============================================================================
# Fitbit User Activity - Exploratory Data Analysis (EDA)
#
# Project:      IIT Ropar Capstone Project
#
# Purpose:      This script performs a comprehensive exploratory data analysis on
#               Fitbit user data. It follows a structured process:
#               1. Load and inspect the datasets.
#               2. Clean and transform the data for analysis.
#               3. Analyze user activity levels and daily patterns.
#               4. Generate a dashboard of visualizations.
#               5. Derive and print key behavioral insights.
# ==============================================================================

# --- Step 1: Import all the necessary libraries ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Step 2: Load the Datasets ---
# For this analysis, we are focusing on the second month of data, as it is more complete.
print("--- Loading Datasets ---")
try:
    act_daily = pd.read_csv(r"Data/Fitabase Data 4.12.16-5.12.16/dailyActivity_2_merged.csv")
    sleep_daily = pd.read_csv(r"Data/Fitabase Data 4.12.16-5.12.16/sleepDay_2_merged.csv")
    wt_log = pd.read_csv(r"Data/Fitabase Data 4.12.16-5.12.16/weightLogInfo_2_merged.csv")
    print("Datasets loaded successfully.")
except FileNotFoundError:
    print("ERROR: Data files not found. Please ensure the data path is correct.")
    exit()

# --- Step 3: Initial Data Inspection ---
# It's good practice to start by understanding the scope of the data.
print("\n--- Initial Data Inspection ---")
n_users_activity = act_daily['Id'].nunique()
n_users_sleep = sleep_daily['Id'].nunique()
n_users_weight = wt_log['Id'].nunique()

print(f"Number of unique users:")
print(f"  - Activity data: {n_users_activity}")
print(f"  - Sleep data:    {n_users_sleep} (Note: Fewer users track sleep)")
print(f"  - Weight data:   {n_users_weight} (Note: Even fewer track weight)")

# Verify that the initial cleaning (merging months) has removed duplicate rows.
print(f"\nDuplicate rows check:")
print(f"  - Daily Activity: {act_daily.duplicated().sum()}")
print(f"  - Sleep Day:      {sleep_daily.duplicated().sum()}")
print(f"  - Weight Log:     {wt_log.duplicated().sum()}")


# ==============================================================================
# SECTION 1: DATA CLEANING AND TRANSFORMATION
# ==============================================================================

def clean_transform_data(act_daily, sleep_daily, wt_log):
    """Cleans, formats, and transforms the raw dataframes for analysis.

    This function performs several key operations:
    - Standardizes column names to lowercase.
    - Converts date columns to the proper datetime format.
    - Extracts useful time features like weekday and month.
    - Removes duplicate entries and null values.

    Args:
        act_daily (pd.DataFrame): The raw daily activity dataframe.
        sleep_daily (pd.DataFrame): The raw daily sleep dataframe.
        wt_log (pd.DataFrame): The raw weight log dataframe.

    Returns:
        tuple: A tuple containing the three cleaned dataframes:
               (act_daily_clean, sleep_daily_clean, wt_log_clean)
    """
    print("\n" + "="*30)
    print("STEP 1: CLEANING AND TRANSFORMING DATA")
    print("="*30)

    # --- Clean daily activity data ---
    act_daily_clean = act_daily.copy()
    act_daily_clean.columns = act_daily_clean.columns.str.lower()
    act_daily_clean['activitydate'] = pd.to_datetime(act_daily_clean['activitydate'], errors='coerce')
    act_daily_clean['weekday'] = act_daily_clean['activitydate'].dt.day_name()
    act_daily_clean['month'] = act_daily_clean['activitydate'].dt.month_name()

    # --- Clean sleep data ---
    sleep_daily_clean = sleep_daily.copy()
    sleep_daily_clean.columns = sleep_daily_clean.columns.str.lower()
    sleep_daily_clean = sleep_daily_clean.drop_duplicates().dropna()
    sleep_daily_clean['sleepday'] = pd.to_datetime(sleep_daily_clean['sleepday'], errors='coerce')
    # Create a simple 'date' column for easier merging with other datasets.
    sleep_daily_clean['date'] = sleep_daily_clean['sleepday'].dt.date
    sleep_daily_clean['weekday'] = sleep_daily_clean['sleepday'].dt.day_name()
    # Some users have multiple sleep entries for one night. We keep only the first one.
    sleep_daily_clean = sleep_daily_clean.drop_duplicates(subset=['id', 'date'])

    # --- Clean weight data ---
    wt_log_clean = wt_log.copy()
    wt_log_clean.columns = wt_log_clean.columns.str.lower()
    wt_log_clean = wt_log_clean.drop_duplicates().dropna()
    wt_log_clean['date'] = pd.to_datetime(wt_log_clean['date'], errors='coerce')

    print("Data cleaning complete.")
    return act_daily_clean, sleep_daily_clean, wt_log_clean


# ==============================================================================
# SECTION 2: USER SEGMENTATION
# Analyze the overall behavior of users to segment them into activity levels.
# ==============================================================================

def analyze_activity_levels(act_daily_clean):
    """Analyzes user activity to segment them into different activity levels.

    This function calculates each user's average daily metrics and then classifies
    them into categories (e.g., 'Sedentary', 'Active') based on their average steps.

    Args:
        act_daily_clean (pd.DataFrame): The cleaned daily activity dataframe.

    Returns:
        tuple: A tuple containing:
               - user_act_byID (pd.DataFrame): A dataframe with one row per user,
                 summarizing their average activity and assigned level.
               - act_distribution (pd.Series): A count of users in each level.
    """
    print("\n" + "="*30)
    print("STEP 2: ANALYZING USER ACTIVITY LEVELS")
    print("="*30)
    
    # First, we calculate the average for each key metric for every user.
    user_act_byID = act_daily_clean.groupby('id').agg({
        'totalsteps': 'mean',
        'calories': 'mean',
        'totaldistance': 'mean',
        'sedentaryminutes': 'mean',
        'veryactiveminutes': 'mean',
        'fairlyactiveminutes': 'mean',
        'lightlyactiveminutes': 'mean'
    }).reset_index()

    # --- Classify users into activity levels based on average steps ---
    # These thresholds are based on common research standards for step counts.
    def classify_activity(steps):
        if steps < 5000: return "Sedentary"
        elif steps < 7500: return "Low Active"
        elif steps < 10000: return "Moderately Active"
        elif steps < 12500: return "Active"
        else: return "Highly Active"
    
    user_act_byID['activity_level'] = user_act_byID['totalsteps'].apply(classify_activity)

    # --- Prepare data for visualization ---
    # We set a specific order for the levels to make our charts logical.
    activity_order = ["Sedentary", "Low Active", "Moderately Active", "Active", "Highly Active"]
    user_act_byID['activity_level'] = pd.Categorical(
        user_act_byID['activity_level'], categories=activity_order, ordered=True
    )
        
    act_distribution = user_act_byID['activity_level'].value_counts().sort_index()
    
    print("User segmentation complete.")
    return user_act_byID, act_distribution


# ==============================================================================
# SECTION 3: PATTERN ANALYSIS
# Look for trends in activity and sleep based on the day of the week.
# ==============================================================================

def analyze_usage_patterns(act_daily_clean, sleep_daily_clean):
    """Analyzes average activity and sleep patterns across days of the week.

    Args:
        act_daily_clean (pd.DataFrame): Cleaned daily activity data.
        sleep_daily_clean (pd.DataFrame): Cleaned daily sleep data.

    Returns:
        tuple: A tuple containing two dataframes for visualization:
               (daily_stats, sleep_pattern)
    """
    print("\n" + "="*30)
    print("STEP 3: ANALYZING WEEKDAY PATTERNS")
    print("="*30)

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # --- Analyze Daily Activity by Weekday ---
    daily_stats = act_daily_clean.groupby("weekday").agg(
        totalsteps=("totalsteps", "mean"),
        calories=("calories", "mean")
    ).reindex(weekday_order)

    # --- Analyze Sleep Patterns by Weekday ---
    sleep_pattern = sleep_daily_clean.groupby("weekday").agg(
        totalminutesasleep=("totalminutesasleep", "mean"),
        totaltimeinbed=("totaltimeinbed", "mean")
    ).reindex(weekday_order)
    sleep_pattern["sleep_efficiency (%)"] = (
        (sleep_pattern["totalminutesasleep"] / sleep_pattern["totaltimeinbed"]) * 100
    ).round(1)

    print("Weekday pattern analysis complete.")
    return daily_stats, sleep_pattern

# ==============================================================================
# SECTION 4: VISUALIZATION
# Create a dashboard of plots to visually represent the findings.
# ==============================================================================

def create_visualizations(act_daily_clean, sleep_daily_clean, user_act_byID, daily_stats):
    """Generates and displays a dashboard of key visualizations.
    
    Args:
        (All the necessary dataframes from previous steps)
    """
    print("\n" + "="*30)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("="*30)

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # Set a consistent and professional style for all plots.
    sns.set_style("whitegrid")
    
    # --- Plot 1: Correlation Heatmap ---
    # This helps us see which activity metrics are strongly related.
    plt.figure(figsize=(10, 8))
    corr_cols = ['totalsteps', 'totaldistance', 'calories', 'veryactiveminutes', 
                 'fairlyactiveminutes', 'lightlyactiveminutes', 'sedentaryminutes']
    sns.heatmap(act_daily_clean[corr_cols].corr(), annot=True, cmap='viridis', fmt=".2f")
    plt.title('Correlation Matrix of Activity Variables', fontsize=16, fontweight='bold')
    plt.show()

    # --- Plot 2: User Activity Level Distribution ---
    # This shows what percentage of users fall into each activity category.
    plt.figure(figsize=(10, 8))
    activity_distribution = user_act_byID['activity_level'].value_counts().sort_index()
    colors = sns.color_palette('pastel')[0:len(activity_distribution)]
    plt.pie(activity_distribution, labels=activity_distribution.index, autopct='%.1f%%', colors=colors, startangle=90)
    plt.title('User Activity Level Distribution', fontsize=16, fontweight='bold')
    plt.ylabel('') # Hides the 'activity_level' label on the side
    plt.show()

    # --- Plot 3: Average Steps vs. Calories Burned ---
    # This visualizes the strong positive relationship between walking and burning calories.
    plt.figure(figsize=(12, 7))
    sns.regplot(x='totalsteps', y='calories', data=act_daily_clean,
                scatter_kws={'alpha':0.4, 'color':'darkgreen'}, line_kws={'color':'red', 'linestyle':'--'})
    plt.title('Steps vs. Calories Burned', fontsize=16, fontweight='bold')
    plt.xlabel('Total Daily Steps')
    plt.ylabel('Total Daily Calories Burned')
    plt.show()

    # --- Plot 4: Average Daily Steps by Day of the Week ---
    plt.figure(figsize=(12, 7))
    sns.barplot(x=daily_stats.index, y=daily_stats['totalsteps'], palette="Blues_d", order=weekday_order)
    plt.title('Average Daily Steps by Day of the Week', fontsize=16, fontweight='bold')
    plt.xlabel('Day of the Week')
    plt.ylabel('Average Steps')
    plt.xticks(rotation=45)
    plt.show()
    
    print("Visualization generation complete.")

# ==============================================================================
# SECTION 5: INSIGHTS CALCULATION
# Distill the analysis into key, human-readable takeaways.
# ==============================================================================

def calculate_key_insights(act_daily_clean, sleep_daily_clean, user_act_byID):
    """Calculates and prints the main summary insights from the analysis.
    
    Returns:
        dict: A dictionary containing the key findings.
    """
    print("\n" + "="*30)
    print("STEP 5: DERIVING KEY INSIGHTS")
    print("="*30)

    # --- Insight 1: What does a typical user's day look like? ---
    mean_steps = act_daily_clean['totalsteps'].mean()
    mean_sleep_hrs = sleep_daily_clean['totalminutesasleep'].mean() / 60
    mean_sedentary_hrs = act_daily_clean['sedentaryminutes'].mean() / 60

    # --- Insight 2: Are users more active or sedentary? ---
    # We consider "Moderately Active" and above as the target group for active lifestyles.
    active_users_mask = user_act_byID['activity_level'].isin(["Moderately Active", "Active", "Highly Active"])
    active_ratio = (active_users_mask.sum() / len(user_act_byID)) * 100

    # --- Insight 3: What are the most and least active days? ---
    day_with_max_steps = act_daily_clean.groupby('weekday')['totalsteps'].mean().idxmax()
    day_with_min_steps = act_daily_clean.groupby('weekday')['totalsteps'].mean().idxmin()
    
    # --- Insight 4: How strong is the link between steps and calories? ---
    step_calorie_corr = act_daily_clean[['totalsteps', 'calories']].corr().iloc[0, 1]

    # --- Print a clean summary of the findings ---
    print("\n### Key Behavioral Insights ###")
    print(f"- The average user takes approximately {mean_steps:,.0f} steps per day, which falls into the 'Low Active' to 'Moderately Active' range.")
    print(f"- On average, users sleep for {mean_sleep_hrs:.1f} hours per night and are sedentary for {mean_sedentary_hrs:.1f} hours per day.")
    print(f"- User Segmentation: About {active_ratio:.1f}% of users can be classified as having an active lifestyle.")
    print(f"- Weekday Patterns: The most active day of the week is typically {day_with_max_steps}, while the least active is {day_with_min_steps}.")
    print(f"- Strong Correlation: There is a very strong positive correlation ({step_calorie_corr:.2f}) between the number of steps taken and calories burned.")
    
    return { 'mean_steps': mean_steps, 'active_ratio': active_ratio }


# ==============================================================================
# MAIN SCRIPT EXECUTION
# This block runs all the steps in sequence when the script is executed.
# ==============================================================================
if __name__ == "__main__":
    
    # Run the entire pipeline from start to finish
    act_daily_clean, sleep_daily_clean, wt_log_clean = clean_transform_data(act_daily, sleep_daily, wt_log)
    user_act_byID, act_distribution = analyze_activity_levels(act_daily_clean)
    daily_stats, sleep_pattern = analyze_usage_patterns(act_daily_clean, sleep_daily_clean)
    create_visualizations(act_daily_clean, sleep_daily_clean, user_act_byID, daily_stats)
    key_findings = calculate_key_insights(act_daily_clean, sleep_daily_clean, user_act_byID)
    
    print("\n\nExploratory Data Analysis complete.")