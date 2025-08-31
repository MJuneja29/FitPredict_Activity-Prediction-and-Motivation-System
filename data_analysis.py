import pandas as pd
import matplotlib.pyplot as plt

# for analysis, we are taking 2nd month data into account
# as in 2nd month, extensive daily data is available as compared to 1st month

act_daily = pd.read_csv(r"Data/Fitabase Data 4.12.16-5.12.16/dailyActivity_2_merged.csv")
cal_daily = pd.read_csv(r"Data/Fitabase Data 4.12.16-5.12.16/dailyCalories_2_merged.csv")
int_daily = pd.read_csv(r"Data/Fitabase Data 4.12.16-5.12.16/dailyIntensities_2_merged.csv")
step_daily = pd.read_csv(r"Data/Fitabase Data 4.12.16-5.12.16/dailySteps_2_merged.csv")
sleep_daily = pd.read_csv(r"Data/Fitabase Data 4.12.16-5.12.16/sleepDay_2_merged.csv")
wt_log = pd.read_csv(r"Data/Fitabase Data 4.12.16-5.12.16/weightLogInfo_2_merged.csv")

# Check for unique users
n_users_activity = act_daily['Id'].nunique()
n_users_sleep = sleep_daily['Id'].nunique()
n_users_weight = wt_log['Id'].nunique()
    
print(f"\nNumber of unique users:")
print(f"Activity data: {n_users_activity}")
print(f"Sleep data: {n_users_sleep}")
print(f"Weight data: {n_users_weight}")

print(f"\nDuplicate rows:")  # showing 0, as its already been cleaned while merging both months dataset
print(f"Daily Activity: {act_daily.duplicated().sum()}")
print(f"Sleep Day: {sleep_daily.duplicated().sum()}")
print(f"Weight Log: {wt_log.duplicated().sum()}")

# As dailyActivity and weight log data has been already cleaned.
# we shall continue to clean Sleep Daily data from 2nd month


def clean_transform_data(act_daily, sleep_daily, wt_log):

    print("\n" + "="*30)
    print("TRANSFORMING DATA")
    print("="*30)

    # Clean daily activity data
    act_daily_clean = act_daily.copy()
    act_daily_clean.columns = act_daily_clean.columns.str.lower()
        
    # Convert date and add time features
    act_daily_clean['activitydate'] = pd.to_datetime(act_daily_clean['activitydate'], errors='coerce')
    act_daily_clean['weekday'] = act_daily_clean['activitydate'].dt.day_name()
    act_daily_clean['month'] = act_daily_clean['activitydate'].dt.month_name()


    # Clean sleep data
    sleep_daily_clean = sleep_daily.copy()
    sleep_daily_clean.columns = sleep_daily_clean.columns.str.lower()
    sleep_daily_clean = sleep_daily_clean.drop_duplicates().dropna()
        
    # Convert datetime
    sleep_daily_clean['sleepday'] = pd.to_datetime(sleep_daily_clean['sleepday'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    sleep_daily_clean['date'] = sleep_daily_clean['sleepday'].dt.date
    sleep_daily_clean['weekday'] = sleep_daily_clean['sleepday'].dt.day_name()
        
    # Remove additional duplicates in sleep data (by id and date)
    sleep_daily_clean = sleep_daily_clean.drop_duplicates(subset=['id', 'date'])


    # Clean weight data
    wt_log_clean = wt_log.copy()
    wt_log_clean.columns = wt_log_clean.columns.str.lower()
    wt_log_clean = wt_log_clean.drop_duplicates().dropna()
    wt_log_clean['date'] = pd.to_datetime(wt_log_clean['date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
        
    print(f"\nAfter cleaning process:")
    print(f"Daily Activity: {len(act_daily_clean)} rows")
    print(f"Sleep Day: {len(sleep_daily_clean)} rows")
    print(f"Weight Log: {len(wt_log_clean)} rows")
        
    # Display statistics
    print(f"\nStatistics for Daily Activity:")
    print(act_daily_clean[['totalsteps', 'totaldistance', 'calories', 'sedentaryminutes']].describe())
        
    return act_daily_clean, sleep_daily_clean, wt_log_clean



def act_level_analysis(act_daily_clean):

    print("\n" + "="*30)
    print("ANALYSISING ACTIVITY LEVELS")
    print("="*30)
    
    # Calculate user activity, grouping by id
    user_act_byID = act_daily_clean.groupby('id').agg({
        'totalsteps': 'mean',
        'calories': 'mean',
        'totaldistance': 'mean',
        'sedentaryminutes': 'mean',
        'veryactiveminutes': 'mean',
        'fairlyactiveminutes': 'mean',
        'lightlyactiveminutes': 'mean'
    }).reset_index()

     # Calculate average active minutes combining all active intensities
    user_act_byID['avg_active_minutes'] = (
        user_act_byID['veryactiveminutes'] +
        user_act_byID['fairlyactiveminutes'] +
        user_act_byID['lightlyactiveminutes']
    )
    
    # Define research-based activity classification 
    def classify_activity(steps):
        if steps < 5000:
            return "Sedentary"
        elif steps < 7500:
            return "Low Active"
        elif steps < 10000:
            return "Moderately Active"
        elif steps < 12500:
            return "Active"
        else:
            return "Highly Active"
    
    # Apply activity classification to average total steps
    user_act_byID['activity_level'] = user_act_byID['totalsteps'].apply(classify_activity)

    activity_order = ["Sedentary", "Low Active", "Moderately Active", "Active", "Highly Active"]

    # Convert the 'activity_level' column to a pandas Categorical with specified order
    user_act_byID['activity_level'] = pd.Categorical(
    user_act_byID['activity_level'], 
    categories=activity_order, 
    ordered=True)
        
    # Calculate distribution and percentages of activity levels
    act_distribution = user_act_byID['activity_level'].value_counts().sort_index()
    act_percent = (act_distribution / len(user_act_byID) * 100).round(1)
    
    print("\nActivity Levels:")
    for level in act_distribution.index:
        count = act_distribution[level]
        percentage = act_percent[level]
        print(f"{level}: {count} users ({percentage}%)")
    
    return user_act_byID, act_distribution, act_percent



def usage_pattern_analysis(act_daily_clean, sleep_daily_clean):
    print("\n" + "="*30)
    print("DAILY USAGE STATS & SLEEP PATTERNS ANALYSIS")
    print("="*30)

    weekday_order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

    # -------- Daily Activity -------- #
    daily_stats = (
        act_daily_clean
        .groupby("weekday")
        .agg({
            "totalsteps": "mean",
            "totaldistance": "mean",
            "calories": "mean",
            "veryactiveminutes": "mean",
            "fairlyactiveminutes": "mean",
            "lightlyactiveminutes": "mean"
        })
        .round(2)
    )

    # Total active minutes (sum of activity levels)
    daily_stats["avg_active_minutes"] = (
        daily_stats["veryactiveminutes"] 
        + daily_stats["fairlyactiveminutes"] 
        + daily_stats["lightlyactiveminutes"]
    )

    daily_stats = daily_stats.reindex(weekday_order)

    print("\nAverage Daily Activity by Weekday:")
    print(daily_stats[["totalsteps", "totaldistance", "calories", "avg_active_minutes"]])

    # -------- Sleep Patterns -------- #
    sleep_pattern = (
        sleep_daily_clean
        .groupby("weekday")
        .agg({
            "totalminutesasleep": "mean",
            "totaltimeinbed": "mean"
        })
        .round(2)
    )

    sleep_pattern["sleep_efficiency (%)"] = (
        (sleep_pattern["totalminutesasleep"] / sleep_pattern["totaltimeinbed"]) * 100
    ).round(1)

    sleep_pattern = sleep_pattern.reindex(weekday_order)

    print("\nAverage Sleep Pattern by Weekday:")
    print(sleep_pattern)

    # -------- Return DataFrames for later use -------- #
    return daily_stats, sleep_pattern

