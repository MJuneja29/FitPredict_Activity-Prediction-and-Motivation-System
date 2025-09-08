import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
    sleep_daily_clean['sleepday'] = pd.to_datetime(sleep_daily_clean['sleepday'], errors='coerce')
    sleep_daily_clean['date'] = sleep_daily_clean['sleepday'].dt.date
    sleep_daily_clean['weekday'] = sleep_daily_clean['sleepday'].dt.day_name()
        
    # Remove additional duplicates in sleep data (by id and date)
    sleep_daily_clean = sleep_daily_clean.drop_duplicates(subset=['id', 'date'])


    # Clean weight data
    wt_log_clean = wt_log.copy()
    wt_log_clean.columns = wt_log_clean.columns.str.lower()
    wt_log_clean = wt_log_clean.drop_duplicates().dropna()
    wt_log_clean['date'] = pd.to_datetime(wt_log_clean['date'], errors='coerce')
        
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


def create_visualizations(act_daily_clean, sleep_daily_clean, act_distribution, 
                         daily_stats, sleep_pattern, user_act_byID):
    
    print("\n" + "="*30)
    print("GENERATING VISUALIZATIONS")
    print("="*30)
    
    # Set consistent plotting style
    plt.rcParams['figure.figsize'] = (15, 10)
    sns.set_style("whitegrid")

    # --- HEATMAP (Separate Figure) ---
    plt.figure(figsize=(10, 8))
    corr_cols = ['totalsteps', 'totaldistance', 'calories', 
                 'veryactiveminutes', 'fairlyactiveminutes', 
                 'lightlyactiveminutes', 'sedentaryminutes']
    corr_matrix = act_daily_clean[corr_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, cbar_kws={"shrink": .9}, fmt=".2f")
    plt.title('Activity Variables Correlation', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show() # Display the heatmap
    
    # --- MAIN DASHBOARD ---
    # Create a dashboard canvas with 3x3 plots
    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    
    # ---------------------------
    # 1. Activity Level Distribution (Pie Chart)
    # ---------------------------
    ax1 = axes[0, 0]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
    act_distribution.plot(kind='pie', autopct='%1.1f%%', colors=colors, ax=ax1, pctdistance=0.85, labeldistance=1.1)
    ax1.set_title('User Activity Level Distribution', fontsize=12, fontweight='bold')
    ax1.set_ylabel('')
    
    # ---------------------------
    # 2. Average Steps by Weekday (Bar Chart with vertical labels)
    # ---------------------------
    ax2 = axes[0, 1]
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_usage_ordered = daily_stats.reindex(weekday_order)
    daily_usage_ordered['totalsteps'].plot(kind='bar', color='steelblue', ax=ax2)
    ax2.set_title('Average Steps by Day', fontsize=12, fontweight='bold')
    ax2.set_xlabel('') 
    ax2.set_ylabel('Total Steps')
    ax2.tick_params(axis='x', rotation=90) # Rotate x-axis labels vertically
    ax2.set_xticklabels(weekday_order) # Set the weekday labels
    
    # ---------------------------
    # 3. Steps vs Calories (Scatter + Trendline)
    # ---------------------------
    ax3 = axes[0, 2]
    ax3.scatter(act_daily_clean['totalsteps'], act_daily_clean['calories'], 
                alpha=0.6, color='darkgreen')
    
    # Fit linear regression line
    z = np.polyfit(act_daily_clean['totalsteps'], act_daily_clean['calories'], 1)
    p = np.poly1d(z)
    ax3.plot(act_daily_clean['totalsteps'], p(act_daily_clean['totalsteps']), 
             "r--", alpha=0.8, label="Trendline")
    
    ax3.set_title('Steps vs Calories Burned', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Total Steps')
    ax3.set_ylabel('Calories Burned')
    
    # Correlation text box
    correlation = np.corrcoef(act_daily_clean['totalsteps'], act_daily_clean['calories'])[0,1]
    ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax3.transAxes, 
             bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    
    # ---------------------------
    # 4. Sleep Efficiency by Weekday (Bar Chart with vertical labels)
    # ---------------------------
    ax4 = axes[1, 0]
    sleep_pattern_ordered = sleep_pattern.reindex(weekday_order)
    sleep_pattern_ordered['sleep_efficiency (%)'].plot(kind='bar', color='purple', ax=ax4)
    ax4.set_title('Sleep Efficiency by Day', fontsize=12, fontweight='bold')
    ax4.set_xlabel('')
    ax4.set_ylabel('Sleep Efficiency (%)')
    ax4.tick_params(axis='x', rotation=90) # Rotate x-axis labels vertically
    ax4.set_xticklabels(weekday_order) # Set the weekday labels
    
    # ---------------------------
    # 5. Steps vs Sedentary Minutes
    # ---------------------------
    ax5 = axes[1, 1]
    ax5.scatter(act_daily_clean['totalsteps'], act_daily_clean['sedentaryminutes'], 
                alpha=0.6, color='darkred')
    
    # Fit regression line
    z = np.polyfit(act_daily_clean['totalsteps'], act_daily_clean['sedentaryminutes'], 1)
    p = np.poly1d(z)
    ax5.plot(act_daily_clean['totalsteps'], p(act_daily_clean['totalsteps']), 
             "b--", alpha=0.8, label="Trendline")
    
    ax5.set_title('Steps vs Sedentary Time', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Total Steps')
    ax5.set_ylabel('Sedentary Minutes')
    
    # ---------------------------
    # 6. Average Activity Minutes Distribution (Bar Chart with vertical labels)
    # ---------------------------
    ax6 = axes[1, 2]
    activity_minutes = act_daily_clean[['veryactiveminutes', 'fairlyactiveminutes', 
                                           'lightlyactiveminutes', 'sedentaryminutes']].mean()
    activity_minutes.plot(kind='bar', 
                          color=['red', 'orange', 'yellow', 'gray'], ax=ax6, edgecolor="black")
    ax6.set_title('Average Daily Activity Distribution', fontsize=12, fontweight='bold')
    ax6.set_xlabel('') 
    ax6.set_ylabel('Average Minutes')
    activity_labels = [label.replace('minutes', '').strip().capitalize() for label in activity_minutes.index.tolist()]
    ax6.tick_params(axis='x', rotation=90) # Rotate x-axis labels vertically
    ax6.set_xticklabels(activity_labels) # Set cleaned activity labels
    
    # ---------------------------
    # 7. Steps vs Sleep Duration (Merged Data)
    # ---------------------------
    ax7 = axes[2, 0]
    
    # Convert activity date to match sleep date
    act_daily_clean['date'] = act_daily_clean['activitydate'].dt.date
    activity_sleep = pd.merge(act_daily_clean, sleep_daily_clean, 
                             on=['id', 'date'], how='inner')
    
    if not activity_sleep.empty:
        ax7.scatter(activity_sleep['totalsteps'], activity_sleep['totalminutesasleep'], 
                    alpha=0.6, color='indigo')
        
        # Fit regression
        z = np.polyfit(activity_sleep['totalsteps'], activity_sleep['totalminutesasleep'], 1)
        p = np.poly1d(z)
        ax7.plot(activity_sleep['totalsteps'], p(activity_sleep['totalsteps']), 
                 "orange", alpha=0.8, label="Trendline")
        
        ax7.set_title('Steps vs Sleep Duration', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Total Steps')
        ax7.set_ylabel('Minutes Asleep')
    else:
        ax7.text(0.5, 0.5, 'No merged data for Steps vs Sleep Duration', 
                 horizontalalignment='center', verticalalignment='center', 
                 transform=ax7.transAxes, fontsize=10, color='gray')
        ax7.set_title('Steps vs Sleep Duration (No Data)', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Total Steps')
        ax7.set_ylabel('Minutes Asleep')
    
    # ---------------------------
    # 8. Steps Distribution by Activity Level (Boxplot with vertical labels)
    # ---------------------------
    ax8 = axes[2, 1]
    user_act_byID.boxplot(column='totalsteps', by='activity_level', ax=ax8, grid=False)
    ax8.set_title('Steps Distribution by Activity Level', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Activity Level')
    ax8.set_ylabel('Total Steps')
    ax8.tick_params(axis='x', rotation=90) # Rotate x-axis labels vertically
    fig.suptitle("") # Suppress the automatic suptitle from boxplot 'by' argument
    
    # ---------------------------
    # 9. Placeholder (or you can add another plot if needed)
    # This slot was previously for the heatmap, which is now separate.
    # We can leave it empty or fill it with another relevant plot.
    # For now, let's turn it off.
    fig.delaxes(axes[2, 2])
    
    # ---------------------------
    # Final layout adjustments for the dashboard
    # ---------------------------
    fig.suptitle("Comprehensive User Activity & Sleep Dashboard", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust rect to make space for suptitle
    fig.subplots_adjust(hspace=0.6, wspace=0.4)
    plt.show() # Display the dashboard

    return correlation

def calculate_key_insights(act_daily_clean, sleep_daily_clean, user_act_byID):
    
    print("\n" + "="*30)
    print("USER BEHAVIOR INSIGHTS")
    print("="*30)

    # -----------------------------------------------------
    # 1. Basic lifestyle averages
    # -----------------------------------------------------
    mean_steps = act_daily_clean['totalsteps'].mean()                      # avg daily steps
    mean_sleep_hrs = sleep_daily_clean['totalminutesasleep'].mean() / 60   # avg sleep (hours)
    mean_sedentary_hrs = act_daily_clean['sedentaryminutes'].mean() / 60   # avg sedentary (hours)

    # -----------------------------------------------------
    # 2. Active vs inactive user distribution
    # -----------------------------------------------------
    movers = user_act_byID[user_act_byID['activity_level'].isin(["Moderately Active", "Active", "Highly Active"])]
    movers_ratio = (len(movers) / len(user_act_byID)) * 100
    inactive_ratio = 100 - movers_ratio

    # -----------------------------------------------------
    # 3. Participation in activity vs sleep tracking
    # -----------------------------------------------------
    tracked_act_users = act_daily_clean['id'].nunique()
    tracked_sleep_users = sleep_daily_clean['id'].nunique()

    # -----------------------------------------------------
    # 4. Print readable insights
    # -----------------------------------------------------
    print("\nGENERAL INSIGHTS")
    print(f"- Typical daily steps: {mean_steps:,.0f}")
    print(f"- Typical sleep duration: {mean_sleep_hrs:.1f} hrs")
    print(f"- Typical sedentary time: {mean_sedentary_hrs:.1f} hrs")

    print("\nACTIVITY LEVELS")
    print(f"- Active users (Moderately Active, Active & Highly Active): {movers_ratio:.1f}%")
    print(f"- Less active users: {inactive_ratio:.1f}%")

    print("\nTRACKING COVERAGE")
    print(f"- Users logging activity: {tracked_act_users}")
    print(f"- Users logging sleep: {tracked_sleep_users} (out of {tracked_act_users})")

    # -----------------------------------------------------
    # 5. Behavioral patterns
    # -----------------------------------------------------
    day_with_max_steps = act_daily_clean.groupby('weekday')['totalsteps'].mean().idxmax()
    day_with_min_steps = act_daily_clean.groupby('weekday')['totalsteps'].mean().idxmin()
    step_calorie_link = act_daily_clean[['totalsteps', 'calories']].corr().iloc[0, 1]

    print("\nWEEKDAY PATTERNS")
    print(f"- Most active day: {day_with_max_steps}")
    print(f"- Least active day: {day_with_min_steps}")

    print("\nCORRELATIONS")
    print(f"- Steps vs Calories correlation: {step_calorie_link:.3f}")

    # -----------------------------------------------------
    # 6. Return structured results
    # -----------------------------------------------------
    return {
        'mean_steps': mean_steps,
        'mean_sleep_hrs': mean_sleep_hrs,
        'mean_sedentary_hrs': mean_sedentary_hrs,
        'movers_ratio': movers_ratio,
        'inactive_ratio': inactive_ratio,
        'day_with_max_steps': day_with_max_steps,
        'day_with_min_steps': day_with_min_steps,
        'step_calorie_link': step_calorie_link,
        'tracked_act_users': tracked_act_users,
        'tracked_sleep_users': tracked_sleep_users
    }


act_daily_clean, sleep_daily_clean, wt_log_clean = clean_transform_data(act_daily, sleep_daily, wt_log)
user_act_byID, act_distribution, act_percent = act_level_analysis(act_daily_clean)
daily_stats, sleep_pattern = usage_pattern_analysis(act_daily_clean, sleep_daily_clean)
correlation = create_visualizations(act_daily_clean, sleep_daily_clean, act_distribution, daily_stats, sleep_pattern, user_act_byID)
key_findings = calculate_key_insights(act_daily_clean, sleep_daily_clean, user_act_byID)