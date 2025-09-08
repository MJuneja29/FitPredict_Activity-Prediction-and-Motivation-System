# ==============================================================================
# Fitbit Daily Activity - Deep Dive Analysis
#
# Project:      IIT Ropar Capstone Project
#
# Purpose:      This script performs a detailed exploratory data analysis (EDA)
#               specifically on the merged `dailyActivity` dataset. It goes deeper
#               than the initial summary, focusing on:
#               1. Feature Engineering (e.g., weekday vs. weekend).
#               2. Distribution analysis using box plots.
#               3. Detailed examination of activity types and user consistency.
# ==============================================================================

# --- Step 1: Import all the necessary libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# --- Step 2: Load the Cleaned, Merged Dataset ---
# This script assumes you have already run the initial merging script.
try:
    dailyActivity_analysis = pd.read_csv(r"Data/Merged_data/activity_data.csv")
    print("Merged daily activity data loaded successfully.")
except FileNotFoundError:
    print("ERROR: 'activity_data.csv' not found. Please run the initial data merging script first.")
    exit()

# ==============================================================================
# SECTION 1: DATA PREPARATION AND FEATURE ENGINEERING
# ==============================================================================

# --- Ensure correct data types for analysis ---
# Proper data types are crucial for time-series and statistical functions.
dailyActivity_analysis["ActivityDate"] = pd.to_datetime(dailyActivity_analysis["ActivityDate"])

# --- Engineer new time-based features ---
# These new columns allow us to analyze patterns based on the day of the week or month.
dailyActivity_analysis["day_name"] = dailyActivity_analysis["ActivityDate"].dt.day_name()
dailyActivity_analysis["month_name"] = dailyActivity_analysis["ActivityDate"].dt.month_name()

# This binary feature is great for testing hypotheses like "Are users more active on weekends?"
dailyActivity_analysis['Weekend_or_Weekday'] = dailyActivity_analysis['day_name'].apply(
    lambda x: 'Weekend' if x in ['Saturday', 'Sunday'] else 'Weekday'
)
print("New time-based features created (day_name, Weekend_or_Weekday).")

# ==============================================================================
# SECTION 2: DISTRIBUTION AND OUTLIER ANALYSIS
#
# Using box plots to understand the spread, median, and potential outliers
# for key metrics across different days of the week.
# ==============================================================================

# Define a consistent order for the days of the week for all plots.
days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
sns.set_style("whitegrid")

# --- Box Plot: Total Steps Distribution by Day ---
# A box plot is more informative than a bar chart of the mean because it shows
# the median, interquartile range (IQR), and outliers.
plt.figure(figsize=(12, 7))
sns.boxplot(x='day_name', y='TotalSteps', data=dailyActivity_analysis, order=days_order, palette="pastel")
plt.title("Distribution of Total Steps by Day of the Week", fontsize=16, fontweight='bold')
plt.xlabel("Day of the Week", fontsize=12)
plt.ylabel("Total Steps", fontsize=12)
plt.xticks(rotation=45)
plt.show()

# --- Box Plot: Calories Burned Distribution by Day ---
plt.figure(figsize=(12, 7))
sns.boxplot(x='day_name', y='Calories', data=dailyActivity_analysis, order=days_order, palette="pastel")
plt.title("Distribution of Calories Burned by Day of the Week", fontsize=16, fontweight='bold')
plt.xlabel("Day of the Week", fontsize=12)
plt.ylabel("Total Calories Burned", fontsize=12)
plt.xticks(rotation=45)
plt.show()

# ==============================================================================
# SECTION 3: ANALYSIS OF ACTIVITY TYPES AND PROPORTIONS
# ==============================================================================

# --- Bar Chart: Average Time Spent in Different Activity Levels ---
# This visualizes how a typical user's 24 hours are divided.
minutes_means = dailyActivity_analysis[['VeryActiveMinutes', 'FairlyActiveMinutes', 'LightlyActiveMinutes', 'SedentaryMinutes']].mean()
plt.figure(figsize=(12, 7))
sns.barplot(x=minutes_means.index, y=minutes_means.values, palette="viridis")
plt.title('Average Daily Time Spent in Each Activity Level', fontsize=16, fontweight='bold')
plt.ylabel('Average Minutes per Day')
plt.xlabel('Activity Level')
# Clean up the x-axis labels for better readability.
plt.xticks(ticks=range(len(minutes_means.index)),
           labels=[label.replace('Minutes', ' Minutes') for label in minutes_means.index],
           rotation=45)
plt.show()

# --- Pie Chart: Proportion of Total Distance by Activity Intensity ---
# This helps us understand which intensity level contributes the most to a user's total distance.
distance_sums = dailyActivity_analysis[['VeryActiveDistance', 'ModeratelyActiveDistance', 'LightActiveDistance']].sum()
labels = ['Very Active', 'Moderately Active', 'Lightly Active']
colors = ['#ff6666', '#ffcc99', '#99ff99']
explode = (0.05, 0, 0) # Slightly separate the most intense category.

plt.figure(figsize=(10, 10))
plt.pie(distance_sums, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, explode=explode,
        textprops={'fontsize': 12})
plt.title('Proportion of Total Distance by Activity Intensity', fontsize=16, fontweight='bold')
plt.ylabel('')
plt.show()



# ==============================================================================
# SECTION 4: DEEPER INSIGHTS AND RELATIONSHIPS
# ==============================================================================

# --- Scatter Plot: Very Active Minutes vs. Calories Burned ---
# This plot helps to visually confirm the strong relationship between high-intensity
# activity and calorie expenditure.
plt.figure(figsize=(12, 7))
sns.scatterplot(x='VeryActiveMinutes', y='Calories', data=dailyActivity_analysis, alpha=0.6)
plt.xlabel('Very Active Minutes per Day')
plt.ylabel('Total Calories Burned')
plt.title('Relationship between Very Active Minutes and Calories Burned', fontsize=16, fontweight='bold')
plt.grid(True)
plt.show()

