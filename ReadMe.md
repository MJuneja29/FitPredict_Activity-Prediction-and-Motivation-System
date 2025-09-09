# Fitbit Calorie Prediction & Daily Goal Forecaster

**A Capstone Project for IIT Ropar**

## Project Overview

This project leverages machine learning to forecast daily calorie expenditure for Fitbit users. Using a time-series dataset of user activity, the primary goal is to build and evaluate a system that can predict, early in the day, whether a user is on track to meet their daily calorie target.

The project follows a complete data science workflow, starting with in-depth Exploratory Data Analysis (EDA) to understand user behavior, followed by rigorous data preprocessing and feature engineering. A Long Short-Term Memory (LSTM) neural network is trained to predict hourly calorie burn. This hourly model is then used as the engine for a daily forecaster, which is systematically evaluated for both its regression accuracy and its classification performance.

The final output is a predictive tool that provides a reliable YES/NO verdict on daily goal achievement, demonstrating a practical application of time-series forecasting in the personal fitness domain.



## Key Features

-   **In-Depth EDA:** Comprehensive analysis of user behavior, including activity levels, weekday patterns, and persona identification (e.g., "Consistent Users" vs. "Weekend Warriors").
-   **Data Preprocessing:** Robust pipeline for cleaning, transforming, and merging multiple granular Fitbit datasets (calories, steps, sleep, heart rate).
-   **Feature Engineering:** Creation of valuable hourly features from minute-level sleep data and second-level heart rate data, including a personalized 'StressLevel' metric.
-   **LSTM Time-Series Model:** A Keras/TensorFlow-based LSTM model trained to predict hourly calorie burn based on a 24-hour sequence of historical activity.
-   **Daily Forecast System:** An intelligent script that uses the hourly model to iteratively forecast the remainder of the day and predict daily goal achievement.
-   **Rigorous Evaluation:** A systematic evaluation of the final forecast model on an unseen test set, reporting both regression (MAE, RMSE) and classification (Accuracy, Precision, Recall, F1-Score) metrics.



## Tech Stack

-   **Language:** Python
-   **Core Libraries:** Pandas, NumPy
-   **Machine Learning:** TensorFlow (with Keras API), Scikit-learn
-   **Data Visualization:** Matplotlib, Seaborn
-   **Utilities:** Joblib (for saving/loading model assets)



## Project Structure

This repository is organized into a series of scripts, each performing a distinct phase of the project:

-   `comparison_first_vs_sec_month`: Performs a preliminary analysis to compare user consistency across the two monthly data dumps, justifying the decision to use the second month's data for modeling.
-   `data_analysis` & `data_deep_dive.py`: These scripts conduct a thorough Exploratory Data Analysis to uncover patterns and insights from the user data.
-   `cleaning_and_transformation.py`: The main data pipeline script. It cleans, transforms, engineers features, and merges all the source files into a single, model-ready `final_hourly.csv`.
-   `train_model.py`: This script takes the preprocessed data, builds the LSTM model, trains it using `EarlyStopping` to find the optimal version, evaluates it, and saves the final model and all necessary assets (scalers, test data) to the `/src` directory.
-   `predict.py`: A dedicated script to rigorously evaluate the saved model's daily forecasting ability on the unseen test set, producing a full classification and regression report, demonstrating the model's practical use.



## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MJuneja29/FitPredict_Activity-Prediction-and-Motivation-System.git
    cd FitPredict_Activity-Prediction-and-Motivation-System
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    A `requirements.txt` file is provided. Install all dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```
    

4.  **Data:**
    Place the raw Fitbit datasets into a `Data/` directory as structured in the scripts.



## How to Run the Project

Execute the scripts in the following order to replicate the project workflow:

**Step 1: Data Preprocessing**
Run the preprocessing script to clean, engineer features, and create the final `final_hourly.csv` dataset.
```bash
python cleaning_and_transformation.py
```

**Step 2: Train the LSTM Model**
This script will train the model, find the best epoch using early stopping, and save the final model and all assets into a `/src` folder.
```bash
python train_model.py
```

**Step 3: Evaluate the Forecast Model**
Run the evaluation script to get a detailed performance report on the model's ability to predict daily goal achievement.
```bash
python predict.py
```



## Results and Evaluation

The final daily forecast model was evaluated on an unseen test set by simulating a prediction at 3 PM for each day. The model demonstrated strong performance in its primary goal of classifying whether a user would meet their daily calorie target.

#### Key Performance Metrics:

-   **Accuracy:** **89.67%**
    -   *The model's YES/NO verdict on meeting the daily goal is correct nearly 9 out of 10 times.*
-   **Precision:** **83.78%**
    -   *When the model predicts a user will succeed, it is correct 84% of the time, indicating high reliability.*
-   **Recall:** **86.11%**
    -   *The model successfully identifies over 86% of the days where the user actually met their goal.*
-   **F1-Score:** **84.93%**
    -   *Shows a strong, balanced performance between Precision and Recall.*
-   **Mean Absolute Error (MAE):** **334.82 calories**
    -   *On average, the model's projection of the final daily calorie total is off by ~335 calories. This is primarily influenced by the model's difficulty in predicting the exact magnitude of spontaneous, high-intensity activities.*




## Future Considerations and Enhancements

While this project successfully demonstrates a robust forecasting model based on historical data, its true potential lies in its application as a real-time, interactive user-coaching system. The following are key areas for future development to transform this analysis into a dynamic, real-world product.

### 1. Proactive Coaching with a Generative AI Engine

-   **Concept:** Integrate the numerical forecast results with a **Generative AI model (like GPT-4)** via its API. Instead of just a YES/NO verdict, the system could provide dynamic, encouraging, and context-aware motivational messages.
-   **Impact:** This transforms the system from a data tool into a personalized AI coach. For example, if a user has a 300-calorie deficit at 4 PM, the AI could generate a message like:
 
    > *"Great effort so far today! It looks like you're about 300 calories short of your goal. A brisk 30-minute walk after dinner would be a perfect way to close the gap. You can do it!"*

### 2. Developing a User-Facing Application

-   **Concept:** Package the prediction logic into a backend service (e.g., using **Flask or FastAPI**) and connect it to a user-facing application.
-   **Impact:** This would deliver the insights directly to the user in a convenient format, such as:
    -   A **mobile app** that sends proactive push notifications.
    -   A **web dashboard** for detailed daily and weekly trend analysis.
    -   A **chatbot** (e.g., on Telegram or WhatsApp) for interactive, on-demand coaching.

### 3. Advanced Model Enhancements

-   **Concept:** Further improve the model's predictive accuracy, especially in reducing the Mean Absolute Error (MAE) for the final calorie number.
-   **Impact:** A more accurate model leads to more precise suggestions.
    -   **Richer Feature Engineering:** Incorporate more granular data that is often available through APIs, such as detailed **sleep stages (REM, deep, light)** and **Heart Rate Variability (HRV)** to better model user fatigue and recovery.
    -   **Exploring Transformer Models:** Investigate state-of-the-art time-series architectures like **Transformers**, which may capture long-term dependencies in user behavior even more effectively than LSTMs.