import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

class Regression_model :
    def __init__(self, data):
        self.data = data
        self.train_data_2008_2021 = None
        self.test_data_2022 = None
        self.train_data_2008_2021_monthly = None
        self.test_data_2022_monthly = None

    # Convert '일시' to datetime and set it as the index
    def convert_datetime(self):
        self.data['일시'] = pd.to_datetime(self.data['일시'])
        self.data.set_index('일시', inplace=True)

    # Filter data for training (2008-2021) and testing (2022)
    def filter_data(self):
        self.train_data_2008_2021 = self.data['2008-01-01':'2021-12-31']
        self.test_data_2022 = self.data['2022-01-01':'2022-12-31']

    # Resample the data to get the monthly averages
    def resample(self):
        self.train_data_2008_2021_monthly = self.train_data_2008_2021[['미세먼지(PM10)', '초미세먼지(PM2.5)']].resample('ME').mean()
        self.test_data_2022_monthly = self.test_data_2022[['미세먼지(PM10)', '초미세먼지(PM2.5)']].resample('ME').mean()


    def prepare_train(self):
        # Prepare the month-based dummy variables for training and testing
        self.train_data_2008_2021_monthly['month'] = self.train_data_2008_2021_monthly.index.month
        self.test_data_2022_monthly['month'] = self.test_data_2022_monthly.index.month

        self.dummy_vars_train = pd.get_dummies(self.train_data_2008_2021_monthly['month'], prefix='month')
        self.dummy_vars_test = pd.get_dummies(self.test_data_2022_monthly['month'], prefix='month')

        # Prepare the target variables (PM10 and PM2.5) for training
        y_train_pm10 = self.train_data_2008_2021_monthly['미세먼지(PM10)'].values
        y_train_pm25 = self.train_data_2008_2021_monthly['초미세먼지(PM2.5)'].values

        # Train linear regression models for PM10 and PM2.5 using dummy variables
        self.model_pm10_dummy_2008_2021 = LinearRegression()
        self.model_pm10_dummy_2008_2021.fit(self.dummy_vars_train, y_train_pm10)

        self.model_pm25_dummy_2008_2021 = LinearRegression()
        self.model_pm25_dummy_2008_2021.fit(self.dummy_vars_train, y_train_pm25)

    def predict_eval(self):
        # Predict using the trained model for PM10 and PM2.5 in 2022
        self.y_pred_pm10_dummy_2022 = self.model_pm10_dummy_2008_2021.predict(self.dummy_vars_test)
        self.y_pred_pm25_dummy_2022 = self.model_pm25_dummy_2008_2021.predict(self.dummy_vars_test)

        # Calculate evaluation metrics for PM10 and PM2.5
        mae_pm10 = mean_absolute_error(self.test_data_2022_monthly['미세먼지(PM10)'], self.y_pred_pm10_dummy_2022)
        mse_pm10 = mean_squared_error(self.test_data_2022_monthly['미세먼지(PM10)'], self.y_pred_pm10_dummy_2022)
        rmse_pm10 = math.sqrt(mse_pm10)
        r2_pm10 = r2_score(self.test_data_2022_monthly['미세먼지(PM10)'], self.y_pred_pm10_dummy_2022)

        mae_pm25 = mean_absolute_error(self.test_data_2022_monthly['초미세먼지(PM2.5)'], self.y_pred_pm25_dummy_2022)
        mse_pm25 = mean_squared_error(self.test_data_2022_monthly['초미세먼지(PM2.5)'], self.y_pred_pm25_dummy_2022)
        rmse_pm25 = math.sqrt(mse_pm25)
        r2_pm25 = r2_score(self.test_data_2022_monthly['초미세먼지(PM2.5)'], self.y_pred_pm25_dummy_2022)

        # Print the performance metrics for PM10 and PM2.5
        print("PM10 Performance Metrics (2022):")
        print(f"MAE: {mae_pm10:.2f}")
        print(f"MSE: {mse_pm10:.2f}")
        print(f"RMSE: {rmse_pm10:.2f}")
        print(f"R^2: {r2_pm10:.2f}")

        print("\nPM2.5 Performance Metrics (2022):")
        print(f"MAE: {mae_pm25:.2f}")
        print(f"MSE: {mse_pm25:.2f}")
        print(f"RMSE: {rmse_pm25:.2f}")
        print(f"R^2: {r2_pm25:.2f}")

    # Plot
    def plot(self):
        # Plot the actual values for 2016-2022 and predicted values for 2022
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # PM10 plot for 2016-2022 (actual values) and predicted values for 2022
        # axes[0].plot(train_data_2016_2021_monthly.index, y_train_pm10, label='Actual PM10 (2016-2022)', color='blue')
        axes[0].plot(self.test_data_2022_monthly.index, self.test_data_2022_monthly['미세먼지(PM10)'], label='Actual PM10 (2022)',
                     color='blue')
        axes[0].plot(self.test_data_2022_monthly.index, self.y_pred_pm10_dummy_2022, label='Predicted PM10 (2022)', color='red',
                     linestyle='--')
        axes[0].set_title('PM10: Actual vs Predicted (Dummy Regression) 2022.01 - 2022.12')
        axes[0].legend()

        # PM2.5 plot for 2016-2022 (actual values) and predicted values for 2022
        # axes[1].plot(train_data_2016_2021_monthly.index, y_train_pm25, label='Actual PM2.5 (2016-2022)', color='green')
        axes[1].plot(self.test_data_2022_monthly.index, self.test_data_2022_monthly['초미세먼지(PM2.5)'], label='Actual PM2.5 (2022)',
                     color='green')
        axes[1].plot(self.test_data_2022_monthly.index, self.y_pred_pm25_dummy_2022, label='Predicted PM2.5 (2022)',
                     color='orange', linestyle='--')
        axes[1].set_title('PM2.5: Actual vs Predicted (Dummy Regression) 2022.01 - 2022.12')
        axes[1].legend()

        plt.tight_layout()
        plt.show()