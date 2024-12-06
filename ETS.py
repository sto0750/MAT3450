import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ETS :
    def __init__(self, data):
        self.data = data
        self.monthly_mean = None
        self.months = 84
        self.flag = 70
        self.monthly_mean_10 = None
        self.monthly_mean_25 = None
        self.fit = None

        self.forecast = None
        self.actual_data = None

    def preset(self):
        self.data['일시'] = pd.to_datetime(self.data['일시'], format='%Y-%m-%d')
        self.data.set_index('일시', inplace=True)
        self.monthly_mean = self.data[['미세먼지(PM10)', '초미세먼지(PM2.5)']].resample('ME').mean()

        # Divide training and test sets
        self.monthly_mean_10 = [self.monthly_mean.iat[k, 0] for k in range(self.flag)]
        self.monthly_mean_25 = [self.monthly_mean.iat[k, 1] for k in range(self.flag)]

    def train(self, trend = "add", seasonal = "add"):
        model = ExponentialSmoothing(self.monthly_mean_10, trend=trend, seasonal=seasonal, seasonal_periods=12)
        self.fit = model.fit()

    def predict(self):
        forecast_steps = self.months - self.flag
        self.forecast = self.fit.forecast(forecast_steps)
        self.actual_data = [self.monthly_mean.iat[k, 0] for k in range(self.flag, self.months)]

    def eval(self):
        mae = mean_absolute_error(self.actual_data, self.forecast)
        mse = mean_squared_error(self.actual_data, self.forecast)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.actual_data, self.forecast)

        # Display results
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R^2 Score: {r2:.4f}")

    def plot(self):
        plt.plot(range(self.flag), self.monthly_mean_10, color='blue')
        plt.plot(range(self.flag), self.fit.fittedvalues, label="Fitted Values", color="red")
        plt.plot(range(self.flag, self.months), self.forecast, label="Forecast", color="green")
        plt.plot(range(self.flag, self.months), self.actual_data, label="Forecast", color=[0.6, 0.6, 0.3])

        plt.show()


# Load data
districts = ['강남구', '강동구', '강북구', '강서구', '관악구', '광진구', '구로구', '금천구', '노원구',
             '도봉구', '동대문구', '동작구', '마포구', '서대문구', '서초구', '성동구', '성북구', '송파구',
             '양천구', '영등포구', '용산구', '은평구', '종로구', '중구', '중랑구']

district_of_interest = districts[1]

df = pd.read_csv('seoul_data_csv/{}_data_filled.csv'.format(district_of_interest))
print(district_of_interest)

ETS = ETS(df)
ETS.preset()
ETS.train()
ETS.predict()
ETS.eval()
ETS.plot()

