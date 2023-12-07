import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.dates as mdates

# I want to reference the kaggle notebook for the preproccessing of the data given in: https://www.kaggle.com/code/dimitriosroussis/electricity-price-forecasting-with-dnns-eda/notebook

# Read the datasets.
energy_df = pd.read_csv('energy_dataset.csv')
weather_df = pd.read_csv('weather_features.csv')

# Convert "time" and "dt_iso" to be datetime instead of object. Convert to UTC to take care of DST and add a timedelta of one hour to match the original start date.
energy_df['time'] = pd.to_datetime(energy_df['time'], utc=True, infer_datetime_format=True) + timedelta(hours=1)
weather_df['dt_iso'] = pd.to_datetime(weather_df['dt_iso'], utc=True, infer_datetime_format=True) + timedelta(hours=1)

# Set the indexes of the dataframes to be "time".
energy_df.set_index('time', inplace=True)
weather_df.rename(columns={'dt_iso': 'time'}, inplace=True)
weather_df.set_index('time', inplace=True)

# Drop columns from weather_df, that we are not going to use.
weather_df = weather_df[['city_name', 'temp']]

# Drop columns from energy that we cannot use since they are actuals. Forecast wind offshore only contains NaN's.
# We keep "price actual" since it is our target variable.
energy_df = energy_df[['forecast solar day ahead', 'forecast wind onshore day ahead', 'total load forecast', 'price day ahead', 'price actual']]

# Drop columns filled with zeros or NaN's.
energy_df.dropna(axis=1, how='all', inplace=True)

# Fill null values in energy_df with interpolation.
energy_df.interpolate(method='linear', limit_direction='forward', inplace=True, axis=0)

# Drop duplicate values for each city in weather_df.
weather_df = weather_df.reset_index().drop_duplicates(subset=['time', 'city_name'], keep='first').set_index('time')

# Split the df_weather into 5 dataframes (one for each city).
dfs = [group for _, group in weather_df.groupby('city_name')]

# Merge the energy dataframe and the weather dataframe.
df_final = energy_df

for df in dfs:
    city = df['city_name'].unique()
    city_str = str(city).replace("'", "").replace('[', '').replace(']', '').replace(' ', '')
    df = df.add_suffix('_{}'.format(city_str))
    df_final = df_final.merge(df, on=['time'], how='outer')
    df_final = df_final.drop('city_name_{}'.format(city_str), axis=1)
    
# Make the temp feature weighted by population of the cities. Population data is taken from: https://en.wikipedia.org/wiki/List_of_metropolitan_areas_in_Spain
populations = [6155116, 5179243, 1645342, 1305342, 987000]
total_pop = sum(populations)

weights = [pop / total_pop for pop in populations]

weight_Madrid, weight_Barcelona, weight_Valencia, weight_Seville, weight_Bilbao = weights

df_final[['temp_Madrid', 'temp_Barcelona', 'temp_Valencia', 'temp_Seville', 'temp_Bilbao']] *= weights
    
# Create dummy variables for month, weekday and hour.
df_final['hour'] = df_final.index.hour
df_final['weekday'] = df_final.index.weekday
df_final['month'] = df_final.index.month

# Plot the time series.
fig, axs = plt.subplots(3, 1, figsize=(12, 10))

# Plot for the entire time series.
x_all = df_final.index
y_all = df_final['price actual']
axs[0].plot(x_all, y_all, label='Spot price')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Actual Price (€/MWh)')
axs[0].set_title('Time series of the electricity price for the whole period, one month and one week')
axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"));
axs[0].grid(True)
axs[0].legend()

# Plot for one month.
start_date_month = df_final.index[0]
end_date_month = start_date_month + pd.DateOffset(months=1)
x_month = df_final[start_date_month:end_date_month].index
y_month = df_final[start_date_month:end_date_month]['price actual']
axs[1].plot(x_month, y_month, label='Spot price')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Actual Price (€/MWh)')
axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d %Y"));
axs[1].grid(True)
axs[1].legend()

# Plot for one week.
start_date_week = df_final.index[0]
end_date_week = start_date_week + pd.DateOffset(weeks=1)
x_week = df_final[start_date_week:end_date_week].index
y_week = df_final[start_date_week:end_date_week]['price actual']
axs[2].plot(x_week, y_week, label='Spot price')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Actual Price (€/MWh)')
axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"));
axs[2].grid(True)
axs[2].legend()

# Show the combined plot.
plt.savefig('Price.png', format='png')
plt.show()

# ADF test to see if the data is stationary.
adf = adfuller(df_final['price actual'])

# Extract and print the results. We reject the null hypothesis under a 1% significance level and hence the time series is stationary.
print("ADF Statistic:", adf[0])
print("p-value:", adf[1])
print("Critical Values:")
for key, value in adf[4].items():
    print(f"{key}: {value}")

# Plot ACF and PACF.
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 6))
plot_acf(df_final['price actual'], lags=48, ax=ax1)
plot_pacf(df_final['price actual'], lags=48, ax=ax2, method = 'ywm')
plt.tight_layout()
plt.show()

# Split the data into training, test and validation sets. I use a hard cutoff date to ensure that each set starts and ends at whole days. The spit is around 70% for training, 15% for validation and 15% for test.
val_cutoff = pd.to_datetime('2017-10-20', utc=True, infer_datetime_format=True)
test_cutoff = pd.to_datetime('2018-05-27', utc=True, infer_datetime_format=True)

train_df = df_final.loc[df_final.index < val_cutoff]
validation_df = df_final.loc[(df_final.index >= val_cutoff) & (df_final.index < test_cutoff)]
test_df = df_final.loc[df_final.index >= test_cutoff]

# Separate feature and target variables
X_train = train_df.drop(columns=['price actual'])
y_train = train_df['price actual']

X_validation = validation_df.drop(columns=['price actual'])
y_validation = validation_df['price actual']

X_test = test_df.drop(columns=['price actual'])
y_test = test_df['price actual']


# Persistence model forecast. Made using the reference: https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
# Walk-forward validation.
history = df_final.loc['2018-05-26']['price actual'].tolist()
predictions = []

for i in range(len(y_test)):
    # Make prediction by using the 24th lagged value in history.
    prediction = history[-24]
    predictions.append(prediction)
    
    # Add the current test value to the history for the next iteration.
    history.append(y_test[i])
    
# Make the predictions to a dataframe.
persistence_predictions = pd.DataFrame(predictions, columns=["prediction"])
persistence_predictions.reset_index(drop=True, inplace=True)
persistence_predictions.index=y_test.index
persistence_predictions['actual'] = y_test

# Calculate MSE and MAE.
persistence_rmse = np.sqrt(mean_squared_error(persistence_predictions['actual'], persistence_predictions['prediction']))
persistence_mae = mean_absolute_error(persistence_predictions['actual'], persistence_predictions['prediction'])
print("RMSE Persistence Model:", persistence_rmse)
print("MAE Persistence Model:", persistence_mae)

# Plot the predictions versus the actuals.
# Create a figure and axis objects.
fig, axs = plt.subplots(3, 1, figsize=(12, 10))

# Plot entire time series
axs[0].plot(persistence_predictions.index, persistence_predictions['actual'], color='blue', label = 'actual')
axs[0].plot(persistence_predictions.index, persistence_predictions['prediction'], color='red', label = 'predictions')
axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"));
axs[0].set_title('Predictions and actual values for the whole out-of-sample period, one month and one week')
axs[0].legend()
axs[0].grid(True)

# Plot one month of data
axs[1].plot(persistence_predictions.loc['2018-06-01':'2018-07-01'].index, persistence_predictions.loc['2018-06-01':'2018-07-01']['actual'], color='blue')
axs[1].plot(persistence_predictions.loc['2018-06-01':'2018-07-01'].index, persistence_predictions.loc['2018-06-01':'2018-07-01']['prediction'], color ='red')
axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"));
axs[1].grid(True)

# Plot one week of data
axs[2].plot(persistence_predictions.loc['2018-06-01':'2018-06-08'].index, persistence_predictions.loc['2018-06-01':'2018-06-08']['actual'], color='blue')
axs[2].plot(persistence_predictions.loc['2018-06-01':'2018-06-08'].index, persistence_predictions.loc['2018-06-01':'2018-06-08']['prediction'], color='red')
axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"));
axs[2].grid(True)

# save the plot as an PNG file
plt.savefig('Persistence.png', format='png')
plt.show()
