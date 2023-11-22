import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
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

# Split the data into training, test and validation sets. I use a hard cutoff date to ensure that each set starts and ends at whole days. The spit is around 70% for training, 15% for validation and 15% for test.
val_cutoff = pd.to_datetime('2017-10-20', utc=True, infer_datetime_format=True)
test_cutoff = pd.to_datetime('2018-05-27', utc=True, infer_datetime_format=True)

train_df = df_final.loc[df_final.index < val_cutoff]
validation_df = df_final.loc[(df_final.index >= val_cutoff) & (df_final.index < test_cutoff)]
test_df = df_final.loc[df_final.index >= test_cutoff]

# Separate feature and target variables.
X_train = train_df.drop(columns=['price actual'])
y_train = train_df['price actual']

X_validation = validation_df.drop(columns=['price actual'])
y_validation = validation_df['price actual']

X_test = test_df.drop(columns=['price actual'])
y_test = test_df['price actual']

# Scale the data.
from sklearn.preprocessing import MinMaxScaler

# Scale the input features.
scaler_X = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = scaler_X.fit_transform(X_train)
X_validation_scaled = scaler_X.transform(X_validation)
X_test_scaled = scaler_X.transform(X_test)

# Convert y_train to a NumPy array and reshape.
y_train_array = np.array(y_train).reshape(-1, 1)
y_validation_array = np.array(y_validation).reshape(-1, 1)

# Scale the target variable.
scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_train_scaled = scaler_y.fit_transform(y_train_array)
y_validation_scaled = scaler_y.transform(y_validation_array)

# Reshape the data to be 3-dimensional in the form [samples, timesteps, features].
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_validation_reshaped = X_validation_scaled.reshape((X_validation_scaled.shape[0], 1, X_validation_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build the LSTM model.
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten

model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model.add(Flatten())
model.add(Dense(200))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model with scaled target variable.
model.fit(X_train_reshaped, y_train_scaled, epochs=50, validation_data=(X_validation_reshaped, y_validation_scaled), verbose=2)

# Make predictions.
y_pred = model.predict(X_test_reshaped)

# Inverse transform the predictions to the original scale.
y_pred_original_scale = scaler_y.inverse_transform(y_pred)

mse = mean_squared_error(y_test, y_pred_original_scale)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_original_scale)
print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % mae)
