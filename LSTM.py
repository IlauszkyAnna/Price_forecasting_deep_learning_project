import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from hyperopt import fmin, tpe, hp

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

# Separate feature and target variables. Make target variables into np.array for minmaxscaler.
X_train = np.array(train_df.drop(columns=['price actual']))
y_train = np.array(train_df['price actual']).reshape(-1,1)

X_validation = np.array(validation_df.drop(columns=['price actual']))
y_validation = np.array(validation_df['price actual']).reshape(-1,1)

X_test = np.array(test_df.drop(columns=['price actual']))
y_test = np.array(test_df['price actual']).reshape(-1,1)

# Scale the input features.
scaler_X = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = scaler_X.fit_transform(X_train)
X_validation_scaled = scaler_X.transform(X_validation)
X_test_scaled = scaler_X.transform(X_test)

# Scale the target variable.
scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_train_scaled = scaler_y.fit_transform(y_train)
y_validation_scaled = scaler_y.transform(y_validation)
y_test_scaled = scaler_y.transform(y_test)

# Create a helper function for the reshaping of the x and y variables. We want to predict the next day's electricity price based on the data of the last 7 days. So we have to create sequences of the desired length.
def create_sequences(data, target, seq_length, step = 24):
    X_sequences, y_sequences = [], []
    for i in range(0, len(data) - seq_length, step):
        X_seq = data[i:i+seq_length]
        y_seq = target[i + seq_length:i+(seq_length // 7) + seq_length]
        X_sequences.append(X_seq)
        y_sequences.append(y_seq)
    return np.array(X_sequences), np.array(y_sequences)

# Reshape the data to be 3-dimensional in the form [samples, timesteps, features].
X_train_reshaped, y_train_reshaped = create_sequences(X_train_scaled, y_train_scaled, 168)
X_validation_reshaped, y_validation_reshaped = create_sequences(X_validation_scaled, y_validation_scaled, 168)
X_test_reshaped, y_test_reshaped = create_sequences(X_test_scaled, y_test_scaled, 168)

# Use Tree Parzen Estimator to tune the hyperparameters for the model. Reference: https://towardsdatascience.com/algorithms-for-hyperparameter-optimisation-in-python-edda4bdb167 for the implementation of the TPE.
# Define the hyperparameter search space.
space = {
    'lstm_units': hp.choice('lstm_units', [128, 256, 512]),
    'dense_units': hp.choice('dense_units', [64, 128, 256]),
    'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.5),
    'epochs': hp.choice('epochs', [50, 100, 150]),
    'batch_size': hp.choice('batch_size', [32, 64, 128])
}

# Define the objective function to minimize.
def objective(params):
    model = Sequential()
    model.add(LSTM(params['lstm_units'], activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), return_sequences=False))
    model.add(Flatten())
    model.add(Dense(params['dense_units']))
    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(24))
    model.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    
    checkpoint = ModelCheckpoint('best_model_weights.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    model.fit(X_train_reshaped, y_train_reshaped, epochs=params['epochs'], batch_size=params['batch_size'],
              validation_data=(X_validation_reshaped, y_validation_reshaped), callbacks=[early_stopping, checkpoint], verbose=0)
    
    model.load_weights('best_model_weights.h5')
    
    y_pred = model.predict(X_validation_reshaped).flatten()
    
    y_pred_original_scale = scaler_y.inverse_transform(y_pred.reshape(-1, 1))

    mse = mean_squared_error(y_validation[168:], y_pred_original_scale)
    return mse

# Use the fmin function to find the best hyperparameters.
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50)

# Print the best hyperparameters.
print("Best Hyperparameters:", best)

# Use the best hyperparameters to train the final model.
best_lstm_units = [128, 256, 512][best['lstm_units']]
best_dense_units = [64, 128, 256][best['dense_units']]
best_epochs = [50, 100, 150][best['epochs']]
best_batch_size = [32, 64, 128][best['batch_size']]
best_dropout_rate = best['dropout_rate']

final_model = Sequential()
final_model.add(LSTM(best_lstm_units, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), return_sequences=False))
final_model.add(Flatten())
final_model.add(Dense(best_dense_units))
final_model.add(Dropout(best_dropout_rate))
final_model.add(Dense(24))
final_model.compile(optimizer='adam', loss='mse')

# Define early stopping and model checkpoint criteria.
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

final_checkpoint = ModelCheckpoint('final_model_weights.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Fit the model.
final_model.fit(X_train_reshaped, y_train_reshaped, epochs=best_epochs, batch_size=best_batch_size,
                validation_data=(X_validation_reshaped, y_validation_reshaped), callbacks=[early_stopping, final_checkpoint], verbose=2)

# Load the best weights from the ModelCheckpoint criteria.
final_model.load_weights('final_model_weights.h5')

# Make final predictions.
y_pred_final = final_model.predict(X_test_reshaped).flatten()

# Inverse transform the predictions to the original scale.
y_pred_final_original_scale = scaler_y.inverse_transform(y_pred_final.reshape(-1, 1))

# Evaluate the final model.
mse_final = mean_squared_error(y_test[168:], y_pred_final_original_scale)
rmse_final = np.sqrt(mse_final)
mae_final = mean_absolute_error(y_test[168:], y_pred_final_original_scale)
print('Final Test RMSE: %.3f' % rmse_final)
print('Final Test MAE: %.3f' % mae_final)
