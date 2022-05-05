import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from utilities import save_model, save_train_history, series_to_supervised, save_test_result, load_model_and_predict, calculate_metric, split_dataset
n_hours = 24
n_features = 1

list_drop = ['Date (LT)','Year','Month','Day','Hour']
dataset = pd.read_csv('./dataset/pm25modified.csv')
datasets2021 = np.asarray([row for row in dataset.values if row[1] == 2021  and row[4] == 0]) 
n_train = 43848
xlabel = [str(row[3]) + '/' + str(row[2]) for row in datasets2021]
dataset = dataset.drop(list_drop, axis = 1)
dataset = dataset.values
dataset = dataset.astype('float32')
reframed = series_to_supervised(dataset, n_hours, 1)
train_x, train_y, val_x, val_y, test_x, test_y = split_dataset(reframed, n_train, n_hours, n_features)


def create_model(n_hours):
	model = Sequential()
	model.add(LSTM(50, input_shape=(n_hours, 1)))
	model.add(Dropout(0.2))
	model.add(Dense(1))
	model.compile(loss='mae', optimizer='adam')
	return model

is_train, is_save, is_test = False, False, True

if is_train:
    model = create_model(n_hours)
    history = model.fit(train_x, train_y, epochs=30, batch_size=128, validation_data=(val_x, val_y), verbose=2, shuffle=False)
    save_train_history(
        history, 
        path = './LSTMmodified/analysis/train/'
    )
if is_save:
    save_model(
        model, 
        path = './LSTMmodified/model/'
    )
if is_test:
    yhat = load_model_and_predict(
        test_x, 
        path = './LSTMmodified/model/'
    )
    yhat = yhat[1:]
    test_y = test_y[:-1]
    print(yhat.shape)
    save_test_result(
        xlabel,
        test_y, 
        yhat,
        low = 100,
        high = 200,
        path = './LSTMmodified/analysis/test/'
    )
    # rmses, mapes, maes = calculate_metric(
    #     test_y, 
    #     yhat, 
    #     path_save_txt = './LSTMmodified/analysis/error_per_hour/',
    #     path_save_plot = './LSTMmodified/analysis/error_per_hour_plot/',
    #     is_save = True, is_plot = False,
    #     type = 'per_hour'
    # )

    rmses, mapes, maes = calculate_metric(
        test_y, 
        yhat, 
        path_save_txt = './LSTMmodified/analysis/error_per_day/',
        path_save_plot = './LSTMmodified/analysis/error_per_day_plot/',
        is_save = True, is_plot = False,
        type = 'per_day'
    )

    metrics = np.stack((test_y, np.squeeze(yhat)), axis=1)
    np.savetxt('./LSTMmodified/analysis/error/' + "pm25.csv", metrics, '%5.4f', delimiter=",", header='observed, predict')


#     print(np.mean(mapes))
