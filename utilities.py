from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def group_by_hours(y):
    """Group test vector to clusters by 24 hours
    """

    y_hours = []
    for i in range(0, 24):
        temp = []
        for j in range(i, len(y), 24):
            temp.append(y[j])
        y_hours.append(temp)
    return y_hours


def group_by_days(y):
    """Group test vector to clusters by each day
    """
    y_days = []
    for i in range(0, y.shape[0], 24):
        temp = []
        for j in range(i, i + 24):
            if j < y.shape[0]:
                temp.append(y[j])
        y_days.append(temp)
    return y_days

#!/usr/bin/python
# -*- coding: utf-8 -*-


def series_to_supervised(
    data,
    n_in=1,
    n_out=1,
    dropnan=True,
    ):
    """Convert each series to a tuple
    """

    n_vars = (1 if type(data) is list else data.shape[1])
    df = pd.DataFrame(data)
    (cols, names) = (list(), list())

    # input sequence (t-n, ... t-1)

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += ['var%d(t-%d)' % (j + 1, i) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)

    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += ['var%d(t)' % (j + 1) for j in range(n_vars)]
        else:
            names += ['var%d(t+%d)' % (j + 1, i) for j in range(n_vars)]

    # put it all together

    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values

    if dropnan:
        agg.dropna(inplace=True)
    return agg


def load_model_and_predict(test_x = [], path: str = ''):
    """Load model and return test vector
    """
    model = keras.models.load_model(path + "pm25.h5")
    return model.predict(test_x)

def save_model(model, path: str = ''):
    """Save model at {path + metal_name} parameter 
    """

    model.save(path + "pm25.h5")

def calculate_metric(y, yhat, path_save_txt: str = '', path_save_plot: str = '', 
    metal_name: str = '',is_save = True, is_plot = False, type = 'per_hour'):
    """Return metrics between y and yhat, this function can save results as .txt file and graph as .png file
    """

    rmses, mapes, maes = [], [], []
    
    print('Calculating error ...')
    if type == 'per_hour':
        y_hours = group_by_hours(y)
        yhat_hours = group_by_hours(yhat)
        for i in range(0, 24):	
            rmses.append(np.sqrt(mean_squared_error(y_hours[i], yhat_hours[i])))
            mapes.append(mean_absolute_percentage_error(y_hours[i], yhat_hours[i]))
            maes.append(mean_absolute_error(y_hours[i], yhat_hours[i]))
        if is_save:
            metrics = np.stack((range(0, 24), maes, rmses, mapes), axis=1)
            plt.plot(maes, label='mae')
            plt.plot(rmses, label='rmse')
            plt.plot(mapes, label='mape')
            plt.xlabel('Hour')
            plt.ylabel('Value')
    if type == 'per_day':
        y_days = group_by_days(y)
        yhat_days = group_by_days(yhat)
        number_of_day = len(y_days)
        for i in range(0, len(y_days)):	
            rmses.append(np.sqrt(mean_squared_error(y_days[i], yhat_days[i])))
            mapes.append(mean_absolute_percentage_error(y_days[i], yhat_days[i]))
            maes.append(mean_absolute_error(y_days[i], yhat_days[i]))
        if is_save:
            metrics = np.stack((range(366 - number_of_day, 366), maes, rmses, mapes), axis=1)
            plt.plot(list(range(366 - number_of_day, 366)), maes, label='mae')
            plt.plot(list(range(366 - number_of_day, 366)), rmses, label='rmse')
            plt.plot(list(range(366 - number_of_day, 366)), mapes, label='mape')
            plt.xlabel('Day')
            plt.ylabel('Value')
    if is_save:
        np.savetxt(path_save_txt + metal_name + ".csv", metrics, '%5.4f', delimiter=",", header='index,mae,rmse,mape')
        plt.legend()
        plt.title(metal_name)
        plt.savefig(path_save_plot + metal_name + '.png')
    if is_plot:
        plt.show()
    plt.clf()
    return rmses, mapes, maes

def split_dataset(dataset, n_hours: int, n_features: int):
    """Split dataset to train / val / test
    """
    # split into train/val/test: 6/2/2
  
    values = dataset.values
    n_train = 32133
    train = values[:32133, :]
    val = values[32133:42844, :]
    test = values[42844:, :]
    # split into input and outputs
    train_x, train_y = train[:, :-1], train[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]
    val_x, val_y = val[:, :-1], val[:, -1]

    # reshape input to be 3D [samples, timesteps, features]
    train_x = train_x.reshape((train_x.shape[0], n_hours, n_features))
    test_x = test_x.reshape((test_x.shape[0], n_hours, n_features))
    val_x = val_x.reshape((val_x.shape[0], n_hours, n_features))
    return train_x, train_y, val_x, val_y, test_x, test_y

def save_train_history(history, path: str = ''):
    """Save train history to a png file
    """
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('pm25')
    plt.legend()
    plt.savefig(path + 'pm25.png')
    plt.clf()

def save_test_result(y, yhat, path: str = ''):
    if y.shape[0] > 100:
        length = 100
    else:
        length = y.shape[0]
    plt.plot(yhat[:length], label='predict')
    plt.plot(y[:length], label='observed')
    plt.title('pm25')
    plt.legend()
    plt.savefig(path + 'pm25.png')
    plt.clf()



def df_column_switch(df, column1, column2):
    """Swap 2 columns by indices
    """
    i = list(df.columns)
    a, b = i.index(column1), i.index(column2)
    i[b], i[a] = i[a], i[b]
    df = df[i]
    return df

def create_psuedo_metal(mean, std, min_val, number):
    """Create array base on mean and std
    """
    s = np.random.normal(mean, std, number)
    for i in range(0, len(s)):
        if s[i] < 0:
            s[i] = min_val
    return s


def plot_sample(y, yhat_samples, alpha, path_save_plot, is_save=False, is_plot=False):
    upper = np.percentile(yhat_samples, [100 - alpha], axis = 0).reshape(-1)
    lower = np.percentile(yhat_samples, [alpha], axis = 0).reshape(-1)

    plt.plot(y, label = 'observed')
    plt.plot(np.median(yhat_samples, axis=0), label = 'predict')
    plt.title('pm25')
    plt.fill_between(x = range(0, upper.shape[0]), y1 = upper, y2 = lower, alpha= 0.2, color='blue', label = 'credible interval')
    plt.legend()
    if is_save:
        plt.savefig(path_save_plot)
    
    if is_plot:
        plt.show()
    plt.clf()

def calculate_error_sample(y, yhat_samples, alpha):

    yhat = np.median(yhat_samples, axis=0)
    upper = np.percentile(yhat_samples, [100 - alpha], axis = 0).reshape(-1)
    lower = np.percentile(yhat_samples, [alpha], axis = 0).reshape(-1)
    k = 0
    for i in range(0, y.shape[0]):
        if y[i] <= upper[i] and y[i] >= lower[i]:
            k = k + 1
    return k/y.shape[0], mean_absolute_percentage_error(y, yhat)