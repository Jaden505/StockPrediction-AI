import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from keras import models, layers
from keras.preprocessing.sequence import TimeseriesGenerator
import pandas as pd
from datetime import timedelta, date
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from math import sqrt

class FormData:
    def __init__(self, cp):
        self.cp = cp

    def getHistory(self):
        ticker = yf.Ticker(self.cp)
        return ticker.history(period='max', interval='1d')

    def normalize(self, h):
        h_min = h.min()
        normalized = (h - h_min) / (h.max() - h_min)

        return normalized

    def generateDataset(self, history, input_amount):
        close = history['Close']
        dividens = history['Dividends']
        tsg = TimeseriesGenerator(close, close,
                                  length=input_amount,
                                  batch_size=len(close))
        i, t = tsg[0]
        has_dividens = np.zeros(len(i))

        for b_row in range(len(t)):
            assert(abs(t[b_row] - close[input_amount]) <= 0.001)
            has_dividens[b_row] = dividens[input_amount] > 0
            input_amount += 1

        return np.concatenate((i, np.transpose([has_dividens])),
                              axis=1), t

class NeuralNet:
    def createModel(self, n):
        m = models.Sequential()
        m.add(layers.Dense(64, activation='relu', input_shape=(n+1,)))
        m.add(layers.Dense(64, activation='relu'))
        m.add(layers.Dense(1))
        return m

    def selectInputs(self, data, start, end, epochs):
        models = {}

        for inputs in range(start, end + 1):
            print('Using {} inputs'.format(inputs))
            model_inputs, targets = dat.generateDataset(data, inputs)

            train_inputs = model_inputs[:-1000]
            val_inputs = model_inputs[-1000:]
            train_targets = targets[:-1000]
            val_targets = targets[-1000:]

            m = self.createModel(inputs)
            print('Training')
            m.compile(optimizer='adam', loss='mse')
            h = m.fit(train_inputs, train_targets,
                      epochs=epochs,
                      batch_size=10,
                      validation_data=(val_inputs, val_targets))

            model_info = {'model': m, 'history': h.history}
            models['info'] = model_info

        lowest_loss_model = min([m['history']['loss'][-1] for m in models.values()])

        best_model = [v for k,v in models.items() if v['history']['loss'][-1] == lowest_loss_model]

        return best_model

# def futureDF(vals):
#     # 3 days in future
#     today = date.today()
#     days = [(today - timedelta(days=ind)).strftime("%Y-%m-%d") for x, ind in enumerate(range(1, len(vals)+1))]
#
#     vals = [v[0] for v in vals]
#     print(type(days), type(days.reverse()))
#
#     df = pd.Series(vals, days)
#
#     return df

def predict(model):
    # model = model_info[0].get('model')
    # model_hist = model_info[0].get('history')
    # model.save('model4')

    inputs = list(model.layers[0].input_shape)[1] - 1
    pred_inputs, pred_targets = dat.generateDataset(x_test, inputs)

    p = model.predict(pred_inputs)

    return p

def standardize():
    scaler = MinMaxScaler()

    scaler.fit(prediction)
    prediction_standardized = scaler.inverse_transform(prediction)

    h_reshaped = x_test['Close'].values

    values = h_reshaped.reshape((len(h_reshaped), 1))
    scaler.fit(values)
    history_standardized = scaler.inverse_transform(values)

    return prediction_standardized, history_standardized

def display(h, prediction):
    h = [x for x in h]
    plt.plot(h, label='History')
    plt.plot(prediction, label='Prediction future')
    plt.legend()
    plt.show()

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

dat = FormData('MSFT')
history = dat.getHistory()
h_normalized = dat.normalize(history)

print(h_normalized)

x_train = h_normalized[:-100]
x_test = h_normalized[-100:]


time_steps = 10

# reshape to [samples, time_steps, n_features]

# 'X_train, y_train = create_dataset(x_train, train.cnt, time_steps)
# X_test, y_test = create_dataset(test, test.cnt, time_steps)
#
# print(X_train.shape, y_train.shape)'

# NN = NeuralNet()
# best_model = NN.selectInputs(x_train, 2, 2, 50)

best_model = models.load_model('model4')
prediction = predict(best_model)
# h, predicted = standardize()
display(x_test['Close'], prediction)

# Note: batch size 10 with around 50 epochs best loss = 3.5