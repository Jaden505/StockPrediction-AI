import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import datetime
from pprint import pprint
from tabulate import tabulate
import pandas as pd
from sklearn.preprocessing import scale
from torchvision import transforms

class Stocks:
    def __init__(self, companies):
        self.companies = companies
        self.sectors = {}
        self.history_range = 1

        self.main()

    def main(self):
        for ind, company in enumerate(companies):

            self.createXtrain(ind)
            self.getStocks(company)

        for sec in self.sectors.values():
            print(len(self.date_list), len(sec))

            PredictSector(self.date_list, sec, self.max_stock)

    def createXtrain(self, ind):
        base = datetime.datetime.today()
        self.date_list = [[int((base - datetime.timedelta(days=((self.history_range * 366) - x))).strftime('%Y%m%d'))]
                     for x in range(0, self.history_range * 367)]

    def getStocks(self, company):
        c = yf.Ticker(company)
        
        stock_info = c.history(period=f"{self.history_range}y")['Close']
        self.max_stock = max(stock_info)
        stock_info = [[s] for s in stock_info]
        sector = c.info['sector']

        print(c.history(period=f"{self.history_range}y"))

        self.date_list = [d for d in self.date_list if str(d[0]) in c.history(period=f"{self.history_range}y")['Close']]

        print(self.date_list)

        if sector in self.sectors:
            self.sectors[sector] += stock_info
        else:
            self.sectors[sector] = stock_info

class PredictSector:
    def __init__(self, x_train, y_train, max_stock):
        # X contains dates and companies, while Y contains the stocks
        self.x_train = x_train
        self.y_train = y_train
        self.max_stock = max_stock
        self.x_predicted = torch.Tensor([20191009])

        self.main()

    def main(self):
        self.format()
        self.normalize()

        print(len(self.y_train), len(self.x_train))
        print(self.x_train.size(), self.y_train.size())

        NN = Neural_Network()
        for i in range(1000):  # trains the NN 1,000 times
            print("#" + str(i) + " Loss: " + str(torch.mean((self.y_train - NN(self.x_train)) ** 2).detach().item()))  # mean sum squared loss
            NN.train_dat(self.x_train, self.y_train)
        NN.saveWeights(NN)
        NN.predict(self.x_predicted)

    def format(self):
        self.x_train = torch.tensor((self.x_train), dtype=torch.float) # dates x 2
        self.y_train = torch.tensor((self.y_train), dtype=torch.float) # dates x 1

    def normalize(self):
        # mean = torch.mean(self.x_train)
        # std = torch.std(self.x_train)
        # min = torch.min(self.x_train)
        # max = torch.max(self.x_train)
        #
        # self.x_train = (self.x_train - min) / (max - min)
        # self.y_train = self.y_train / self.max_stock

        X_max, _ = torch.max(self.x_train, 0)
        xPredicted_max, _ = torch.max(self.x_predicted, 0)

        self.x_train = torch.div(self.x_train, X_max)
        self.x_predicted = torch.div(self.x_predicted, xPredicted_max)
        self.y_train = self.y_train / self.max_stock

class Neural_Network(nn.Module):
    def __init__(self):
        super(Neural_Network, self).__init__()
        # parameters
        # TODO: parameters can be parameterized instead of declaring them here
        self.inputSize = 1
        self.outputSize = 1
        self.hiddenSize = 3

        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize)  # 2 X 3 tensor
        self.W2 = torch.randn(self.hiddenSize, self.outputSize)  # 3 X 1 tensor

    def forward(self, X):
        self.z = torch.matmul(X, self.W1)  # 3 X 3 ".dot" does not broadcast in PyTorch
        self.z2 = self.sigmoid(self.z)  # activation function
        self.z3 = torch.matmul(self.z2, self.W2)
        o = self.sigmoid(self.z3)  # final activation function
        return o

    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))

    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)

    def backward(self, X, y, o):
        self.o_error = y - o  # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o)  # derivative of sig to error
        self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.W1 += torch.matmul(torch.t(X), self.z2_delta)
        self.W2 += torch.matmul(torch.t(self.z2), self.o_delta)

    def train_dat(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o)

    def saveWeights(self, model):
        # we will use the PyTorch internal storage functions
        torch.save(model, "NN")
        # you can reload model with all the weights and so forth with:
        # torch.load("NN")

    def predict(self, xPredicted):
        print("Input: " + str(xPredicted))
        print("Output: " + str(self.forward(xPredicted)))

companies = ['MSFT']

Stocks(companies)
