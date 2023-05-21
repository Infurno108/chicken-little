import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras as kr
timeSteps = 1000
forSteps = abs(1000 - 2601)
blocks = 125

data = []

with open('appleTenYears.csv', newline='') as csvfile:
    read = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in read: 
        data.append(row)

def showGraph(prices):
    plt.figure()
    width = 5
    width2 = 1
    up = prices[prices.close>=prices.open]
    down = prices[prices.close<prices.open]
    col1 = 'green'
    col2 = 'red'
    #plot up prices
    plt.bar(up.index,up.close-up.open,width,bottom=up.open,color=col1)
    plt.bar(up.index,up.high-up.close,width2,bottom=up.close,color=col1)
    plt.bar(up.index, up.low-up.open, width2, bottom=up.open, color=col1)

    plt.bar(down.index,down.close-down.open,width,bottom=down.open,color=col2)
    plt.bar(down.index,down.high-down.open,width2,bottom=down.open,color=col2)
    plt.bar(down.index, down.low-down.close, width2, bottom=down.close, color=col2)

    plt.xticks(rotation=45, ha='right')

    plt.show()

#def trainerBuilder(prices, column):
#    for i in 

open = []
close = []
high = []
low = []
for i in range(1, len(data) - forSteps):
    open.append(float(data[i][1]))
    high.append(float(data[i][2]))
    low.append(float(data[i][3]))
    close.append(float(data[i][4]))
prices = pd.DataFrame({
    'open': open,
    'close': close,
    'high': high,
    'low': low},
    index = pd.date_range(str(data[1][0]), periods=timeSteps, freq="d")
)


#showGraph(prices)

inputLayer = kr.Input(shape=(4,))

openModel = kr.models.Sequential()
openModel.add(kr.layers.LSTM(blocks, input_dim=6))
openModel.add(kr.layers.Dense(1))
openModel.build()
openModel.summary()

#chicken.add(kr.layers.Dense(30, input_dim=))



