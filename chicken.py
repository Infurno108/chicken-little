import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras as kr
timeSteps = 1871
forSteps = abs(timeSteps - 2601)
blocks = 270
maxSize = 10000
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

def trainerBuilder(prices):#we need an array of: [days data, next days column data]
    global maxSize 
    returnArray = []
    for i in range(1, len(prices)):
        returnArray.append([prices[i][0]/maxSize,prices[i][1]/maxSize, prices[i][2]/maxSize, prices[i][3]/maxSize,prices[i][4]/maxSize])
    return returnArray

open = []
close = []
high = []
low = []
adj = []
volume = []
for i in range(1, len(data) - forSteps):
    open.append(float(data[i][1]))
    high.append(float(data[i][2]))
    low.append(float(data[i][3]))
    close.append(float(data[i][4]))
    adj.append(float(data[i][5]))
    volume.append(float(data[i][6]))

prices = pd.DataFrame({
    'open': open,
    'close': close,
    'high': high,
    'low': low,
    'adj': adj},
    index = pd.date_range(str(data[1][0]), periods=timeSteps, freq="d")
)

#showGraph(prices)

inputs = trainerBuilder(prices.values)
outputs = inputs[1:]
inputs = inputs[:(timeSteps - 2)]

inputs = np.array(inputs)
outputs = np.array(outputs)

def trainAndBuild():
    chicken = kr.models.Sequential()
    chicken.add(kr.layers.Dense(blocks, input_shape=(5, 1)))
    chicken.add(kr.layers.LSTM(blocks, return_sequences=True))
    chicken.add(kr.layers.LSTM(blocks, return_sequences=True))
    chicken.add(kr.layers.LSTM(blocks, return_sequences=True))
    chicken.add(kr.layers.Dense(1))
    chicken.compile(
        loss='mean_squared_error',
        optimizer="adam",
    )	
    chicken.summary()
    chicken.fit(inputs, outputs, epochs=200, batch_size=32, verbose=2)
    chicken.save("chicken.h5")

def runTests(timeWacky):
    difference = 0
    totalDifference = 0
    start = 0
    chicken = kr.models.load_model('chicken.h5')
    for i in range(0, timeWacky):
        prediction = chicken.predict(inputs[i].reshape(1,5))
        prediction = prediction*maxSize
        real = outputs[i + start]*maxSize
        print("prediction: \n", prediction[0], "\nactual: ", real)
        predictScore = sum(prediction[0])
        realScore = sum(real)
        print("\nPredict Score: ", predictScore[0], "\nReal Score: ", realScore)
        difference = abs(predictScore - realScore)
        print("\nDifference: ", difference)
        totalDifference = totalDifference + difference
    print("\nAverage Difference: ", (totalDifference/timeWacky))

chicken = kr.models.load_model('chicken.h5')    
test = chicken.predict(inputs[0].reshape(1,5))
print("\n Input: ", inputs[0] * maxSize)
print("\n Prediction: ", test * maxSize)
print("\n Expected: ", outputs[0] * maxSize)
print("\n Difference: ",)



#runTests(timeSteps - 2)
#trainAndBuild()

#chicken.add(kr.layers.Dense(30, input_dim=))


#268: 2.488612
#
