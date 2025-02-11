import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


# Creating dataframe from csv file
df = pd.read_csv('weight-height.csv')

# Preparing the data
x = df[['Weight']]
y = df[['Height']]


xMean = np.mean(x)
yMean = np.mean(y)

# Creating and training the model with training data
model = linear_model.LinearRegression()
model.fit(x,y)

# Predicting weights using model.predict
yhat = model.predict(x)

# Plotting points and generating regression line
plt.scatter(x, y, label="Actual Data", alpha=.5)
plt.scatter(xMean, yMean, color='red')
plt.plot(x, yhat, color="blue")
plt.title("Scatter plot of height-weight")
plt.ylabel("Weight")
plt.xlabel("Height")
plt.show()