import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np



df = pd.read_csv("FuelConsumptionCo2.csv", encoding='ISO-8859-1')
# Take a look at the dataset
df.head()

# Summarize data
df.describe()

cdf = df[['Engine size (L)','Cylinders','Combined (mpg)','CO2 emissions (g/km)']]
cdf.head(9)

# Plot each of these features
viz = cdf[['Cylinders','Engine size (L)','CO2 emissions (g/km)','Combined (mpg)']]
viz.hist()
plt.show()

# Now, lets plot each of these features against the Emission, to see how linear their relationship is:
plt.scatter(cdf.Combined, cdf.emissions,  color='blue')
plt.xlabel("Combined (mpg)")
plt.ylabel("Emissions")
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

##PRACTICE

# Plot Cylinder versus the Emission, to see how linear their relationship is
plt.scatter(cdf.CYLINDER, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Cylinder")
plt.ylabel("Emission")
plt.show()


## Creating train and test dataset
# Train/Test Split

## Lets split our dataset: 80% of the entire dataset will be used for training and 20% for testing. We create a mask to select random rows using np.random.rand()


msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]



## Train Data Distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


##Modeling

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


#Plot outputs

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")


##Evaluation

from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )



##Exercise

# Lets see what the evaluation metrics are if we trained a regression model using the FUELCONSUMPTION_COMB feature. 

# train_x = np.asanyarray(train[('FUELCONSUMPTION_COMB')])
# test_x = np.asanyarray(test[('FUELCONSUMPTION_COMB')])
train_x = train[["FUELCONSUMPTION_COMB"]]

test_x = test[["FUELCONSUMPTION_COMB"]]

## Now train a linear regression model using train_x you created and the train_y created previously


regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)

## Find the predictions 
predictions = regr.predict(test_x)


# Finally, use the predictions and the test_y data and find the Mean absolute Error value using the np.absolute and np.mean function like done previously. 

print("Mean absolute error: %.2f" % np.mean(np.absolute(predictions - test_y)))

# MAE = np.mean(np.absolute(predictions - test_y))