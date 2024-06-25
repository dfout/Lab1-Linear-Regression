import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

"Model year,Make,Model,Vehicle class,Engine size (L),Cylinders,Transmission,Fuel type,City (L/100 km),Highway (L/100 km),Combined (L/100 km),Combined (mpg),CO2 emissions (g/km),CO2 rating,Smog rating"

df = pd.read_csv("FuelConsumptionCo2.csv", encoding='ISO-8859-1')
# Take a look at the dataset
df.head()

# Summarize data
df.describe()

cdf = df[['EngineSize','Cylinders','CombinedMPG','CO2Emissions']]
cdf.head(9)

# Plot each of these features
viz = cdf[['Cylinders','EngineSize','CO2Emissions','CombinedMPG']]
viz.hist()
plt.show()

# Now, lets plot each of these features against the Emission, to see how linear their relationship is:
plt.scatter(cdf.CombinedMPG, cdf.CO2Emissions,  color='blue')
plt.xlabel("Combined (mpg)")
plt.ylabel("CO2 emissions (g/km)")
plt.show()

plt.scatter(cdf.EngineSize, cdf.CO2Emissions,  color='blue')
plt.xlabel("Engine Size (L)")
plt.ylabel("Co2 Emissions (g/km)")
plt.show()

##PRACTICE

# Plot Cylinder versus the Emission, to see how linear their relationship is
plt.scatter(cdf.Cylinders, cdf.CO2Emissions,  color='blue')
plt.xlabel("Cylinders")
plt.ylabel("C02 Emissions (g/km)")
plt.show()


## Creating train and test dataset
# Train/Test Split

## Lets split our dataset: 80% of the entire dataset will be used for training and 20% for testing. We create a mask to select random rows using np.random.rand()


msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]



## Train Data Distribution
plt.scatter(train.EngineSize, train.CO2Emissions,  color='blue')
plt.xlabel("Engine size (L)")
plt.ylabel("C02 Emissions (g/km)")
plt.show()


##Modeling

## Engine Size vs. C02 Emissisons

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['EngineSize']])
train_y = np.asanyarray(train[['CO2Emissions']])
regr.fit(train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


#Plot outputs

plt.scatter(train.EngineSize, train.CO2Emissions,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size (L)")
plt.ylabel("C02 Emissions (g/km)")


##Evaluation

from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['EngineSize']])
test_y = np.asanyarray(test[['CO2Emissions']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )



##Exercise

## Model instead..
### CombinedMPG vs. CO2 Emissions

# Lets see what the evaluation metrics are if we trained a regression model using the FUELCONSUMPTION_COMB feature. 

# train_x = np.asanyarray(train[('FUELCONSUMPTION_COMB')])
# test_x = np.asanyarray(test[('FUELCONSUMPTION_COMB')])
train_x = train[["CombinedMPG"]]

test_x = test[["CombinedMPG"]]

## Now train a linear regression model using train_x you created and the train_y created previously


regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)

## Find the predictions 
predictions = regr.predict(test_x)


# Finally, use the predictions and the test_y data and find the Mean absolute Error value using the np.absolute and np.mean function like done previously. 

print("Mean absolute error: %.2f" % np.mean(np.absolute(predictions - test_y)))

# MAE = np.mean(np.absolute(predictions - test_y))


##OUTPUT:
"""
Coefficients:  [[39.48502943]]
Intercept:  [137.09080016]
Mean absolute error: 29.21
Residual sum of squares (MSE): 1601.81
R2-score: 0.65
Mean absolute error: 18.10
"""


## We can see that the MAE is much worse when we trained with the Engine Size versus combined CombinedMPG



"""
IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: SPSS Modeler

Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at Watson Studio"""