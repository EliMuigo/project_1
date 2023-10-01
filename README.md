#**Week 1 project: Churn Prediction for Sprint**
 Imagine you're working with Sprint, one of the biggest telecom companies in the USA. They're really keen on figuring out how many customers might decide to leave them in the coming months. Luckily, they've got a bunch of past data about when customers have left before, as well as info about who these customers are, what they've bought, and other things like that.
   So, if you were in charge of predicting customer churn, how would you go about using machine learning to make a good guess about which customers might leave? What steps would you take to create a machine learning model that can predict if someone's going to leave or not?

*As a data scientist you're required to evaluate the churn dataset and predict the future trends of customers especially the trend at which the organisation is losing customers and the rate at which to reduce it.*


# 1:Data Collection:

I collected the dataset from kaggle notebook, loaded the dataset into the notebook.

---
<pre>
'''python
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import plotly.express as px

import plotly.io as pio

pio.templates

import seaborn as sns

from sklearn.linear_model import LinearRegression



data_load=pd.read_csv('churn-bigml-20.csv')

data_load
'''
</pre>

# 2.Data Preprocessing:
 After loading the dataset I processed the data to know if there were any nulls in the dataset. I then proceeded to remove the nulls.

 <pre>
 1.data_load.isnull().sum()

 2.data_load.dropna(inplace=True)

 </pre>

# 3.Data Exploration and Training
I split my dataset Split the data into training and testing sets for model evaluation.This is to enusre I achieve a high precision accuracy when working with my dataset.

***I used the following line of code***

<pre>
#split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
</pre>

# 4. Model Selection and testing

I used the linear regression model since it uses the mathematical relationship between independent variable and dependent variables to show the customer's characteristics to estimate the probability of a customer churning out.

We calculate the Mean Squared Error (MSE) and R-squared (R2) as performance metrics to assess the model's accuracy.

***my code***

<pre>
'''python

model=LinearRegression
#Train the model on the training data
model.fit(X,y)

#Make predictions on the test data
y_pred=model.predict(X_test)

#Calculate the perfomance metrics
mse=mean_squared_error(y,y_pred)

r2=r2_score(y,y_pred)

print("Mean Squared Error:",mse)

print("R-squared:"r2)
'''
</pre>

Finally, we visualize the data points and the linear regression line using Matplotlib.

<pre>
plt.scatter(X_test, y_test, label='Test Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Linear Regression')
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable')
plt.legend()
plt.show()
</pre>