"""
1. Loading The Data
Dataset I’m going to use includes charges from patients. 
I highly recommend that you download this dataset and write your codes with me. 
You can access my codes from here. First, let’s import the dataset. 
I’m going to load the dataset using Pandas. Let me first import Pandas.
"""

import pandas as pd

"""
Pandas is an excellent library for data loading and data preprocessing. 
Now, let me load the dataset with the read_csv method.
"""

data = pd.read_csv("insurance.csv")
data.head()

"""
2. Understanding The Dataset
Understanding the data is very important before building a machine learning model. 
For example, let’s see the number of rows and columns of the dataset. 
I’m going to use the shape attribute to do this.
"""

data.shape
data.info()
data.isnull()
data.isnull().sum()

data.dtypes

"""
3. Data Preprocessing
Let’s convert object types to category types.
"""

data['sex'] = data['sex'].astype('category')
data['region'] = data['region'].astype('category')
data['smoker'] = data['smoker'].astype('category')

data.dtypes

"""
Now, let’s go ahead and take a look at the statistics of numeric variables with the describe method. 
If we use the transpose of the dataset, you can see the statistics better.
"""

data.describe().T

"""
Now, let’s look at the mean charges for smokers and non-smokers. 
To do this, let’s first group with the groupby method. 
I’m going to use the round method to see only two numbers after the comma.
"""

smoke_data = data.groupby("smoker").mean(numeric_only=True).round(2)
smoke_data

"""4. Data Visualization
You can understand the dataset better with data visualization. 
Now, let’s look at the relationships of numeric variables using the seaborn. 
First, let me import seaborn.
"""

import seaborn as sns

sns.set_style("whitegrid")

sns.pairplot(
    data[["age", "bmi", "charges", "smoker"]],
    hue = "smoker",
    height = 3,
    palette = "Set1")

"""
For example, when the age variable increases, both smokers and non-smokers pay more. 
Now, let’s look at the correlation between the variables.
"""

sns.heatmap(data[["age", "bmi", "children", "charges"]].corr(), annot=True)

"""
Notice that there is a relationship between charges and the other variables.
"""

"""
One-Hot Encoding
Now, I’m going to do a one-hot encoding of the categorical variables in the dataset. 
This is very easy to do with Pandas. You can automatically convert categorical data into one-hot encoding using the get_dummies method in Pandas. 
Let’s convert categorical data to one-hot encoding.
"""

data = pd.get_dummies(data)

"""
Thus, only categorical data were converted to one-hot encoding. Now let’s look at the columns of the dataset.
"""

data.columns

"""As you can see, new columns have been created for each subcategory. Using Pandas was very easy. Thanks, Pandas! 
Thus, the dataset is ready to build the model. Let’s go ahead and build a regression model.
"""

"""
5. Building a Regression Model
When building a model, you should start with the simplest model. 
If you don’t get good accuracy, you can try more complex models. 
I’ll build a linear regression model because the output variable charges is numeric type.
"""

"""
Before building a machine learning model, we need to determine the input and output variables. 
The input variables are features. In statistics, these are called independent variables. 
The output variable is the target variable. In statistics, this variable is called the dependent variable. 
Let’s assign the target variable charges to variable y.
"""

y = data["charges"]

"""
If we drop the target variable, the remainders are the features.
"""

x = data.drop("charges", axis = 1)

"""
Before the model is built, the dataset is split into training and testing. 
The model is built with the training data, and the model is evaluated with the test data. 
You can use the train_test_split method in scikit-learn to split the dataset into training and testing. 
With this method, you can easily split the dataset. First, let’s import this method.
Let’s split the dataset into 80 percent training and 20 percent testing.
"""

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(
    x,y, 
    train_size = 0.80, 
    random_state = 1)

from sklearn.linear_model import LinearRegression



lr = LinearRegression()
lr.fit(x_train,y_train)

"""
6. Model Evaluation
Let’s take a look at the performance of the model. To do this, I’m going to use the coefficient of determination. 
The closer this value is to 1, the better the model. First, let’s take a look at the score of the model on the test data.
"""

round(lr.score(x_test, y_test),3)

"""
Now let’s take a look at another metric, mean squared error, to evaluate the model. 
For this, let’s first predict the test data with the predict method.
"""

y_pred = lr.predict(x_test)

from sklearn.metrics import mean_squared_error
import math

"""Let’s take a look at the square root of the mean squares error."""

math.sqrt(mean_squared_error(y_test, y_pred))

"""7. Model Prediction
Now, I’m going to predict the first row as an example. First, let’s select the first row of the test data.
"""

data_new = x_train[:1]

"""
Let me predict the data with our model.
"""

lr.predict(data_new)
"""
Let’s take a look at the real value.
"""

y_train[:1]

"""
As you can see, our model predicted close to the real value.
"""

"""
MLFlow is used for MLOps.  It lets us run repeatable experiements, as well as saving the model and loading it.
"""
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skl2onnx import to_onnx

import numpy as np
import uuid

"""
As we will be rendering some large tables, lets remove the page wrapping
"""
pd.set_option('display.width', 1000)

"""
Define a helper meathod for calculating a series of interesting metrics from the actual and predicted values.
This can be used on single values or whole lists of values.
"""
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

"""
Step1:  Define the experiment, this will provide a wrapper for the implementation of several runs.
For example each run, may adjust one or more input parameters.
mlflow.log_param() can be used to log input variables and output variables.
"""
mlflow.set_experiment(f'Linear Regression Initial Run (Onnx)')

"""
Step2:  Runs help us to record seperate model training and prediction steps
You can loop within an experiement, generating mulitple runs.
When evaluating each run you can store the metrics with 
mlflow.log_metric()
These metrics can then be compared across runs.
"""

  
"""
Step3:  
Within the run, train and test the model instance.
"""
  fit_intercept=True
  positive=False
  mlflow.log_param("fit_intercept", fit_intercept) 
  mlflow.log_param("positive", positive)
  lr = LinearRegression(fit_intercept=fit_intercept, positive=positive)
  lr.fit(x_train,y_train)
  """
Step4:  
Generate predictions from the model, and create the associated metrics.
Log the metrics and the model
"""
  y_pred = lr.predict(x_test)
  signature = infer_signature(y_test, y_pred)
  
  (rmse, mae, r2) = eval_metrics(y_test, y_pred)
  mlflow.log_metric("rmse", rmse)
  mlflow.log_metric("r2", r2)
  mlflow.log_metric("mae", mae)
  

  olr = to_onnx(lr, x_test[:1])
  model_meta = mlflow.sklearn.log_model(olr, "ml_lr101", registered_model_name="ml_lr101", signature=signature)
  print(model_meta)
  print(model_meta.artifact_path)
  
  mlflow.end_run()
  
with mlflow.start_run():
  
  fit_intercept=False
  positive=True
  mlflow.log_param("fit_intercept", fit_intercept)
  mlflow.log_param("positive", positive) 
  lr = LinearRegression(fit_intercept=fit_intercept, positive=positive)
  lr.fit(x_train,y_train)
  
  y_pred = lr.predict(x_test)
  signature = infer_signature(y_test, y_pred)  

  (rmse, mae, r2) = eval_metrics(y_test, y_pred)
  mlflow.log_metric("rmse", rmse)
  mlflow.log_metric("r2", r2)
  mlflow.log_metric("mae", mae)  

  model_meta = mlflow.sklearn.log_model(lr, "ml_lr101", registered_model_name="ml_lr101", signature=signature)
  print(model_meta)
  
  mlflow.end_run()  

  
mlflow.set_experiment(f'Linear Regression Model Loaded')
  
with mlflow.start_run():
  
  mlflow_model = mlflow.sklearn.load_model('/home/cdsw/.experiments/jo0w-98ty-zbh8-lupa/q1ye-inik-8wyh-lrlw/artifacts/ml_lr101')
  prediction = mlflow_model.predict(x_test[:1])
  print(x_test[:1])
  print(prediction)
  mlflow.log_param("input", x_test[:1])
  mlflow.log_param("prediction", prediction)
  
  mlflow.end_run()




