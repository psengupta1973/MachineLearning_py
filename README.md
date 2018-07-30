# machine_learning_py

<h3> Linear Regression with single variable </h3>
<br>
<b>UniLinReg.py</b> reads data from a CSV file (test_score_vs_hour_studied.csv) with 2 columns in it - test scores and hours of study.
Loads them in x (filled with hours of study) and y (filled with test scores) data arrays.
Checks for initial error with checkError() method.
Executes train() method to train the model with the data and finally use predict method to predict a test score with 
a give value of hours of study.
The train method uses gradientDescent() method to adjust the training paramters i.e. theta
The training and error correction loop runs for 100 iterations to find a linear relation between the give data points.
<p><br>
<b>UnivariateLinearRegressor.py</b> has a class UnivariateLinearRegressor implementing the Linear Regression Algorithm with a predict method of for a set of values (i.e. array) rather than a single value.
<p><br>
<b>MultivariateLinearRegressor.py</b> has a class MultivariateLinearRegressor implementing the Linear Regression Algorithm with a predict method of for a set of features (i.e. multiple variables) rather than a single variable.The data used for this one has multiple columns or features e.g. sqft area of houses, number of bedrooms, age of house in years and a target column for the price of house (i.e. Y). The sample data file is reused from Prof Andrew Ng's machine learning lessions on Coursera. The initial code was influences from Girish Kuniyal's implementation on https://github.com/girishkuniyal/Predict-housing-prices-in-Portland.<b>
  There are two alternative methods (e.g. gradient descent and linear equation) used for minimizing the cost and adjusting the value of the model (i.e. theta).
  
