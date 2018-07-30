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
<b>MultivariateLinearRegressor.py</b> has a class MultivariateLinearRegressor implementing the Linear Regression Algorithm with a predict method of for a set of features (i.e. multiple variables) rather than a single variable.The data used for this one has 2 columns or features - square feet area of houses (feature column 1), number of bedrooms (feature column 2) and 1 target column for the price of house. The sample data file is reused from Prof Andrew Ng's machine learning lessions on Coursera. The code has referred to Girish Kuniyal's implementation 
on https://github.com/girishkuniyal/Predict-housing-prices-in-Portland.
  
