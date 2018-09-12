# Machine Learning using Python

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
<b>MultivariateLinearRegressor.py</b> has a class MultivariateLinearRegressor implementing the Linear Regression Algorithm with a predict method of for a set of features (i.e. multiple variables) rather than a single variable.The data used for this one has multiple columns or features e.g. sqft area of houses, number of bedrooms, age of house in years and a target column for the price of house (i.e. Y). The sample data file is reused from Prof Andrew Ng's machine learning lessions on Coursera. The initial code was influences from Girish Kuniyal's implementation on https://github.com/girishkuniyal/Predict-housing-prices-in-Portland.<br>
  There are two alternative methods (e.g. gradient descent and normal equation) used for minimizing the cost and adjusting the value of the model (i.e. theta).<br><p>
  
<b>BinaryClassifier.py</b> has a class BinaryClassifier which uses Logistic regression with a predict() method to classify a set of house data having four features i.e. Sqft Area, Number of bedrooms, Age in years and Price. The houses are categorized in two classes i.e. 0 for Standard ones and 1 for Premium categories.<br> The binary classifier also uses a validate() method to compare the predicted values with the training data to check the accuracy of the model after training. The classifier also saves the model after training (in a specified file in the 'model' folder) and loads the same before validation or prediction. It also save the output of validation and prediction in the 'output' folder.<br><p>
<b>NeuralNetwork.py</b> has a mainly 3 classes - Neuron, NeuralLayer and NeuralNetowrk and few associated classes called ActivationFunction and LossFunction (in Neurofunctions.py file) which get trained using the back propagation method. A Neuron has weights and biases, it produces an output using a selected activation function (from the ActivationFunction class in NeuroFunctions.py) and propagates to the neurons in the next layer. A NeuralLayer is a collection of neurons. During the forward and backward propagation the NeuralLayer feeds the inputs or errors to all neurons in it. Finally a NeuralNetwork is a collection of NeuralLayers created based on the topology specified while instanciating the network.<br><p>
  
