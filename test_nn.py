from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from NeuralNetwork import NeuralNetwork
from sklearn.datasets import load_digits, load_iris, load_wine
from sklearn.model_selection import train_test_split
import numpy as np
from layer import  Dense
from activation import ReLU, Sigmoid
from optimizer import SGD, Adam, RMSprop
from loss import BinaryCrossEntropy


# Load the datasets
digits = load_digits()
X_digits, y_digits = digits.data, digits.target

iris = load_iris()
X_iris, y_iris = iris.data, iris.target

wine = load_wine()
X_wine, y_wine = wine.data, wine.target

# Split the datasets into training and testing sets
X_digits_train, X_digits_test, y_digits_train, y_digits_test = train_test_split(X_digits, y_digits, test_size=0.2, random_state=42)
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)
X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(X_wine, y_wine, test_size=0.2, random_state=42)

# Train and test scikit-learn's MLPClassifier model for digits dataset
sklearn_nn_digits = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=1000, random_state=42)
sklearn_nn_digits.fit(X_digits_train, y_digits_train)
sklearn_nn_predictions_digits = sklearn_nn_digits.predict(X_digits_test)
sklearn_nn_accuracy_digits = accuracy_score(y_digits_test, sklearn_nn_predictions_digits)
print("Scikit-learn MLPClassifier Accuracy (Digits):", sklearn_nn_accuracy_digits)

# Train and test scikit-learn's MLPClassifier model for iris dataset
sklearn_nn_iris = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=1000, random_state=42)
sklearn_nn_iris.fit(X_iris_train, y_iris_train)
sklearn_nn_predictions_iris = sklearn_nn_iris.predict(X_iris_test)
sklearn_nn_accuracy_iris = accuracy_score(y_iris_test, sklearn_nn_predictions_iris)
print("Scikit-learn MLPClassifier Accuracy (Iris):", sklearn_nn_accuracy_iris)

# Train and test scikit-learn's MLPClassifier model for wine dataset
sklearn_nn_wine = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=1000, random_state=42)
sklearn_nn_wine.fit(X_wine_train, y_wine_train)
sklearn_nn_predictions_wine = sklearn_nn_wine.predict(X_wine_test)
sklearn_nn_accuracy_wine = accuracy_score(y_wine_test, sklearn_nn_predictions_wine)
print("Scikit-learn MLPClassifier Accuracy (Wine):", sklearn_nn_accuracy_wine)

your_nn_digits = NeuralNetwork()
X_digits_train = X_digits_train.T
X_digits_test = X_digits_test.T
y_digits_train = y_digits_train.reshape(1, len(y_digits_train))
y_digits_test = y_digits_test.reshape(1, len(y_digits_test))
your_nn_digits.fit(X_digits_train, y_digits_train, layer_dims=[X_digits_train.shape[0], 100, 10], epoch=1000, learning_rate=0.01, minibatch_size=64)
your_nn_predictions_digits = your_nn_digits.predict(X_digits_test)
your_nn_accuracy_digits = accuracy_score(y_digits_test, your_nn_predictions_digits)
print("Your NeuralNetwork Accuracy (Digits):", your_nn_accuracy_digits)

# Train and test your NeuralNetwork model for iris dataset
#your_nn_iris = NeuralNetwork(sizes=[X_iris_train.shape[1], 10,10, 3])
#your_nn_iris.train(X_iris_train, y_iris_train,X_iris_test,y_iris_test)
#your_nn_predictions_iris = your_nn_iris.predict(X_iris_test)
#your_nn_accuracy_iris = accuracy_score(y_iris_test, your_nn_predictions_iris)
#print("Your NeuralNetwork Accuracy (Iris):", your_nn_accuracy_iris)

# Train and test your NeuralNetwork model for wine dataset
#your_nn_wine = NeuralNetwork(sizes=[X_wine_train.shape[1], 100, 3], activation='relu')
#your_nn_wine.fit(X_wine_train, y_wine_train)
#your_nn_predictions_wine = your_nn_wine.predict(X_wine_test)
#your_nn_accuracy_wine = accuracy_score(y_wine_test, your_nn_predictions_wine)
#print("Your NeuralNetwork Accuracy (Wine):", your_nn_accuracy_wine)