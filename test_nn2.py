from NeuralNetwork import NeuralNetwork
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets, metrics, model_selection, preprocessing

iris = datasets.load_iris()

X = np.array(iris.data[:100])
Y = np.array(iris.target[:100])

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = X_train.T
Y_train = Y_train.reshape(1,len(Y_train))
X_test = X_test.T
Y_test = Y_test.reshape(1,len(Y_test))

print(X_train.shape)
print(X_test.shape)


def binary_dnn_model_results(model, X_train, Y_train, X_test, Y_test):
    
    print("DNN Model Results")
    
    # Learning Phase
    print("GRADIENT DESCENT CHECK")
    plt.plot(model.cost_during_training)
    plt.title("Cross Entropy Cost Value During Training")
    plt.xlabel("Iteration")
    plt.ylabel("Cross Entropy Cost")
    plt.show()
    
    # Accuracy and ROC on Train data
    print("\nTRAINING METRICS")
    train_preds = model.predict(X_train)
    train_pred_labels = train_preds > 0.5
    train_accuracy = np.sum(train_pred_labels == Y_train) / Y_train.shape[1]
    print(f"Train Accuracy: {train_accuracy}")
    train_roc_auc_score = metrics.roc_auc_score(Y_train.T, train_preds.T)
    print(f"Train ROC AUC SCORE: {train_roc_auc_score}")

    # Accuracy and ROC on Test data
    print("\nTEST METRICS")
    test_preds = model.predict(X_test)
    test_pred_labels = test_preds > 0.5
    test_accuracy = np.sum(test_pred_labels == Y_test) / Y_test.shape[1]
    print(f"Test Accuracy: {test_accuracy}")
    test_roc_auc_score = metrics.roc_auc_score(Y_test.T, test_preds.T)
    print(f"Test ROC AUC SCORE: {test_roc_auc_score}")    


model = NeuralNetwork()
#print number of units in input layer
print(X_train.shape[0])

model.fit(X_train=X_train, Y_train=Y_train, layer_dims=[5,4,1], epoch=1_000)
binary_dnn_model_results(model, X_train, Y_train, X_test, Y_test)





