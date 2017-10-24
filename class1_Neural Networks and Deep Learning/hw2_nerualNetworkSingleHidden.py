# coding:utf-8
import random
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from testCases import *
from planar_utils import plot_decision_boundary, sigmoid, tanh, load_planar_dataset, load_extra_datasets


def layer_size(X, Y):
    '''
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    '''
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]


def initialize_parameters(n_x, n_h, n_y):
    '''
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    '''
    np.random.seed(2)  # we set up a seed so that your output matches ours although the initialization is random.
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters


def forward_propagation(X, parameters):
    '''
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" , "A2"
    '''
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    Z1 = np.dot(W1, X) + b1
    A1 = tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


def compute_cost(A2, Y, parameters):
    '''
    Computes the cross-entropy cost given in equation (13);表达预测值与真实值的差异时，交叉熵比平方差效果好
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    Returns:
    cost -- cross-entropy cost given equation (13)
    notice: mutiply "数量乘" dot"矢量乘"；当为数组是"*"代表数量乘；当为matric时"*"代表矢量乘
    '''
    m = Y.shape[1]
    logprobs = np.multiply(Y, np.log(A2)) + np.multiply(1 - Y, np.log(1 - A2))
    cost = (-1 / m) * np.sum(logprobs)
    # cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) 因为Y和A虽然是矩阵matrix，但因为是一维的，所以也可为数组array
    cost = np.squeeze(cost)
    assert (isinstance(cost, float))
    return cost


def backward_propagation(parameters, cache, X, Y):
    '''
    rguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    g'(z)=tanh'(z)=(1-z*z)
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    '''
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    '''
    Updates parameters using the gradient descent update rule given above

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients

    Returns:
    parameters -- python dictionary containing your updated parameters
    '''
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - dW1 * learning_rate
    b1 = b1 - db1 * learning_rate
    W2 = W2 - dW2 * learning_rate
    b2 = b2 - db2 * learning_rate

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    '''
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    '''
    np.random.seed(3)
    n_x = layer_size(X, Y)[0]
    n_y = layer_size(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b1"]

    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)
        if print_cost and i % 1000 == 0:
            print ("cost after iteration %i:%f") % (i, cost)
    return parameters


def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    return predictions


if __name__ == "__main__":
    np.random.seed(1)  # set a seed so that the results are consistent
    '''
    1.Visualize the data;looks like a “flower” with some red (label y=0) and some blue (y=1) points. Your goal is to build a model to fit this data.
    - a numpy-array (matrix) X that contains your features (x1, x2)
    - a numpy-array (vector) Y that contains your labels (red:0, blue:1).
    '''
    X, Y = load_planar_dataset()
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    # plt.show()
    shape_X = X.shape
    shape_Y = Y.shape
    m = X.shape[1]
    print ('The shape of X is: ' + str(shape_X))
    print ('The shape of Y is: ' + str(shape_Y))
    print ('I have m = %d training examples!' % (m))
    '''
    2.Plot the decision boundary for logistic regression,Accuracy:53%,so Logistic regression did not work well on the "flower dataset".
    '''
    # Plot the decision boundary for logistic regression
    plot_decision_boundary(lambda x: clf.predict(x), X, Y)
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X.T, Y.T)
    plt.title("Logistics Regression")
    LR_predictions = clf.predict(X.T)
    percentage = float((np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100)
    print "Accuracy of logistic regression:%d" % percentage + "% (percentage of correctly labelled datapoints)"

    '''
    3.build my model to classifier:
        1.the activate function first hidden layer: tanh
        2.the activate function of second-output hidden layer : ReLU
    methodology:
        1. Define the neural network structure ( # of input units,  # of hidden units, etc).
        2. Initialize the model's parameters
        3. Loop:
            - Implement forward propagation
            - Compute loss
            - Implement backward propagation to get the gradients
            - Update parameters (gradient descent)
    '''
    # parameters,cache,X_assess,Y_assess = backward_propagation()
    # Tuning hidden layer size
    plt.figure(figsize=(16, 32))
    hidden_layer_size = [1, 2, 3, 4, 5, 20, 50]
    for i, n_h in enumerate(hidden_layer_size):
        plt.subplot(5, 2, i + 1)
        plt.title("Hidden layer of size %d" % n_h)
        parameters = nn_model(X, Y, n_h, num_iterations=5000)
        plot_decision_boundary(lambda x: predict(parameters, X.T), X, Y)
        predictions = predict(parameters, X)
        accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions)) / float(Y.size) * 100)
        print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))

    # Performance on other datasets
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
    datasets = {"noisy_circles": noisy_circles,
                "noisy_moons": noisy_moons,
                "blobs": blobs,
                "gaussian_quantiles": gaussian_quantiles}
    dataset = "noisy_moons"
    X, Y = datasets[dataset]
    X, Y = X.T, Y.reshape(1, Y.shape[0])
    if dataset == "blobs":
        Y = Y % 2
    # Visualize the data
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
