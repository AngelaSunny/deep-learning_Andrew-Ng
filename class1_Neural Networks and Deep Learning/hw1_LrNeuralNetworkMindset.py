# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from urllib import urlretrieve

import os
import gzip
import pickle
import matplotlib.cm as cm


def load_dataset():
	'''
	load source data,generate train/dev/test sets
	'''
	url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
	filename = 'mnist.pkl.gz'
	if not os.path.exists(filename):
		print "Downloading MNIST dataset..."
	urlretrieve(url, filename)
	with gzip.open(filename, 'rb') as f:
		data = pickle.load(f)
		X_train, y_train = data[0]
		X_val, y_val = data[1]
		X_test, y_test = data[2]
		X_train = X_train.reshape((-1, 1, 28, 28))
		X_val = X_val.reshape((-1, 1, 28, 28))
		X_test = X_test.reshape((-1, 1, 28, 28))
		y_train = y_train.astype(np.uint8)
		y_val = y_val.astype(np.uint8)
		y_test = y_test.astype(np.uint8)
	return X_train, y_train, X_val, y_val, X_test, y_test


X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
plt.imshow(X_train[0][0], cmap=cm.binary)

print "end get images,start build nerual network architecture"


def sigmoid(z):
	'''
	compute the sigmoid of z
	:param z:
	:return:
	'''
	s = 1 / (1 + np.exp(-z))
	return s


def initialize_with_zero(dim):
	'''
	this function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
	Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
	:param dim:
	:return:
	'''
	w = np.zeros(shape=(dim, 1))
	b = 0
	assert (w.shape == (dim, 1))
	assert (issubclass(b, float) or isinstance(b, int))
	return w, b


def propagate(w, b, X, Y):
	'''
	Implement the cost function and its gradient for the propagation explained above
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
	'''
	m = X.shape[1]
	A = sigmoid(np.dot(w.T, X) + b)
	cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
	dw = (1 / m) * np.dot(X, (A - Y).T)
	db = (1 / m) * np.sum(A - Y)
	assert (dw.shape == w.shape)
	assert (db.dtype == float)
	cost = np.squeeze(cost)
	grads = {"dw": dw, "db": db}
	return grads, cost


def optimize(w, b, X, Y, num_iterations, learinig_rate, print_cost=False):
	'''
	tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.'''
	costs = []
	for i in range(num_iterations):
		grads, cost = propagate(w, b, X, Y)
		dw = grads["dw"]
		db = grads["db"]
		w = w - learinig_rate * dw
		b = b - learinig_rate * db
		if i % 100 == 0:
			costs.append(cost)
		if print_cost and i % 100 == 0:
			print "cost after iteration %i:%f" % (i, cost)
	params = {"w": w, "b": b}
	grads = {"dw": dw, "db": db}
	return params, grads, costs


def predict(w, b, X):
	'''
	Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
	:param w:
	:param b:
	:param X:
	:return:
	'''
	m = X.shape[1]
	Y_prediction = np.zeros(1, m)
	w = w.reshape(X.shape[0], 1)
	A = sigmoid(np.dot(w.T, X) + b)
	for i in range(A.shape[1]):
		# Convert probabilities a[0,i] to actual predictions p[0,i]
		Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
	assert (Y_prediction.shape == [1, m])
	return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
	w, b = initialize_with_zero(X_train.shape[0])
	parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
	w = parameters["w"]
	b = parameters["b"]
	Y_prediction_test = predict(w, b, X_test)
	Y_prediction_train = predict(w, b, X_train)
	print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
	print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
	d = {"costs": costs,
	     "Y_prediction_test": Y_prediction_test,
	     "Y_prediction_train": Y_prediction_train,
	     "w": w,
	     "b": b,
	     "learning_rate": learning_rate,
	     "num_iterations": num_iterations}
	return d


if __name__ == "__main__":
	dim = 2
	w, b = initialize_with_zero(dim)
	print ("w = " + str(w))
	print ("b = " + str(b))
	print "**********************************************************************"
	'''
	1.获取训练样本，测试样本数据
	description:
	- a training set of m_train images labeled as cat (y=1) or non-cat (y=0)
	- a test set of m_test images labeled as cat or non-cat
	- each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB). Thus, each image is square (height = num_px) and (width = num_px).
	'''
	train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
	index = 25
	plt.show(train_set_x_orig[index])
	print "y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode(
		"utf-8") + "' picture."
	m_train = train_set_y.shape[1]
	m_test = test_set_y.shape[1]
	num_px = train_set_x_orig.shape[1]
	print "Number if training examples:m_train=" + str(m_train)
	print ("Number of testing examples: m_test = " + str(m_test))
	print ("Height/Width of each image: num_px = " + str(num_px))
	print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
	print ("train_set_x shape: " + str(train_set_x_orig.shape))
	print ("train_set_y shape: " + str(train_set_y.shape))
	print ("test_set_x shape: " + str(test_set_x_orig.shape))
	print ("test_set_y shape: " + str(test_set_y.shape))
	print "**********************************************************************"
	train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
	test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
	print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
	print ("train_set_y shape: " + str(train_set_y.shape))
	print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
	print ("test_set_y shape: " + str(test_set_y.shape))
	print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))
	train_set_x = train_set_x_flatten / 255
	test_set_x = test_set_x_flatten / 255
	'''
	 2.训练模型 General Architecture of the learning algorithm
	 steps:
	    - Initialize the parameters of the model
		- Learn the parameters for the model by minimizing the cost
		- Use the learned parameters to make predictions (on the test set)
	'''
	d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005,
	          print_cost=True)
	'''
	3.Analyse the results and conclude
	'''

	index = 5
	plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
	print ("y = " + str(test_set_y[0, index]) + ", you predicted that it is a \"" + classes[
		d["Y_prediction_test"][0, index]].decode("utf-8") + "\" picture.")
	# 4.代价函数变化曲线
	# Plot learning curve (with costs)
	costs = np.squeeze(d['costs'])
	plt.plot(costs)
	plt.ylabel('cost')
	plt.xlabel('iterations (per hundreds)')
	plt.title("Learning rate =" + str(d["learning_rate"]))
	plt.show()
	# 5.深入分析（不同学习率，对应的cost变化趋势图）
	learning_rates = [0.01, 0.001, 0.0001]
	models = {}
	for i in learning_rates:
		print ("learning rate is: " + str(i))
		models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=i,
		                       print_cost=False)
		print ('\n' + "-------------------------------------------------------" + '\n')

	for i in learning_rates:
		plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))
	plt.ylabel('cost')
	plt.xlabel('iterations')
	legend = plt.legend(loc='upper center', shadow=True)
	frame = legend.get_frame()
	frame.set_facecolor('0.90')
	plt.show()

	# 6.用自己的数据测试
	my_image = "my_image.jpg"  # change this to the name of your image file
	fname = "images/" + my_image
	image = np.array(ndimage.imread(fname, flatten=False))
	my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T
	my_predicted_image = predict(d["w"], d["b"], my_image)  # 直接传入参数预测结果，不用重复训练	plt.imshow(image)
	print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[
		int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")
