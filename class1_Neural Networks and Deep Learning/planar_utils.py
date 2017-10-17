# coding:utf-8
import numpy as np
import random


def plot_decision_boundary():
	pass


def sigmoid(z):
	s = 1 / (1 + np.exp(-z))
	return s


def tanh(z):
	'''0均值，【-1,1】'''
	return 2 * sigmoid(2 * z) - 1


def load_planar_dataset():
	dim = 400
	X = []
	for i in range(dim * 2):
		X.append(random.uniform(-4, 4))
	X = np.array(X).reshape(dim, 2).T
	Y = np.random.randint(0, 2, (dim,)).T
	return X, Y


def plot_decision_boundry():
	pass


def load_extra_datasets():
	pass
