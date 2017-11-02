#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import numpy as np
from cnn import element_wise_op
from activators import ReluActivator, IdentityActivator


class RecurrentLayer(object):
	def __init__(self, input_width, state_width, activator, learning_rate):
		'''
		假设输入向量x的维度是m，输出向量s的维度是n，则矩阵U的维度是n*m，矩阵W的维度是n*n。
		'''
		self.input_width = input_width  # 输入向量x的维度(输入特征)m=3
		self.state_width = state_width  # 输出向量s的维度是n=2
		self.activator = activator
		self.learning_rate = learning_rate
		self.times = 0  # 当前时刻初始化为t0
		self.state_list = []  # 保存各个时刻的state(3,2)
		self.state_list.append(np.zeros((state_width, 1)))  # 初始化s0
		self.U = np.random.uniform(-1e-4, 1e-4, (state_width, input_width))  # 初始化U（2,3）
		self.W = np.random.uniform(-1e-4, 1e-4, (state_width, state_width))  # 初始化W（2,2）

	def forward(self, input_array):
		'''
		根据『式2』进行前向计算
		'''
		self.times += 1
		state = (np.dot(self.U, input_array) + np.dot(self.W, self.state_list[-1]))  # U*Xt+W*St-1
		element_wise_op(state, self.activator.forward)  # St = f(U*Xt+W*St-1)
		self.state_list.append(state)

	def backward(self, sensitivity_array, activator):
		'''
		实现BPTT算法
		'''
		self.calc_delta(sensitivity_array, activator)
		self.calc_gradient()

	def calc_delta(self, sensitivity_array, activator):
		self.delta_list = []  # 用来保存各个时刻的误差项,共m=3个
		for i in range(self.times):#2次
			self.delta_list.append(np.zeros((self.state_width, 1)))#delta_list=[(0,0),(0,0)]
		self.delta_list.append(sensitivity_array)#第3次delta_list=[(0,0),(0,0),sensitivity_array]
		# 迭代计算每个时刻k的误差项delta[k]：
		for k in range(self.times - 1, 0, -1):
			self.calc_delta_k(k, activator)

	def calc_delta_k(self, k, activator):
		'''
		根据k+1时刻的delta计算k时刻的delta(公式4)
		'''
		state = self.state_list[k + 1].copy()
		element_wise_op(self.state_list[k + 1], activator.backward)#激活函数求导
		self.delta_list[k] = np.dot(np.dot(self.delta_list[k + 1].T, self.W), np.diag(state[:, 0])).T#delta[l-1] = delta[l]*U*diag(state)

	def calc_gradient(self):
		self.gradient_list = []  # 保存各个时刻的权重梯度(3,2)
		for t in range(self.times + 1):
			self.gradient_list.append(np.zeros((self.state_width, self.state_width)))#gradient_list=[(0,0),(0,0),(0,0)]
		for t in range(self.times, 0, -1):
			self.calc_gradient_t(t)
		# 实际的梯度是各个时刻梯度之和
		self.gradient = reduce(lambda a, b: a + b, self.gradient_list, self.gradient_list[0])  # [0]被初始化为0且没有被修改过

	def calc_gradient_t(self, t):
		'''
		计算每个时刻t权重的梯度
		'''
		gradient = np.dot(self.delta_list[t], self.state_list[t - 1].T)  # 权重矩阵W在t时刻的梯度
		self.gradient_list[t] = gradient

	def reset_state(self):
		self.times = 0  # 当前时刻初始化为t0
		self.state_list = []  # 保存各个时刻的state
		self.state_list.append(np.zeros((self.state_width, 1)))  # 初始化s0

	def update(self):
		'''
		按照梯度下降，更新权重
		'''
		self.W -= self.learning_rate * self.gradient


def data_set():
	x = [np.array([[1], [2], [3]]),
	     np.array([[2], [3], [4]])]
	d = np.array([[1], [2]])
	return x, d


def gradient_check():
	'''
	梯度检查
	'''
	# 设计一个误差函数，取所有节点输出项之和
	error_function = lambda o: o.sum()

	rl = RecurrentLayer(3, 2, IdentityActivator(), 1e-3)

	# 计算forward值
	x, d = data_set()
	rl.forward(x[0])
	rl.forward(x[1])

	# 求取sensitivity map
	sensitivity_array = np.ones(rl.state_list[-1].shape, dtype=np.float64)
	# 计算梯度
	rl.backward(sensitivity_array, IdentityActivator())

	# 检查梯度
	epsilon = 10e-4
	for i in range(rl.W.shape[0]):
		for j in range(rl.W.shape[1]):
			rl.W[i, j] += epsilon
			rl.reset_state()
			rl.forward(x[0])
			rl.forward(x[1])
			err1 = error_function(rl.state_list[-1])
			rl.W[i, j] -= 2 * epsilon
			rl.reset_state()
			rl.forward(x[0])
			rl.forward(x[1])
			err2 = error_function(rl.state_list[-1])
			expect_grad = (err1 - err2) / (2 * epsilon)
			rl.W[i, j] += epsilon
			print 'weights(%d,%d): expected - actural %f - %f' % (i, j, expect_grad, rl.gradient[i, j])


def test():
	x, d = data_set()
	l = RecurrentLayer(3, 2, ReluActivator(), 1e-3)
	print "X:", x, "\ny:", d
	l.forward(x[0])
	l.forward(x[1])
	l.backward(d, ReluActivator())
	print "activation:", l.activator, "\ndelta_list", l.delta_list, "\ngradient_list：", l.gradient_list,"\nstate_list：",l.state_list,"\nW：",l.W,"\nU：",l.U
	return l


if __name__ == "__main__":
	test()
	gradient_check()
