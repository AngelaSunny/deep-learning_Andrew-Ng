#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import matplotlib.pyplot as plt
import numpy as np
from cnn import element_wise_op
from activators import SigmoidActivator, TanhActivator, IdentityActivator


class LstmLayer(object):
	'''
	隐藏层相比rnn（只有一个hidden state）,额外增加cell state
	怎样控制长期状态cell state？有三个开关，（每个开关的实现涉及到门的输入到输出的激活计算）:
		第一个开关，负责控制继续保存长期状态c；
			遗忘门：决定上一个时刻的单元状态Ct-1有多少保留到当前时刻的Ct
					1.遗忘门：Ft=f(Wf*[Ht-1,Xt]+Bf)
		第二个开关，负责控制把即时状态输入到长期状态c；
			输入门：决定当前时刻网络的输入Xt有多少保存到单元状态Ct
					1.输入门:It=f(Wc*[Ht-1,Xt]+Bi)
					2.当前输入的单元状态Ct' = tanh(Wc*[Ht-1,Xt]+Bc)
					3.当前时刻的单元状态（当前的记忆Ct'和长期的记忆Ct-1融合一起）Ct = Ft。Ct-1+It。Ct'
		第三个开关，负责控制是否把长期状态c作为当前的LSTM的输出。
			输出门：控制单元状态Ct有多少输出到LSTM的当前输出值ht;
					1.输出门：Ot = f(Wo*[Ht-1,Xt]+Bo)
					2.LSTM最终的输出：Ht = Ot。tanh(Ct)

	'''

	def __init__(self, input_width, state_width, learning_rate):
		self.input_width = input_width
		self.state_width = state_width
		self.learning_rate = learning_rate
		# 门的激活函数
		self.gate_activator = SigmoidActivator()
		# 输出的激活函数
		self.output_activator = TanhActivator()
		# 当前时刻初始化为t0
		self.times = 0
		# 各个时刻的单元cell状态向量c
		self.c_list = self.init_state_vec()
		# 各个时刻的输出向量h
		self.h_list = self.init_state_vec()
		# 各个时刻的遗忘门f
		self.f_list = self.init_state_vec()
		# 各个时刻的输入门i
		self.i_list = self.init_state_vec()
		# 各个时刻的输出门o
		self.o_list = self.init_state_vec()
		# 各个时刻的即时状态c~
		self.ct_list = self.init_state_vec()
		# 遗忘门权重矩阵Wfh, Wfx, 偏置项bf
		self.Wfh, self.Wfx, self.bf = (self.init_weight_mat())
		# 输入门权重矩阵Wfh, Wfx, 偏置项bf
		self.Wih, self.Wix, self.bi = (self.init_weight_mat())
		# 输出门权重矩阵Wfh, Wfx, 偏置项bf
		self.Woh, self.Wox, self.bo = (self.init_weight_mat())
		# 单元状态权重矩阵Wfh, Wfx, 偏置项bf
		self.Wch, self.Wcx, self.bc = (self.init_weight_mat())

	def init_state_vec(self):
		'''
		初始化保存状态的向量
		'''
		state_vec_list = []
		state_vec_list.append(np.zeros((self.state_width, 1)))
		return state_vec_list

	def init_weight_mat(self):
		'''
		初始化权重矩阵
		'''
		Wh = np.random.uniform(-1e-4, 1e-4, (self.state_width, self.state_width))
		Wx = np.random.uniform(-1e-4, 1e-4, (self.state_width, self.input_width))
		b = np.zeros((self.state_width, 1))
		return Wh, Wx, b

	def forward(self, x):
		'''
		前向传播
		'''
		self.times += 1
		# 遗忘门
		fg = self.calc_gate(x, self.Wfx, self.Wfh, self.bf, self.gate_activator)
		self.f_list.append(fg)
		# 输入门
		ig = self.calc_gate(x, self.Wix, self.Wih, self.bi, self.gate_activator)
		self.i_list.append(ig)
		# 输出门
		og = self.calc_gate(x, self.Wox, self.Woh, self.bo, self.gate_activator)
		self.o_list.append(og)
		# 即时状态
		ct = self.calc_gate(x, self.Wcx, self.Wch, self.bc, self.output_activator)
		self.ct_list.append(ct)
		# 单元状态
		c = fg * self.c_list[self.times - 1] + ig * ct
		self.c_list.append(c)
		# 输出
		h = og * self.output_activator(c)
		self.h_list.append(h)

	def calc_gate(self, x, Wx, Wh, b, activator):
		'''
		计算门
		'''
		h = self.h_list[self.times - 1]  # 上次的LSTM输出
		net = np.dot(Wh, h) + np.dot(Wx, x) + b
		gate = activator.forward(net)
		return gate

	def backward(self, x, delta_h, activator):
		'''
		实现lstm训练算法
		'''
		self.calc_delta(delta_h, activator)
		self.calc_gradient(x)

	def calc_delta(self, delta_h, activator):
		##初始化各个时刻误差
		self.delta_h_list = self.init_delta()  # 输出误差项
		self.delta_o_list = self.init_delta()  # 输出门误差项
		self.delta_i_list = self.init_delta()  # 输入门误差项
		self.delta_f_list = self.init_delta()  # 遗忘门误差项
		self.delta_ct_list = self.init_delta()  # 即时输出误差项
		# 保存上一层传递下来的当前时刻的误差项
		self.delta_h_list[-1] = delta_h
		# 迭代计算每个时刻的误差项
		for k in range(self.times, 0, -1):
			self.calc_delta_k(k)

	def calc_delta_k(self, k):
		'''
		t时刻的误差沿着时间的反向传播公式：
		根据k时刻的delta_h，计算k时刻的delta_f、delta_i、delta_o、delta_ct，以及k-1时刻的delta_h
		'''
		ig = self.i_list[k]
		og = self.o_list[k]
		fg = self.f_list[k]
		ct = self.ct_list[k]
		c = self.c_list[k]
		c_prev = self.c_list[k - 1]
		tanh_c = self.output_activator.forward(c)
		delta_k = self.delta_h_list[k]
		delta_o = (delta_k * tanh_c * self.gate_activator.backward(og))
		delta_f = (delta_k * og * (1 - tanh_c * tanh_c) * c_prev * self.gate_activator.backward(fg))
		delta_i = (delta_k * og * (1 - tanh_c * tanh_c) * ct * self.gate_activator.backward(ig))
		delta_ct = (delta_k * og * (1 - tanh_c * tanh_c) * ig * self.output_activator.backward(ct))
		delta_h_prev = (np.dot(delta_o.transpose(), self.Woh)
		                + np.dot(delta_i.transpose(), self.Wih)
		                + np.dot(delta_f.transpose(), self.Wfh)
		                + np.dot(delta_ct.transpose(), self.Wch)).transpose()
		# 保存全部的delta值
		self.delta_h_list[k - 1] = delta_h_prev
		self.delta_f_list[k] = delta_f
		self.delta_i_list[k] = delta_i
		self.delta_o_list[k] = delta_o
		self.delta_ct_list[k] = delta_ct

	def init_delta(self):
		'''
		初始化误差项
		'''
		delta_list = []
		for i in range(self.times + 1):
			delta_list.append(np.zeros((self.state_width, 1)))
		return delta_list

	def calc_gradient(self, x):
		# 初始化遗忘门权重梯度矩阵和偏置项
		self.Wfh_grad, self.Wfx_grad, self.bf_grad = (self.init_weight_gradient_mat())
		# 初始化输入门权重梯度矩阵和偏置项
		self.Wih_grad, self.Wix_grad, self.bi_grad = (self.init_weight_gradient_mat())
		# 初始化输出门权重梯度矩阵和偏置项
		self.Woh_grad, self.Wox_grad, self.bo_grad = (self.init_weight_gradient_mat())
		# 初始化单元状态权重梯度矩阵和偏置项
		self.Wch_grad, self.Wcx_grad, self.bc_grad = (self.init_weight_gradient_mat())
		# 计算对上一次输出h的权重梯度
		for t in range(self.times, 0, -1):
			# 计算各个时刻的梯度
			(Wfh_grad, bf_grad, Wih_grad, bi_grad, Woh_grad, bo_grad, Wch_grad, bc_grad) = (self.calc_gradient_t(t))

	def init_weight_gradient_mat(self):
		'''
		初始化权重矩阵
		'''
		Wh_grad = np.zeros((self.state_width, self.state_width))
		Wx_grad = np.zeros((self.state_width, self.input_width))
		b_grad = np.zeros((self.state_width, 1))
		return Wh_grad, Wx_grad, b_grad

	def calc_gradient_t(self, t):
		'''
		计算每个时刻t权重的梯度
		'''
		h_prev = self.h_list[t - 1].transpose()
		Wf