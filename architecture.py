##################################################
## Project: CapsuleNetworks
## Script purpose: Utility file for functions
## Date: 22nd April 2018
## Author: Chaitanya Baweja, Imperial College London
##################################################
import tensorflow as tf
import numpy as np
import os

'''
Batch Normalization
input: input layer
Returns Normalized output
This is used to act as regularization
'''
def batch_norm(input, name="batch_norm"):
	#validates that the values are from the same graph 
	#and pushes a name scope and a variable scope.
	with tf.variable_scope(name) as scope:
		#used as a dummy node to update a reference to the tensor 
		input = tf.identity(input)
		channels = input.get_shape()[3]
		
		#create new variables in this scope
		offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
		scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
		
		#calculating mean and variance
		mean, variance = tf.nn.moments(input, axes=[0,1,2], keep_dims=False)

		normalized_batch = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=1e-5)

		return normalized_batch

'''
Max_pool operation
input: input layer
kernel_size: dimension of kernels
stride: length of stride
Returns max pooled Tensor
'''
def max_pool(inputs, kernel_size=5, stride=1, scope=None, name=""):
  return tf.nn.max_pool(inputs,
                       ksize=[1, kernel_size, kernel_size, 1],
                       strides=[1, stride, stride, 1],
                       padding='SAME')

#Function to count parameters
def count_param(total_param):
	total_num = 0
	for v in total_param:
		shape = v.get_shape()
		variable_parameters = 1
		for dim in shape:
			variable_parameters *= dim.value

		total_num += variable_parameters

	return total_num
