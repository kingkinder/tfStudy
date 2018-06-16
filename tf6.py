# 神经网络学习
'''
1实现门函数
2使用门函数和激励函数
3实现单层神经网络
4实现神经网络常见层
5使用多层神经网络
6线性预测模型优化
7基于神经网络实现#字棋游戏
'''

import tensorflow as tf
from tensorflow.python.framework import ops

import matplotlib.pyplot as plt
import numpy as np
# Introduce tensors in tf
#--------------------6.1门函数------------------
# Implementing Gates
#----------------------------------
#
# This function shows how to implement
# various gates in Tensorflow
#
# One gate will be one operation with
# a variable and a placeholder.
# We will ask Tensorflow to change the
# variable based on our loss function

def _tf601Gate():
	ops.reset_default_graph()
	# Start Graph Session

	sess = tf.Session()
	#----------------------------------
	# Create a multiplication gate:
	#   f(x) = a * x
	#
	#  a --
	#      |
	#      |---- (multiply) --> output
	#  x --|
	#

	a = tf.Variable(tf.constant(4.))
	x_val = 5.
	x_data = tf.placeholder(dtype=tf.float32)

	multiplication = tf.multiply(a, x_data)

	# Declare the loss function as the difference between
	# the output and a target value, 50.
	loss = tf.square(tf.subtract(multiplication, 50.))

	# Initialize variables
	init = tf.global_variables_initializer()
	sess.run(init)

	# Declare optimizer
	my_opt = tf.train.GradientDescentOptimizer(0.01)
	train_step = my_opt.minimize(loss)

	# Run loop across gate
	print('Optimizing a Multiplication Gate Output to 50.')
	for i in range(10):
		sess.run(train_step, feed_dict={x_data: x_val})
		a_val = sess.run(a)
		mult_output = sess.run(multiplication, feed_dict={x_data: x_val})
		print(str(a_val) + ' * ' + str(x_val) + ' = ' + str(mult_output))
		
	#----------------------------------
	# Create a nested gate:
	#   f(x) = a * x + b
	#
	#  a --
	#      |
	#      |-- (multiply)--
	#  x --|              |
	#                     |-- (add) --> output
	#                 b --|
	#
	#

	# Start a New Graph Session
	ops.reset_default_graph()
	sess = tf.Session()

	a = tf.Variable(tf.constant(1.))
	b = tf.Variable(tf.constant(1.))
	x_val = 5.
	x_data = tf.placeholder(dtype=tf.float32)

	two_gate = tf.add(tf.multiply(a, x_data), b)

	# Declare the loss function as the difference between
	# the output and a target value, 50.
	loss = tf.square(tf.subtract(two_gate, 50.))

	# Initialize variables
	init = tf.initialize_all_variables()
	sess.run(init)

	# Declare optimizer
	my_opt = tf.train.GradientDescentOptimizer(0.01)
	train_step = my_opt.minimize(loss)

	# Run loop across gate
	print('\nOptimizing Two Gate Output to 50.')
	for i in range(10):
		sess.run(train_step, feed_dict={x_data: x_val})
		a_val, b_val = (sess.run(a), sess.run(b))
		two_gate_output = sess.run(two_gate, feed_dict={x_data: x_val})
		print(str(a_val) + ' * ' + str(x_val) + ' + ' + str(b_val) + ' = ' + str(two_gate_output))

#--------------------6.2 激活函数------------------
# Implementing Gates
#----------------------------------
#
# This function shows how to implement
# various gates in Tensorflow
#
# One gate will be one operation with
# a variable and a placeholder.
# We will ask Tensorflow to change the
# variable based on our loss function

def _tf602Active():

	ops.reset_default_graph()

	# Start Graph Session
	sess = tf.Session()
	tf.set_random_seed(5)
	np.random.seed(42)

	batch_size = 50

	a1 = tf.Variable(tf.random_normal(shape=[1,1]))
	b1 = tf.Variable(tf.random_uniform(shape=[1,1]))
	a2 = tf.Variable(tf.random_normal(shape=[1,1]))
	b2 = tf.Variable(tf.random_uniform(shape=[1,1]))
	x = np.random.normal(2, 0.1, 500)
	x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)

	sigmoid_activation = tf.sigmoid(tf.add(tf.matmul(x_data, a1), b1))

	relu_activation = tf.nn.relu(tf.add(tf.matmul(x_data, a2), b2))

	# Declare the loss function as the difference between
	# the output and a target value, 0.75.
	loss1 = tf.reduce_mean(tf.square(tf.subtract(sigmoid_activation, 0.75)))
	loss2 = tf.reduce_mean(tf.square(tf.subtract(relu_activation, 0.75)))

	# Initialize variables
	init = tf.global_variables_initializer()
	sess.run(init)

	# Declare optimizer
	my_opt = tf.train.GradientDescentOptimizer(0.01)
	train_step_sigmoid = my_opt.minimize(loss1)
	train_step_relu = my_opt.minimize(loss2)

	# Run loop across gate
	print('\nOptimizing Sigmoid AND Relu Output to 0.75')
	loss_vec_sigmoid = []
	loss_vec_relu = []
	activation_sigmoid = []
	activation_relu = []
	for i in range(750):
		rand_indices = np.random.choice(len(x), size=batch_size)
		x_vals = np.transpose([x[rand_indices]])
		sess.run(train_step_sigmoid, feed_dict={x_data: x_vals})
		sess.run(train_step_relu, feed_dict={x_data: x_vals})
		
		loss_vec_sigmoid.append(sess.run(loss1, feed_dict={x_data: x_vals}))
		loss_vec_relu.append(sess.run(loss2, feed_dict={x_data: x_vals}))    
		
		activation_sigmoid.append(np.mean(sess.run(sigmoid_activation, feed_dict={x_data: x_vals})))
		activation_relu.append(np.mean(sess.run(relu_activation, feed_dict={x_data: x_vals})))


	# Plot the activation values
	plt.plot(activation_sigmoid, 'k-', label='Sigmoid Activation')
	plt.plot(activation_relu, 'r--', label='Relu Activation')
	plt.ylim([0, 1.0])
	plt.title('Activation Outputs')
	plt.xlabel('Generation')
	plt.ylabel('Outputs')
	plt.legend(loc='upper right')
	plt.show()

		
	# Plot the loss
	plt.plot(loss_vec_sigmoid, 'k-', label='Sigmoid Loss')
	plt.plot(loss_vec_relu, 'r--', label='Relu Loss')
	plt.ylim([0, 1.0])
	plt.title('Loss per Generation')
	plt.xlabel('Generation')
	plt.ylabel('Loss')
	plt.legend(loc='upper right')
	plt.show()

#--------------------6.3 单层神经网络------------------
# Implementing a one-layer Neural Network
#---------------------------------------
#
# We will illustrate how to create a one hidden layer NN
#
# We will use the iris data for this exercise
#
# We will build a one-hidden layer neural network
#  to predict the fourth attribute, Petal Width from
#  the other three (Sepal length, Sepal width, Petal length).

def _tf603SIGlayer():
	ops.reset_default_graph()
#加载Iris数据集
	iris = datasets.load_iris()
	x_vals = np.array([x[0:3] for x in iris.data])
	y_vals = np.array([x[3] for x in iris.data])

	# Create graph session 
	sess = tf.Session()

	# Set Seed
	seed = 3
	tf.set_random_seed(seed)
	np.random.seed(seed)

	# Split data into train/test = 80%/20%
	train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
	test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
	x_vals_train = x_vals[train_indices]
	x_vals_test = x_vals[test_indices]
	y_vals_train = y_vals[train_indices]
	y_vals_test = y_vals[test_indices]

	# Normalize by column (min-max norm)
	def normalize_cols(m):
		col_max = m.max(axis=0)
		col_min = m.min(axis=0)
		return (m-col_min) / (col_max - col_min)
		
	x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
	x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

	# Declare batch size
	batch_size = 50

	# Initialize placeholders
	x_data = tf.placeholder(shape=[None, 3], dtype=tf.float32)
	y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

	# Create variables for both Neural Network Layers
	hidden_layer_nodes = 10
	A1 = tf.Variable(tf.random_normal(shape=[3,hidden_layer_nodes])) # inputs -> hidden nodes
	b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))   # one biases for each hidden node
	A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,1])) # hidden inputs -> 1 output
	b2 = tf.Variable(tf.random_normal(shape=[1]))   # 1 bias for the output


	# Declare model operations
	hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, A1), b1))
	final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output, A2), b2))

	# Declare loss function
	loss = tf.reduce_mean(tf.square(y_target - final_output))

	# Declare optimizer
	my_opt = tf.train.GradientDescentOptimizer(0.005)
	train_step = my_opt.minimize(loss)

	# Initialize variables
	init = tf.initialize_all_variables()
	sess.run(init)

	# Training loop
	loss_vec = []
	test_loss = []
	for i in range(500):
		rand_index = np.random.choice(len(x_vals_train), size=batch_size)
		rand_x = x_vals_train[rand_index]
		rand_y = np.transpose([y_vals_train[rand_index]])
		sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

		temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
		loss_vec.append(np.sqrt(temp_loss))
		
		test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
		test_loss.append(np.sqrt(test_temp_loss))
		if (i+1)%50==0:
			print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss))


	# Plot loss (MSE) over time
	plt.plot(loss_vec, 'k-', label='Train Loss')
	plt.plot(test_loss, 'r--', label='Test Loss')
	plt.title('Loss (MSE) per Generation')
	plt.xlabel('Generation')
	plt.ylabel('Loss')
	plt.legend(loc='upper right')
	plt.show()


#--------------------6.4 神经网络常见层------------------
# Implementing Different Layers
#---------------------------------------
#
# We will illustrate how to use different types
# of layers in Tensorflow
#
# The layers of interest are:
#  (1) Convolutional Layer
#  (2) Activation Layer
#  (3) Max-Pool Layer
#  (4) Fully Connected Layer
#
# We will generate two different data sets for this
#  script, a 1-D data set (row of data) and
#  a 2-D data set (similar to picture)

def _tf604SIGlayer():
	ops.reset_default_graph()

	#---------------------------------------------------|
	#-------------------1D-data-------------------------|
	#---------------------------------------------------|
	print('\n----------1D Arrays----------')

	# Create graph session 
	sess = tf.Session()

	# Generate 1D data
	data_size = 25
	data_1d = np.random.normal(size=data_size)

	# Placeholder
	x_input_1d = tf.placeholder(dtype=tf.float32, shape=[data_size])

	#--------Convolution--------
	def conv_layer_1d(input_1d, my_filter):
		# Tensorflow's 'conv2d()' function only works with 4D arrays:
		# [batch#, width, height, channels], we have 1 batch, and
		# width = 1, but height = the length of the input, and 1 channel.
		# So next we create the 4D array by inserting dimension 1's.
		input_2d = tf.expand_dims(input_1d, 0)
		input_3d = tf.expand_dims(input_2d, 0)
		input_4d = tf.expand_dims(input_3d, 3)
		# Perform convolution with stride = 1, if we wanted to increase the stride,
		# to say '2', then strides=[1,1,2,1]
		convolution_output = tf.nn.conv2d(input_4d, filter=my_filter, strides=[1,1,1,1], padding="VALID")
		# Get rid of extra dimensions
		conv_output_1d = tf.squeeze(convolution_output)
		return(conv_output_1d)

	# Create filter for convolution.
	my_filter = tf.Variable(tf.random_normal(shape=[1,5,1,1]))
	# Create convolution layer
	my_convolution_output = conv_layer_1d(x_input_1d, my_filter)

	#--------Activation--------
	def activation(input_1d):
		return(tf.nn.relu(input_1d))

	# Create activation layer
	my_activation_output = activation(my_convolution_output)

	#--------Max Pool--------
	def max_pool(input_1d, width):
		# Just like 'conv2d()' above, max_pool() works with 4D arrays.
		# [batch_size=1, width=1, height=num_input, channels=1]
		input_2d = tf.expand_dims(input_1d, 0)
		input_3d = tf.expand_dims(input_2d, 0)
		input_4d = tf.expand_dims(input_3d, 3)
		# Perform the max pooling with strides = [1,1,1,1]
		# If we wanted to increase the stride on our data dimension, say by
		# a factor of '2', we put strides = [1, 1, 2, 1]
		# We will also need to specify the width of the max-window ('width')
		pool_output = tf.nn.max_pool(input_4d, ksize=[1, 1, width, 1],
									 strides=[1, 1, 1, 1],
									 padding='VALID')
		# Get rid of extra dimensions
		pool_output_1d = tf.squeeze(pool_output)
		return(pool_output_1d)

	my_maxpool_output = max_pool(my_activation_output, width=5)

	#--------Fully Connected--------
	def fully_connected(input_layer, num_outputs):
		# First we find the needed shape of the multiplication weight matrix:
		# The dimension will be (length of input) by (num_outputs)
		weight_shape = tf.squeeze(tf.pack([tf.shape(input_layer),[num_outputs]]))
		# Initialize such weight
		weight = tf.random_normal(weight_shape, stddev=0.1)
		# Initialize the bias
		bias = tf.random_normal(shape=[num_outputs])
		# Make the 1D input array into a 2D array for matrix multiplication
		input_layer_2d = tf.expand_dims(input_layer, 0)
		# Perform the matrix multiplication and add the bias
		full_output = tf.add(tf.matmul(input_layer_2d, weight), bias)
		# Get rid of extra dimensions
		full_output_1d = tf.squeeze(full_output)
		return(full_output_1d)

	my_full_output = fully_connected(my_maxpool_output, 5)

	# Run graph
	# Initialize Variables
	init = tf.initialize_all_variables()
	sess.run(init)

	feed_dict = {x_input_1d: data_1d}

	# Convolution Output
	print('Input = array of length 25')
	print('Convolution w/filter, length = 5, stride size = 1, results in an array of length 21:')
	print(sess.run(my_convolution_output, feed_dict=feed_dict))

	# Activation Output
	print('\nInput = the above array of length 21')
	print('ReLU element wise returns the array of length 21:')
	print(sess.run(my_activation_output, feed_dict=feed_dict))

	# Max Pool Output
	print('\nInput = the above array of length 21')
	print('MaxPool, window length = 5, stride size = 1, results in the array of length 17:')
	print(sess.run(my_maxpool_output, feed_dict=feed_dict))

	# Fully Connected Output
	print('\nInput = the above array of length 17')
	print('Fully connected layer on all four rows with five outputs:')
	print(sess.run(my_full_output, feed_dict=feed_dict))

	#---------------------------------------------------|
	#-------------------2D-data-------------------------|
	#---------------------------------------------------|
	print('\n----------2D Arrays----------')


	# Reset Graph
	ops.reset_default_graph()
	sess = tf.Session()

	#Generate 2D data
	data_size = [10,10]
	data_2d = np.random.normal(size=data_size)

	#--------Placeholder--------
	x_input_2d = tf.placeholder(dtype=tf.float32, shape=data_size)

	# Convolution
	def conv_layer_2d(input_2d, my_filter):
		# Tensorflow's 'conv2d()' function only works with 4D arrays:
		# [batch#, width, height, channels], we have 1 batch, and
		# 1 channel, but we do have width AND height this time.
		# So next we create the 4D array by inserting dimension 1's.
		input_3d = tf.expand_dims(input_2d, 0)
		input_4d = tf.expand_dims(input_3d, 3)
		# Note the stride difference below!
		convolution_output = tf.nn.conv2d(input_4d, filter=my_filter, strides=[1,2,2,1], padding="VALID")
		# Get rid of unnecessary dimensions
		conv_output_2d = tf.squeeze(convolution_output)
		return(conv_output_2d)

	# Create Convolutional Filter
	my_filter = tf.Variable(tf.random_normal(shape=[2,2,1,1]))
	# Create Convolutional Layer
	my_convolution_output = conv_layer_2d(x_input_2d, my_filter)

	#--------Activation--------
	def activation(input_2d):
		return(tf.nn.relu(input_2d))

	# Create Activation Layer
	my_activation_output = activation(my_convolution_output)

	#--------Max Pool--------
	def max_pool(input_2d, width, height):
		# Just like 'conv2d()' above, max_pool() works with 4D arrays.
		# [batch_size=1, width=given, height=given, channels=1]
		input_3d = tf.expand_dims(input_2d, 0)
		input_4d = tf.expand_dims(input_3d, 3)
		# Perform the max pooling with strides = [1,1,1,1]
		# If we wanted to increase the stride on our data dimension, say by
		# a factor of '2', we put strides = [1, 2, 2, 1]
		pool_output = tf.nn.max_pool(input_4d, ksize=[1, height, width, 1],
									 strides=[1, 1, 1, 1],
									 padding='VALID')
		# Get rid of unnecessary dimensions
		pool_output_2d = tf.squeeze(pool_output)
		return(pool_output_2d)

	# Create Max-Pool Layer
	my_maxpool_output = max_pool(my_activation_output, width=2, height=2)


	#--------Fully Connected--------
	def fully_connected(input_layer, num_outputs):
		# In order to connect our whole W byH 2d array, we first flatten it out to
		# a W times H 1D array.
		flat_input = tf.reshape(input_layer, [-1])
		# We then find out how long it is, and create an array for the shape of
		# the multiplication weight = (WxH) by (num_outputs)
		weight_shape = tf.squeeze(tf.pack([tf.shape(flat_input),[num_outputs]]))
		# Initialize the weight
		weight = tf.random_normal(weight_shape, stddev=0.1)
		# Initialize the bias
		bias = tf.random_normal(shape=[num_outputs])
		# Now make the flat 1D array into a 2D array for multiplication
		input_2d = tf.expand_dims(flat_input, 0)
		# Multiply and add the bias
		full_output = tf.add(tf.matmul(input_2d, weight), bias)
		# Get rid of extra dimension
		full_output_2d = tf.squeeze(full_output)
		return(full_output_2d)

	# Create Fully Connected Layer
	my_full_output = fully_connected(my_maxpool_output, 5)

	# Run graph
	# Initialize Variables
	init = tf.initialize_all_variables()
	sess.run(init)

	feed_dict = {x_input_2d: data_2d}

	# Convolution Output
	print('Input = [10 X 10] array')
	print('2x2 Convolution, stride size = [2x2], results in the [5x5] array:')
	print(sess.run(my_convolution_output, feed_dict=feed_dict))

	# Activation Output
	print('\nInput = the above [5x5] array')
	print('ReLU element wise returns the [5x5] array:')
	print(sess.run(my_activation_output, feed_dict=feed_dict))

	# Max Pool Output
	print('\nInput = the above [5x5] array')
	print('MaxPool, stride size = [1x1], results in the [4x4] array:')
	print(sess.run(my_maxpool_output, feed_dict=feed_dict))

	# Fully Connected Output
	print('\nInput = the above [4x4] array')
	print('Fully connected layer on all four rows with five outputs:')
	print(sess.run(my_full_output, feed_dict=feed_dict))

#--------------------6.5 多层神经网络------------------
# Using a Multiple Layer Network
#---------------------------------------
#
# We will illustrate how to use a Multiple
# Layer Network in Tensorflow
#
# Low Birthrate data:
#
#Columns    Variable                                              Abbreviation
#-----------------------------------------------------------------------------
# Identification Code                                     ID
# Low Birth Weight (0 = Birth Weight >= 2500g,            LOW
#                          1 = Birth Weight < 2500g)
# Age of the Mother in Years                              AGE
# Weight in Pounds at the Last Menstrual Period           LWT
# Race (1 = White, 2 = Black, 3 = Other)                  RACE
# Smoking Status During Pregnancy (1 = Yes, 0 = No)       SMOKE
# History of Premature Labor (0 = None  1 = One, etc.)    PTL
# History of Hypertension (1 = Yes, 0 = No)               HT
# Presence of Uterine Irritability (1 = Yes, 0 = No)      UI
# Number of Physician Visits During the First Trimester   FTV
#                (0 = None, 1 = One, 2 = Two, etc.)
# Birth Weight in Grams                                   BWT
#------------------------------
# The multiple neural network layer we will create will be composed of
# three fully connected hidden layers, with node sizes 25, 10, and 3
import requests
def _tf605Multilayer():
	ops.reset_default_graph()
	# Set Seed
	seed = 3
	tf.set_random_seed(seed)
	np.random.seed(seed)


	birthdata_url = 'https://www.umass.edu/statdata/statdata/data/lowbwt.dat'
	birth_file = requests.get(birthdata_url)
	birth_data = birth_file.text.split('\r\n')[5:]
	birth_header = [x for x in birth_data[0].split(' ') if len(x)>=1]
	birth_data = [[float(x) for x in y.split(' ') if len(x)>=1] for y in birth_data[1:] if len(y)>=1]


	batch_size = 100

	# Extract y-target (birth weight)
	y_vals = np.array([x[10] for x in birth_data])

	# Filter for features of interest
	cols_of_interest = ['AGE', 'LWT', 'RACE', 'SMOKE', 'PTL', 'HT', 'UI', 'FTV']
	x_vals = np.array([[x[ix] for ix, feature in enumerate(birth_header) if feature in cols_of_interest] for x in birth_data])

	# Create graph session 
	sess = tf.Session()

	# Split data into train/test = 80%/20%
	train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
	test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
	x_vals_train = x_vals[train_indices]
	x_vals_test = x_vals[test_indices]
	y_vals_train = y_vals[train_indices]
	y_vals_test = y_vals[test_indices]


	# Normalize by column (min-max norm to be between 0 and 1)
	def normalize_cols(m):
		col_max = m.max(axis=0)
		col_min = m.min(axis=0)
		return (m-col_min) / (col_max - col_min)
		
	x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
	x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))


	# Define Variable Functions (weights and bias)
	def init_weight(shape, st_dev):
		weight = tf.Variable(tf.random_normal(shape, stddev=st_dev))
		return(weight)
		

	def init_bias(shape, st_dev):
		bias = tf.Variable(tf.random_normal(shape, stddev=st_dev))
		return(bias)
		
		
	# Create Placeholders
	x_data = tf.placeholder(shape=[None, 8], dtype=tf.float32)
	y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)


	# Create a fully connected layer:
	def fully_connected(input_layer, weights, biases):
		layer = tf.add(tf.matmul(input_layer, weights), biases)
		return(tf.nn.relu(layer))


	#--------Create the first layer (25 hidden nodes)--------
	weight_1 = init_weight(shape=[8, 25], st_dev=10.0)
	bias_1 = init_bias(shape=[25], st_dev=10.0)
	layer_1 = fully_connected(x_data, weight_1, bias_1)

	#--------Create second layer (10 hidden nodes)--------
	weight_2 = init_weight(shape=[25, 10], st_dev=10.0)
	bias_2 = init_bias(shape=[10], st_dev=10.0)
	layer_2 = fully_connected(layer_1, weight_2, bias_2)


	#--------Create third layer (3 hidden nodes)--------
	weight_3 = init_weight(shape=[10, 3], st_dev=10.0)
	bias_3 = init_bias(shape=[3], st_dev=10.0)
	layer_3 = fully_connected(layer_2, weight_3, bias_3)


	#--------Create output layer (1 output value)--------
	weight_4 = init_weight(shape=[3, 1], st_dev=10.0)
	bias_4 = init_bias(shape=[1], st_dev=10.0)
	final_output = fully_connected(layer_3, weight_4, bias_4)

	# Declare loss function (L1)
	loss = tf.reduce_mean(tf.abs(y_target - final_output))

	# Declare optimizer
	my_opt = tf.train.AdamOptimizer(0.05)
	train_step = my_opt.minimize(loss)

	# Initialize Variables
	init = tf.initialize_all_variables()
	sess.run(init)

	# Training loop
	loss_vec = []
	test_loss = []
	for i in range(200):
		rand_index = np.random.choice(len(x_vals_train), size=batch_size)
		rand_x = x_vals_train[rand_index]
		rand_y = np.transpose([y_vals_train[rand_index]])
		sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

		temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
		loss_vec.append(temp_loss)
		
		test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
		test_loss.append(test_temp_loss)
		if (i+1)%25==0:
			print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss))


	# Plot loss over time
	plt.plot(loss_vec, 'k-', label='Train Loss')
	plt.plot(test_loss, 'r--', label='Test Loss')
	plt.title('Loss per Generation')
	plt.xlabel('Generation')
	plt.ylabel('Loss')
	plt.legend(loc="upper right")
	plt.show()

	# Find the % classified correctly above/below the cutoff of 2500 g
	# >= 2500 g = 0
	# < 2500 g = 1
	actuals = np.array([x[1] for x in birth_data])
	test_actuals = actuals[test_indices]
	train_actuals = actuals[train_indices]

	test_preds = [x[0] for x in sess.run(final_output, feed_dict={x_data: x_vals_test})]
	train_preds = [x[0] for x in sess.run(final_output, feed_dict={x_data: x_vals_train})]
	test_preds = np.array([1.0 if x<2500.0 else 0.0 for x in test_preds])
	train_preds = np.array([1.0 if x<2500.0 else 0.0 for x in train_preds])

	# Print out accuracies
	test_acc = np.mean([x==y for x,y in zip(test_preds, test_actuals)])
	train_acc = np.mean([x==y for x,y in zip(train_preds, train_actuals)])
	print('On predicting the category of low birthweight from regression output (<2500g):')
	print('Test Accuracy: {}'.format(test_acc))
	print('Train Accuracy: {}'.format(train_acc))	

	
	
	



#--------------main()-------------

if __name__=='__main__':
	_tf602Active()



