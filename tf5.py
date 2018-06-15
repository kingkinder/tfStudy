'''
5.1最近邻法的使用
5.2如何度量文本距离
5.3实现混合距离
5.4实现地址匹配
5.5实现图像识别
'''
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
import requests
import pandas as pd
from tensorflow.python.framework import ops
#--------------------5.1KNN使用------------------
# k-Nearest Neighbor
#----------------------------------
#
# This function illustrates how to use
# k-nearest neighbors in tensorflow
#
# We will use the 1970s Boston housing dataset
# which is available through the UCI
# ML data repository.
#
# Data:
#----------x-values-----------
# CRIM   : per capita crime rate by town
# ZN     : prop. of res. land zones
# INDUS  : prop. of non-retail business acres
# CHAS   : Charles river dummy variable
# NOX    : nitrix oxides concentration / 10 M
# RM     : Avg. # of rooms per building
# AGE    : prop. of buildings built prior to 1940
# DIS    : Weighted distances to employment centers
# RAD    : Index of radian highway access
# TAX    : Full tax rate value per $10k
# PTRATIO: Pupil/Teacher ratio by town
# B      : 1000*(Bk-0.63)^2, Bk=prop. of blacks
# LSTAT  : % lower status of pop
#------------y-value-----------
# MEDV   : Median Value of homes in $1,000's

def _tf501CalNeighbor():
	ops.reset_default_graph()

	# Create graph
	sess = tf.Session()

	# Load the data
	# housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
	housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
	cols_used = ['CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']
	num_features = len(cols_used)
	# housing_file = requests.get(housing_url)
	# housing_data = [[float(x) for x in y.split(' ') if len(x)>=1] for y in housing_file.text.split('\n') if len(y)>=1]
	housing_file=pd.read_csv('datahousing.csv')
	housing_data=np.array(housing_file).tolist()
	y_vals = np.transpose([np.array([y[13] for y in housing_data])])
	x_vals = np.array([[x for i,x in enumerate(y) if housing_header[i] in cols_used] for y in housing_data])

	## Min-Max Scaling
	x_vals = (x_vals - x_vals.min(0)) / x_vals.ptp(0)

	# Split the data into train and test sets
	train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
	test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
	x_vals_train = x_vals[train_indices]
	x_vals_test = x_vals[test_indices]
	y_vals_train = y_vals[train_indices]
	y_vals_test = y_vals[test_indices]

	# Declare k-value and batch size
	k = 4
	batch_size=len(x_vals_test)

	# Placeholders
	x_data_train = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
	x_data_test = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
	y_target_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
	y_target_test = tf.placeholder(shape=[None, 1], dtype=tf.float32)

	# Declare distance metric
	# L1
	distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1))), reduction_indices=2)

	# L2
	#distance = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(x_data_train, tf.expand_dims(x_data_test,1))), reduction_indices=1))

	# Predict: Get min distance index (Nearest neighbor)
	top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
	x_sums = tf.expand_dims(tf.reduce_sum(top_k_xvals, 1),1)
	x_sums_repeated = tf.matmul(x_sums,tf.ones([1, k], tf.float32))
	x_val_weights = tf.expand_dims(tf.div(top_k_xvals,x_sums_repeated), 1)

	top_k_yvals = tf.gather(y_target_train, top_k_indices)
	prediction = tf.squeeze(tf.matmul(x_val_weights,top_k_yvals), squeeze_dims=[1])

	# Calculate MSE
	mse = tf.div(tf.reduce_sum(tf.square(tf.subtract(prediction, y_target_test))), batch_size)

	# Calculate how many loops over training data
	num_loops = int(np.ceil(len(x_vals_test)/batch_size))

	for i in range(num_loops):
		min_index = i*batch_size
		max_index = min((i+1)*batch_size,len(x_vals_train))
		x_batch = x_vals_test[min_index:max_index]
		y_batch = y_vals_test[min_index:max_index]
		predictions = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch,
											 y_target_train: y_vals_train, y_target_test: y_batch})
		batch_mse = sess.run(mse, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch,
											 y_target_train: y_vals_train, y_target_test: y_batch})

		print('Batch #' + str(i+1) + ' MSE: ' + str(np.round(batch_mse,3)))

	# Plot prediction and actual distribution
	bins = np.linspace(5, 50, 45)

	plt.hist(predictions, bins, alpha=0.5, label='Prediction')
	plt.hist(y_batch, bins, alpha=0.5, label='Actual')
	plt.title('Histogram of Predicted and Actual Values')
	plt.xlabel('Med Home Value in $1,000s')
	plt.ylabel('Frequency')
	plt.legend(loc='upper right')
	plt.show()
	

#--------------------5.2 度量文本距离------------------
# Text Distances
#----------------------------------
#
# This function illustrates how to use
# the Levenstein distance (edit distance)
# in Tensorflow.

def _tf502CalTextd():
	ops.reset_default_graph()
	# Start Graph Session
	sess = tf.Session()

	#----------------------------------
	# First compute the edit distance between 'bear' and 'beers'
	hypothesis = list('bear')
	truth = list('beers')
	
	'''
	表示一个稀疏张量，作为三个独立的稠密张量：indices，values和dense_shape。在Python中，三个张量被集合到一个SparseTensor类中，以方便使用。如果你有单独的indices，values和dense_shape张量，SparseTensor在传递给下面的操作之前，将它们包装在一个对象中。 
	具体来说，该稀疏张量SparseTensor(indices, values, dense_shape)包括以下组件，其中N和ndims分别是在SparseTensor中的值的数目和维度的数量：
	indices：density_shape[N, ndims]的2-D int64张量，指定稀疏张量中包含非零值（元素为零索引）的元素的索引。例如，indices=[[1,3], [2,4]]指定索引为[1,3]和[2,4]的元素具有非零值。
	values：任何类型和dense_shape [N]的一维张量，它提供了indices中的每个元素的值。例如，给定indices=[[1,3], [2,4]]的参数values=[18, 3.6]指定稀疏张量的元素[1,3]的值为18，张量的元素[2,4]的值为3.6。
	dense_shape：density_shape[ndims]的一个1-D int64张量，指定稀疏张量的dense_shape。获取一个列表，指出每个维度中元素的数量。例如，dense_shape=[3,6]指定二维3x6张量，dense_shape=[2,3,4]指定三维2x3x4张量，并且dense_shape=[9]指定具有9个元素的一维张量。
	
	举例：
	tf.SparseTensor(values=['en','fr','zh'],indices=[[0,0],[0,1],[2,0]],shape=[3,2])
	1、shape=[3,2] ，生成3*2的数组[[0,0],[0,0],[0,0]]    
	2、indices=[[0,0],[0,1],[2,0]]，说明上面数组的这些下标的位置是有值的
	3、values=['en','fr','zh']，那么这些不为0的位置的值是多少呢？就是与values给的这些值相对应，最终表示的矩阵为:
	[['en' 'fr'] 
	['0' '0'] 
	['zh' '0']]
	'''
	h1 = tf.SparseTensor([[0,0,0], [0,0,1], [0,0,2], [0,0,3]],
						 hypothesis,
						 [1,1,1])

	t1 = tf.SparseTensor([[0,0,0], [0,0,1], [0,0,2], [0,0,3],[0,0,4]],
						 truth,
						 [1,1,1])

	print(sess.run(tf.edit_distance(h1, t1, normalize=False)))
	#结果为2
	#----------------------------------
	# Compute the edit distance between ('bear','beer') and 'beers':
	hypothesis2 = list('bearbeer')
	truth2 = list('beersbeers')
	h2 = tf.SparseTensor([[0,0,0], [0,0,1], [0,0,2], [0,0,3], [0,1,0], [0,1,1], [0,1,2], [0,1,3]],
						 hypothesis2,
						 [1,2,4])

	t2 = tf.SparseTensor([[0,0,0], [0,0,1], [0,0,2], [0,0,3], [0,0,4], [0,1,0], [0,1,1], [0,1,2], [0,1,3], [0,1,4]],
						 truth2,
						 [1,2,5])

	print(sess.run(tf.edit_distance(h2, t2, normalize=True)))

	#----------------------------------
	# Now compute distance between four words and 'beers' more efficiently:
	hypothesis_words = ['bear','bar','tensor','flow']
	truth_word = ['beers']

	num_h_words = len(hypothesis_words)
	h_indices = [[xi, 0, yi] for xi,x in enumerate(hypothesis_words) for yi,y in enumerate(x)]
	h_chars = list(''.join(hypothesis_words))

	h3 = tf.SparseTensor(h_indices, h_chars, [num_h_words,1,1])

	truth_word_vec = truth_word*num_h_words
	t_indices = [[xi, 0, yi] for xi,x in enumerate(truth_word_vec) for yi,y in enumerate(x)]
	t_chars = list(''.join(truth_word_vec))

	t3 = tf.SparseTensor(t_indices, t_chars, [num_h_words,1,1])

	print(sess.run(tf.edit_distance(h3, t3, normalize=True)))

	#----------------------------------
	# Now we show how to use sparse tensors in a feed dictionary

	# Create input data
	hypothesis_words = ['bear','bar','tensor','flow']
	truth_word = ['beers']

	def create_sparse_vec(word_list):
		num_words = len(word_list)
		indices = [[xi, 0, yi] for xi,x in enumerate(word_list) for yi,y in enumerate(x)]
		chars = list(''.join(word_list))
		return(tf.SparseTensorValue(indices, chars, [num_words,1,1]))

	hyp_string_sparse = create_sparse_vec(hypothesis_words)
	truth_string_sparse = create_sparse_vec(truth_word*len(hypothesis_words))

	hyp_input = tf.sparse_placeholder(dtype=tf.string)
	truth_input = tf.sparse_placeholder(dtype=tf.string)

	edit_distances = tf.edit_distance(hyp_input, truth_input, normalize=True)

	feed_dict = {hyp_input: hyp_string_sparse,
				 truth_input: truth_string_sparse}
				 
	print(sess.run(edit_distances, feed_dict=feed_dict))



#--------------------5.3 混合距离计算------------------
# Mixed Distance Functions for  k-Nearest Neighbor
#----------------------------------
#
# This function shows how to use different distance
# metrics on different features for kNN.
#
# Data:
#----------x-values-----------
# CRIM   : per capita crime rate by town
# ZN     : prop. of res. land zones
# INDUS  : prop. of non-retail business acres
# CHAS   : Charles river dummy variable
# NOX    : nitrix oxides concentration / 10 M
# RM     : Avg. # of rooms per building
# AGE    : prop. of buildings built prior to 1940
# DIS    : Weighted distances to employment centers
# RAD    : Index of radian highway access
# TAX    : Full tax rate value per $10k
# PTRATIO: Pupil/Teacher ratio by town
# B      : 1000*(Bk-0.63)^2, Bk=prop. of blacks
# LSTAT  : % lower status of pop
#------------y-value-----------
# MEDV   : Median Value of homes in $1,000's

def _tf503CalMix():
	ops.reset_default_graph()

	# Create graph
	sess = tf.Session()

	# Load the data
	#housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
	housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
	cols_used = ['CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']
	num_features = len(cols_used)
	#housing_file = requests.get(housing_url)
	#housing_data = [[float(x) for x in y.split(' ') if len(x)>=1] for y in housing_file.text.split('\n') if len(y)>=1]
	housing_file=pd.read_csv('datahousing.csv')
	housing_data=np.array(housing_file).tolist()
	y_vals = np.transpose([np.array([y[13] for y in housing_data])])
	x_vals = np.array([[x for i,x in enumerate(y) if housing_header[i] in cols_used] for y in housing_data])

	## Min-Max Scaling
	x_vals = (x_vals - x_vals.min(0)) / x_vals.ptp(0)

	## Create distance metric weight matrix weighted by standard deviation
	weight_diagonal = x_vals.std(0)
	weight_matrix = tf.cast(tf.diag(weight_diagonal), dtype=tf.float32)

	# Split the data into train and test sets
	train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
	test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
	x_vals_train = x_vals[train_indices]
	x_vals_test = x_vals[test_indices]
	y_vals_train = y_vals[train_indices]
	y_vals_test = y_vals[test_indices]

	# Declare k-value and batch size
	k = 4
	batch_size=len(x_vals_test)

	# Placeholders
	x_data_train = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
	x_data_test = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
	y_target_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
	y_target_test = tf.placeholder(shape=[None, 1], dtype=tf.float32)

	# Declare weighted distance metric
	# Weighted - L2 = sqrt((x-y)^T * A * (x-y))
	subtraction_term =  tf.subtract(x_data_train, tf.expand_dims(x_data_test,1))
	first_product = tf.matmul(subtraction_term, tf.tile(tf.expand_dims(weight_matrix,0), [batch_size,1,1]))
	second_product = tf.matmul(first_product, tf.transpose(subtraction_term, perm=[0,2,1]))
	distance = tf.sqrt(tf.matrix_diag(second_product))

	# Predict: Get min distance index (Nearest neighbor)
	top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
	x_sums = tf.expand_dims(tf.reduce_sum(top_k_xvals, 1),1)
	x_sums_repeated = tf.matmul(x_sums,tf.ones([1, k], tf.float32))
	x_val_weights = tf.expand_dims(tf.div(top_k_xvals,x_sums_repeated), 1)

	top_k_yvals = tf.gather(y_target_train, top_k_indices)
	prediction = tf.squeeze(tf.matmul(x_val_weights,top_k_yvals), squeeze_dims=[1])

	# Calculate MSE
	mse = tf.div(tf.reduce_sum(tf.square(tf.subtract(prediction, y_target_test))), batch_size)

	# Calculate how many loops over training data
	num_loops = int(np.ceil(len(x_vals_test)/batch_size))

	for i in range(num_loops):
		min_index = i*batch_size
		max_index = min((i+1)*batch_size,len(x_vals_train))
		x_batch = x_vals_test[min_index:max_index]
		y_batch = y_vals_test[min_index:max_index]
		predictions = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch,
											 y_target_train: y_vals_train, y_target_test: y_batch})
		batch_mse = sess.run(mse, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch,
											 y_target_train: y_vals_train, y_target_test: y_batch})

		print('Batch #' + str(i+1) + ' MSE: ' + str(np.round(batch_mse,3)))

	# Plot prediction and actual distribution
	bins = np.linspace(5, 50, 45)

	plt.hist(predictions, bins, alpha=0.5, label='Prediction')
	plt.hist(y_batch, bins, alpha=0.5, label='Actual')
	plt.title('Histogram of Predicted and Actual Values')
	plt.xlabel('Med Home Value in $1,000s')
	plt.ylabel('Frequency')
	plt.legend(loc='upper right')
	plt.show()

#--------------------5.4 地址匹配------------------
# 度量既包含文本特征，又包含数值特征的数据观测点距离
#----------------------------------
#
# Address Matching with k-Nearest Neighbors
#----------------------------------
#
# This function illustrates a way to perform
# address matching between two data sets.
#
# For each test address, we will return the
# closest reference address to it.
#
# We will consider two distance functions:
# 1) Edit distance for street number/name and
# 2) Euclidian distance (L2) for the zip codes
import random
import string
def _tf504CalMix():
	# First we generate the data sets we will need
	# n = Size of created data sets
	n = 10
	street_names = ['abbey', 'baker', 'canal', 'donner', 'elm']
	street_types = ['rd', 'st', 'ln', 'pass', 'ave']
	rand_zips = [random.randint(65000,65999) for i in range(5)]

	# Function to randomly create one typo in a string w/ a probability
	def create_typo(s, prob=0.75):
		if random.uniform(0,1) < prob:
			rand_ind = random.choice(range(len(s)))
			s_list = list(s)
			s_list[rand_ind]=random.choice(string.ascii_lowercase)
			s = ''.join(s_list)
		return(s)

	# Generate the reference dataset
	numbers = [random.randint(1, 9999) for i in range(n)]
	streets = [random.choice(street_names) for i in range(n)]
	street_suffs = [random.choice(street_types) for i in range(n)]
	zips = [random.choice(rand_zips) for i in range(n)]
	full_streets = [str(x) + ' ' + y + ' ' + z for x,y,z in zip(numbers, streets, street_suffs)]
	reference_data = [list(x) for x in zip(full_streets,zips)]

	# Generate test dataset with some typos
	typo_streets = [create_typo(x) for x in streets]
	typo_full_streets = [str(x) + ' ' + y + ' ' + z for x,y,z in zip(numbers, typo_streets, street_suffs)]
	test_data = [list(x) for x in zip(typo_full_streets,zips)]

	# Now we can perform address matching
	# Create graph
	sess = tf.Session()

	# Placeholders
	test_address = tf.sparse_placeholder( dtype=tf.string)
	test_zip = tf.placeholder(shape=[None, 1], dtype=tf.float32)
	ref_address = tf.sparse_placeholder(dtype=tf.string)
	ref_zip = tf.placeholder(shape=[None, n], dtype=tf.float32)

	# Declare Zip code distance for a test zip and reference set
	zip_dist = tf.square(tf.subtract(ref_zip, test_zip))

	# Declare Edit distance for address
	address_dist = tf.edit_distance(test_address, ref_address, normalize=True)

	# Create similarity scores
	zip_max = tf.gather(tf.squeeze(zip_dist), tf.argmax(zip_dist, 1))
	zip_min = tf.gather(tf.squeeze(zip_dist), tf.argmin(zip_dist, 1))
	zip_sim = tf.div(tf.subtract(zip_max, zip_dist), tf.subtract(zip_max, zip_min))
	address_sim = tf.subtract(1., address_dist)

	# Combine distance functions
	address_weight = 0.5
	zip_weight = 1. - address_weight
	weighted_sim = tf.add(tf.transpose(tf.multiply(address_weight, address_sim)), tf.multiply(zip_weight, zip_sim))

	# Predict: Get max similarity entry
	top_match_index = tf.argmax(weighted_sim, 1)


	# Function to Create a character-sparse tensor from strings
	def sparse_from_word_vec(word_vec):
		num_words = len(word_vec)
		indices = [[xi, 0, yi] for xi,x in enumerate(word_vec) for yi,y in enumerate(x)]
		chars = list(''.join(word_vec))
		return(tf.SparseTensorValue(indices, chars, [num_words,1,1]))

	# Loop through test indices
	reference_addresses = [x[0] for x in reference_data]
	reference_zips = np.array([[x[1] for x in reference_data]])

	# Create sparse address reference set
	sparse_ref_set = sparse_from_word_vec(reference_addresses)

	for i in range(n):
		test_address_entry = test_data[i][0]
		test_zip_entry = [[test_data[i][1]]]
		
		# Create sparse address vectors
		test_address_repeated = [test_address_entry] * n
		sparse_test_set = sparse_from_word_vec(test_address_repeated)
		
		feeddict={test_address: sparse_test_set,
				   test_zip: test_zip_entry,
				   ref_address: sparse_ref_set,
				   ref_zip: reference_zips}
		best_match = sess.run(top_match_index, feed_dict=feeddict)
		best_street = reference_addresses[best_match[0]]
		[best_zip] = reference_zips[0][best_match]
		[[test_zip_]] = test_zip_entry
		print('Address: ' + str(test_address_entry) + ', ' + str(test_zip_))
		print('Match  : ' + str(best_street) + ', ' + str(best_zip))

#--------------------5.5 实现图像识别------------------
#实现手写字的图形识别
#----------------------------------
# MNIST Digit Prediction with k-Nearest Neighbors
#-----------------------------------------------
#
# This script will load the MNIST data, and split
# it into test/train and perform prediction with
# nearest neighbors
#
# For each test integer, we will return the
# closest image/integer.
#
# Integer images are represented as 28x8 matrices
# of floating point numbers
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

def _tf505CalPic():
	ops.reset_default_graph()

	# Create graph
	sess = tf.Session()

	# Load the data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	# Random sample
	train_size = 1000
	test_size = 102
	rand_train_indices = np.random.choice(len(mnist.train.images), train_size, replace=False)
	rand_test_indices = np.random.choice(len(mnist.test.images), test_size, replace=False)
	x_vals_train = mnist.train.images[rand_train_indices]
	x_vals_test = mnist.test.images[rand_test_indices]
	y_vals_train = mnist.train.labels[rand_train_indices]
	y_vals_test = mnist.test.labels[rand_test_indices]

	# Declare k-value and batch size
	k = 4
	batch_size=6

	# Placeholders
	x_data_train = tf.placeholder(shape=[None, 784], dtype=tf.float32)
	x_data_test = tf.placeholder(shape=[None, 784], dtype=tf.float32)
	y_target_train = tf.placeholder(shape=[None, 10], dtype=tf.float32)
	y_target_test = tf.placeholder(shape=[None, 10], dtype=tf.float32)

	# Declare distance metric
	# L1
	distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1))), reduction_indices=2)

	# L2
	#distance = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(x_data_train, tf.expand_dims(x_data_test,1))), reduction_indices=1))

	# Predict: Get min distance index (Nearest neighbor)
	top_k_xvals, top_k_indices = tf.nn.top_k(tf.neg(distance), k=k)
	prediction_indices = tf.gather(y_target_train, top_k_indices)
	# Predict the mode category
	count_of_predictions = tf.reduce_sum(prediction_indices, reduction_indices=1)
	prediction = tf.argmax(count_of_predictions, dimension=1)

	# Calculate how many loops over training data
	num_loops = int(np.ceil(len(x_vals_test)/batch_size))

	test_output = []
	actual_vals = []
	for i in range(num_loops):
		min_index = i*batch_size
		max_index = min((i+1)*batch_size,len(x_vals_train))
		x_batch = x_vals_test[min_index:max_index]
		y_batch = y_vals_test[min_index:max_index]
		predictions = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch,
											 y_target_train: y_vals_train, y_target_test: y_batch})
		test_output.extend(predictions)
		actual_vals.extend(np.argmax(y_batch, axis=1))

	accuracy = sum([1./test_size for i in range(test_size) if test_output[i]==actual_vals[i]])
	print('Accuracy on test set: ' + str(accuracy))

	# Plot the last batch results:
	actuals = np.argmax(y_batch, axis=1)

	Nrows = 2
	Ncols = 3
	for i in range(len(actuals)):
		plt.subplot(Nrows, Ncols, i+1)
		plt.imshow(np.reshape(x_batch[i], [28,28]), cmap='Greys_r')
		plt.title('Actual: ' + str(actuals[i]) + ' Pred: ' + str(predictions[i]),
								   fontsize=10)
		frame = plt.gca()
		frame.axes.get_xaxis().set_visible(False)
		frame.axes.get_yaxis().set_visible(False)








	
	
#--------------main()-------------

if __name__=='__main__':
	_tf505CalPic()
