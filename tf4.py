import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
#--------------------4.1变量设置------------------
# Get graph handle

def _tf401CalSvm():

	ops.reset_default_graph()
	# Create graph
	sess = tf.Session()

	# Load the data
	# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
	iris = datasets.load_iris()
	x_vals = np.array([[x[0], x[3]] for x in iris.data])
	y_vals = np.array([1 if y==0 else -1 for y in iris.target])

	# Split data into train/test sets
	train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
	test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
	x_vals_train = x_vals[train_indices]
	x_vals_test = x_vals[test_indices]
	y_vals_train = y_vals[train_indices]
	y_vals_test = y_vals[test_indices]

	# Declare batch size
	batch_size = 100

	# Initialize placeholders
	x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
	y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

	# Create variables for linear regression
	A = tf.Variable(tf.random_normal(shape=[2,1]))
	b = tf.Variable(tf.random_normal(shape=[1,1]))

	# Declare model operations
	model_output = tf.subtract(tf.matmul(x_data, A), b)

	# Declare vector L2 'norm' function squared
	l2_norm = tf.reduce_sum(tf.square(A))

	# Declare loss function
	# = max(0, 1-pred*actual) + alpha * L2_norm(A)^2
	# L2 regularization parameter, alpha
	alpha = tf.constant([0.01])
	# Margin term in loss
	classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))
	# Put terms together
	loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))

	# Declare prediction function
	prediction = tf.sign(model_output)
	accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))

	# Declare optimizer
	my_opt = tf.train.GradientDescentOptimizer(0.01)
	train_step = my_opt.minimize(loss)

	# Initialize variables
	init = tf.global_variables_initializer()
	sess.run(init)

	# Training loop
	loss_vec = []
	train_accuracy = []
	test_accuracy = []
	for i in range(500):
		rand_index = np.random.choice(len(x_vals_train), size=batch_size)
		rand_x = x_vals_train[rand_index]
		rand_y = np.transpose([y_vals_train[rand_index]])
		sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
		
		temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
		loss_vec.append(temp_loss)
		
		train_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_train, y_target: np.transpose([y_vals_train])})
		train_accuracy.append(train_acc_temp)
		
		test_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
		test_accuracy.append(test_acc_temp)
		
		if (i+1)%100==0:
			print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
			print('Loss = ' + str(temp_loss))

	# Extract coefficients
	[[a1], [a2]] = sess.run(A)
	[[b]] = sess.run(b)
	slope = -a2/a1
	y_intercept = b/a1

	# Extract x1 and x2 vals
	x1_vals = [d[1] for d in x_vals]

	# Get best fit line
	best_fit = []
	for i in x1_vals:
	  best_fit.append(slope*i+y_intercept)

	# Separate I. setosa
	setosa_x = [d[1] for i,d in enumerate(x_vals) if y_vals[i]==1]
	setosa_y = [d[0] for i,d in enumerate(x_vals) if y_vals[i]==1]
	not_setosa_x = [d[1] for i,d in enumerate(x_vals) if y_vals[i]==-1]
	not_setosa_y = [d[0] for i,d in enumerate(x_vals) if y_vals[i]==-1]

	# Plot data and line
	plt.plot(setosa_x, setosa_y, 'o', label='I. setosa')
	plt.plot(not_setosa_x, not_setosa_y, 'x', label='Non-setosa')
	plt.plot(x1_vals, best_fit, 'r-', label='Linear Separator', linewidth=3)
	plt.ylim([0, 10])
	plt.legend(loc='lower right')
	plt.title('Sepal Length vs Pedal Width')
	plt.xlabel('Pedal Width')
	plt.ylabel('Sepal Length')
	plt.show()

	# Plot train/test accuracies
	plt.plot(train_accuracy, 'k-', label='Training Accuracy')
	plt.plot(test_accuracy, 'r--', label='Test Accuracy')
	plt.title('Train and Test Set Accuracies')
	plt.xlabel('Generation')
	plt.ylabel('Accuracy')
	plt.legend(loc='lower right')
	plt.show()

	# Plot loss over time
	plt.plot(loss_vec, 'k-')
	plt.title('Loss per Generation')
	plt.xlabel('Generation')
	plt.ylabel('Loss')
	plt.show()


#--------------------4.2SVM回归------------------
# Get graph handle

# SVM Regression
#----------------------------------
#
# This function shows how to use Tensorflow to
# solve support vector regression. We are going
# to find the line that has the maximum margin
# which INCLUDES as many points as possible
#
# We will use the iris data, specifically:
#  y = Sepal Length
#  x = Pedal Width
def _tf402CalSvmreg():

	ops.reset_default_graph()

	# Create graph
	sess = tf.Session()

	# Load the data
	# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
	iris = datasets.load_iris()
	x_vals = np.array([x[3] for x in iris.data])
	y_vals = np.array([y[0] for y in iris.data])

	# Split data into train/test sets
	train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
	test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
	x_vals_train = x_vals[train_indices]
	x_vals_test = x_vals[test_indices]
	y_vals_train = y_vals[train_indices]
	y_vals_test = y_vals[test_indices]

	# Declare batch size
	batch_size = 50

	# Initialize placeholders
	x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
	y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

	# Create variables for linear regression
	A = tf.Variable(tf.random_normal(shape=[1,1]))
	b = tf.Variable(tf.random_normal(shape=[1,1]))

	# Declare model operations
	model_output = tf.add(tf.matmul(x_data, A), b)

	# Declare loss function
	# = max(0, abs(target - predicted) + epsilon)
	# 1/2 margin width parameter = epsilon
	epsilon = tf.constant([0.5])
	# Margin term in loss
	loss = tf.reduce_mean(tf.maximum(0., tf.subtract(tf.abs(tf.subtract(model_output, y_target)), epsilon)))

	# Declare optimizer
	my_opt = tf.train.GradientDescentOptimizer(0.075)
	train_step = my_opt.minimize(loss)

	# Initialize variables
	init = tf.global_variables_initializer()
	sess.run(init)

	# Training loop
	train_loss = []
	test_loss = []
	for i in range(200):
		rand_index = np.random.choice(len(x_vals_train), size=batch_size)
		rand_x = np.transpose([x_vals_train[rand_index]])
		rand_y = np.transpose([y_vals_train[rand_index]])
		sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
		
		temp_train_loss = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_train]), y_target: np.transpose([y_vals_train])})
		train_loss.append(temp_train_loss)
		
		temp_test_loss = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_test]), y_target: np.transpose([y_vals_test])})
		test_loss.append(temp_test_loss)
		if (i+1)%50==0:
			print('-----------')
			print('Generation: ' + str(i+1))
			print('A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
			print('Train Loss = ' + str(temp_train_loss))
			print('Test Loss = ' + str(temp_test_loss))

	# Extract Coefficients
	[[slope]] = sess.run(A)
	[[y_intercept]] = sess.run(b)
	[width] = sess.run(epsilon)

	# Get best fit line
	best_fit = []
	best_fit_upper = []
	best_fit_lower = []
	for i in x_vals:
	  best_fit.append(slope*i+y_intercept)
	  best_fit_upper.append(slope*i+y_intercept+width)
	  best_fit_lower.append(slope*i+y_intercept-width)

	# Plot fit with data
	plt.plot(x_vals, y_vals, 'o', label='Data Points')
	plt.plot(x_vals, best_fit, 'r-', label='SVM Regression Line', linewidth=3)
	plt.plot(x_vals, best_fit_upper, 'r--', linewidth=2)
	plt.plot(x_vals, best_fit_lower, 'r--', linewidth=2)
	plt.ylim([0, 10])
	plt.legend(loc='lower right')
	plt.title('Sepal Length vs Pedal Width')
	plt.xlabel('Pedal Width')
	plt.ylabel('Sepal Length')
	plt.show()

	# Plot loss over time
	plt.plot(train_loss, 'k-', label='Train Set Loss')
	plt.plot(test_loss, 'r--', label='Test Set Loss')
	plt.title('L2 Loss per Generation')
	plt.xlabel('Generation')
	plt.ylabel('L2 Loss')
	plt.legend(loc='upper right')
	plt.show()

#--------------------4.3 非线性SVM  采用高斯核函数------------------
# Nonlinear SVM Example
#----------------------------------
#
# This function wll illustrate how to
# implement the gaussian kernel on
# the iris dataset.
#
# Gaussian Kernel:
# K(x1, x2) = exp(-gamma * abs(x1 - x2)^2)
def _tf403CalSvm():
	ops.reset_default_graph()
	# Create graph
	sess = tf.Session()

	# Load the data
	# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
	iris = datasets.load_iris()
	x_vals = np.array([[x[0], x[3]] for x in iris.data])
	y_vals = np.array([1 if y==0 else -1 for y in iris.target])
	class1_x = [x[0] for i,x in enumerate(x_vals) if y_vals[i]==1]
	class1_y = [x[1] for i,x in enumerate(x_vals) if y_vals[i]==1]
	class2_x = [x[0] for i,x in enumerate(x_vals) if y_vals[i]==-1]
	class2_y = [x[1] for i,x in enumerate(x_vals) if y_vals[i]==-1]

	# Declare batch size
	batch_size = 150

	# Initialize placeholders
	x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
	y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
	prediction_grid = tf.placeholder(shape=[None, 2], dtype=tf.float32)

	# Create variables for svm
	b = tf.Variable(tf.random_normal(shape=[1,batch_size]))

	# Gaussian (RBF) kernel
	gamma = tf.constant(-25.0)
	dist = tf.reduce_sum(tf.square(x_data), 1)#
	'''
	tensorflow中有很多在维度上的操作，本例以常用的tf.reduce_sum进行说明。官方给的api
	reduce_sum(
    input_tensor,
    axis=None,
    keep_dims=False,
    name=None,
    reduction_indices=None
	)
	input_tensor:表示输入 
	axis:表示在那个维度进行sum操作。 
	keep_dims:表示是否保留原始数据的维度，False相当于执行完后原始数据就会少一个维度。 
	reduction_indices:为了跟旧版本的兼容，现在已经不使用了。 
	官方的例子：

	# 'x' is [[1, 1, 1]
	#         [1, 1, 1]]
	tf.reduce_sum(x) ==> 6
	tf.reduce_sum(x, 0) ==> [2, 2, 2]
	tf.reduce_sum(x, 1) ==> [3, 3]
	tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
	tf.reduce_sum(x, [0, 1]) ==> 6
	'''
	
	dist = tf.reshape(dist, [-1,1])#转化为N行，1列
	sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))), tf.transpose(dist))
	my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

	# Compute SVM Model
	model_output = tf.matmul(b, my_kernel)
	first_term = tf.reduce_sum(b)
	b_vec_cross = tf.matmul(tf.transpose(b), b)
	y_target_cross = tf.matmul(y_target, tf.transpose(y_target))
	second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)))
	
	#tf.negative(x,name=None) ＃取负运算（ｙ＝－ｘ）
	loss = tf.negative(tf.subtract(first_term, second_term))

	# Gaussian (RBF) prediction kernel
	rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1),[-1,1])
	rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1),[-1,1])
	pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(rB))
	pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

	prediction_output = tf.matmul(tf.multiply(tf.transpose(y_target),b), pred_kernel)
	
	#tf.sign(x,name=None)＃返回符合　ｘ大于０，则返回１，小于０，则返回－１；
	prediction = tf.sign(prediction_output-tf.reduce_mean(prediction_output))
	
	
	'''
	# 'x' is [[1., 2.]
	#         [3., 4.]]
	x是一个2维数组，分别调用reduce_*函数如下：
	首先求平均值：
	tf.reduce_mean(x) ==> 2.5 #如果不指定第二个参数，那么就在所有的元素中取平均值
	tf.reduce_mean(x, 0) ==> [2.,  3.] #指定第二个参数为0，则第一维的元素取平均值，即每一列求平均值
	tf.reduce_mean(x, 1) ==> [1.5,  3.5] #
	指定第二个参数为1，则第二维的元素取平均值，即每一行求平均值
	'''
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(y_target)), tf.float32))
	'''
	tf.squeeze()

	Function

	tf.squeeze(input, squeeze_dims=None, name=None)

	Removes dimensions of size 1 from the shape of a tensor. 
	从tensor中删除所有大小是1的维度

	Given a tensor input, this operation returns a tensor of the same type with all dimensions of size 1 removed. If you don’t want to remove all size 1 dimensions, you can remove specific size 1 dimensions by specifying squeeze_dims. 

	给定张量输入，此操作返回相同类型的张量，并删除所有尺寸为1的尺寸。 如果不想删除所有尺寸1尺寸，可以通过指定squeeze_dims来删除特定尺寸1尺寸。
	如果不想删除所有大小是1的维度，可以通过squeeze_dims指定。

	For example:
	# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
	shape(squeeze(t)) ==> [2, 3]
	Or, to remove specific size 1 dimensions:

	# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
	shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
	'''
	
	# Declare optimizer
	my_opt = tf.train.GradientDescentOptimizer(0.01)
	train_step = my_opt.minimize(loss)

	# Initialize variables
	init = tf.global_variables_initializer()
	sess.run(init)

	# Training loop
	loss_vec = []
	batch_accuracy = []
	for i in range(300):
		rand_index = np.random.choice(len(x_vals), size=batch_size)
		rand_x = x_vals[rand_index]
		rand_y = np.transpose([y_vals[rand_index]])
		sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
		
		temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
		loss_vec.append(temp_loss)
		
		acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x,
												 y_target: rand_y,
												 prediction_grid:rand_x})
		batch_accuracy.append(acc_temp)
		
		if (i+1)%75==0:
			print('Step #' + str(i+1))
			print('Loss = ' + str(temp_loss))

	# Create a mesh to plot points in
	x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
	y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
						 np.arange(y_min, y_max, 0.02))
	grid_points = np.c_[xx.ravel(), yy.ravel()]
	[grid_predictions] = sess.run(prediction, feed_dict={x_data: rand_x,
													   y_target: rand_y,
													   prediction_grid: grid_points})
	grid_predictions = grid_predictions.reshape(xx.shape)

	# Plot points and grid
	plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha=0.8)
	plt.plot(class1_x, class1_y, 'ro', label='I. setosa')
	plt.plot(class2_x, class2_y, 'kx', label='Non setosa')
	plt.title('Gaussian SVM Results on Iris Data')
	plt.xlabel('Pedal Length')
	plt.ylabel('Sepal Width')
	plt.legend(loc='lower right')
	plt.ylim([-0.5, 3.0])
	plt.xlim([3.5, 8.5])
	plt.show()

	# Plot batch accuracy
	plt.plot(batch_accuracy, 'k-', label='Accuracy')
	plt.title('Batch Accuracy')
	plt.xlabel('Generation')
	plt.ylabel('Accuracy')
	plt.legend(loc='lower right')
	plt.show()

	# Plot loss over time
	plt.plot(loss_vec, 'k-')
	plt.title('Loss per Generation')
	plt.xlabel('Generation')
	plt.ylabel('Loss')
	plt.show()


#--------------------4.4SVM核函数------------------
# Illustration of Various Kernels
#----------------------------------
#
# This function wll illustrate how to
# implement various kernels in Tensorflow.
#
# Linear Kernel:
# K(x1, x2) = t(x1) * x2
#
# Gaussian Kernel (RBF):
# K(x1, x2) = exp(-gamma * abs(x1 - x2)^2)

def _tf404CalSvmKernel():
	ops.reset_default_graph()

	# Create graph
	sess = tf.Session()

	# Generate non-lnear data
	(x_vals, y_vals) = datasets.make_circles(n_samples=350, factor=.5, noise=.1)
	y_vals = np.array([1 if y==1 else -1 for y in y_vals])
	class1_x = [x[0] for i,x in enumerate(x_vals) if y_vals[i]==1]
	class1_y = [x[1] for i,x in enumerate(x_vals) if y_vals[i]==1]
	class2_x = [x[0] for i,x in enumerate(x_vals) if y_vals[i]==-1]
	class2_y = [x[1] for i,x in enumerate(x_vals) if y_vals[i]==-1]

	# Declare batch size
	batch_size = 350

	# Initialize placeholders
	x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
	y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
	prediction_grid = tf.placeholder(shape=[None, 2], dtype=tf.float32)

	# Create variables for svm
	b = tf.Variable(tf.random_normal(shape=[1,batch_size]))

	# Apply kernel
	# Linear Kernel
	# my_kernel = tf.matmul(x_data, tf.transpose(x_data))

	# Gaussian (RBF) kernel
	gamma = tf.constant(-50.0)
	dist = tf.reduce_sum(tf.square(x_data), 1)
	dist = tf.reshape(dist, [-1,1])
	sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))), tf.transpose(dist))
	my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

	# Compute SVM Model
	model_output = tf.matmul(b, my_kernel)
	first_term = tf.reduce_sum(b)
	b_vec_cross = tf.matmul(tf.transpose(b), b)
	y_target_cross = tf.matmul(y_target, tf.transpose(y_target))
	second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)))
	loss = tf.negative(tf.subtract(first_term, second_term))

	# Create Prediction Kernel
	# Linear prediction kernel
	# my_kernel = tf.matmul(x_data, tf.transpose(prediction_grid))

	# Gaussian (RBF) prediction kernel
	rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1),[-1,1])
	rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1),[-1,1])
	pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(rB))
	pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

	prediction_output = tf.matmul(tf.multiply(tf.transpose(y_target),b), pred_kernel)
	prediction = tf.sign(prediction_output-tf.reduce_mean(prediction_output))
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(y_target)), tf.float32))

	# Declare optimizer
	my_opt = tf.train.GradientDescentOptimizer(0.002)
	train_step = my_opt.minimize(loss)

	# Initialize variables
	init = tf.global_variables_initializer()
	sess.run(init)

	# Training loop
	loss_vec = []
	batch_accuracy = []
	for i in range(1000):
		rand_index = np.random.choice(len(x_vals), size=batch_size)
		rand_x = x_vals[rand_index]
		rand_y = np.transpose([y_vals[rand_index]])
		sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
		
		temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
		loss_vec.append(temp_loss)
		
		acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x,
												 y_target: rand_y,
												 prediction_grid:rand_x})
		batch_accuracy.append(acc_temp)
		
		if (i+1)%250==0:
			print('Step #' + str(i+1))
			print('Loss = ' + str(temp_loss))

	# Create a mesh to plot points in
	x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
	y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
						 np.arange(y_min, y_max, 0.02))
	grid_points = np.c_[xx.ravel(), yy.ravel()]
	[grid_predictions] = sess.run(prediction, feed_dict={x_data: rand_x,
													   y_target: rand_y,
													   prediction_grid: grid_points})
	grid_predictions = grid_predictions.reshape(xx.shape)

	# Plot points and grid
	plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha=0.8)
	plt.plot(class1_x, class1_y, 'ro', label='Class 1')
	plt.plot(class2_x, class2_y, 'kx', label='Class -1')
	plt.title('Gaussian SVM Results')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.legend(loc='lower right')
	plt.ylim([-1.5, 1.5])
	plt.xlim([-1.5, 1.5])
	plt.show()

	# Plot batch accuracy
	plt.plot(batch_accuracy, 'k-', label='Accuracy')
	plt.title('Batch Accuracy')
	plt.xlabel('Generation')
	plt.ylabel('Accuracy')
	plt.legend(loc='lower right')
	plt.show()

	# Plot loss over time
	plt.plot(loss_vec, 'k-')
	plt.title('Loss per Generation')
	plt.xlabel('Generation')
	plt.ylabel('Loss')
	plt.show()



#--------------------4.5 多分类------------------
# Multi-class (Nonlinear) SVM Example
#----------------------------------
#
# This function wll illustrate how to
# implement the gaussian kernel with
# multiple classes on the iris dataset.
#
# Gaussian Kernel:
# K(x1, x2) = exp(-gamma * abs(x1 - x2)^2)
#
# X : (Sepal Length, Petal Width)
# Y: (I. setosa, I. virginica, I. versicolor) (3 classes)
#
# Basic idea: introduce an extra dimension to do
# one vs all classification.
#
# The prediction of a point will be the category with
# the largest margin or distance to boundary.

def _tf405CalSvmMulti():
	ops.reset_default_graph()

	# Create graph
	sess = tf.Session()

	# Load the data
	# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
	iris = datasets.load_iris()
	x_vals = np.array([[x[0], x[3]] for x in iris.data])
	y_vals1 = np.array([1 if y==0 else -1 for y in iris.target])
	y_vals2 = np.array([1 if y==1 else -1 for y in iris.target])
	y_vals3 = np.array([1 if y==2 else -1 for y in iris.target])
	y_vals = np.array([y_vals1, y_vals2, y_vals3])
	class1_x = [x[0] for i,x in enumerate(x_vals) if iris.target[i]==0]
	class1_y = [x[1] for i,x in enumerate(x_vals) if iris.target[i]==0]
	class2_x = [x[0] for i,x in enumerate(x_vals) if iris.target[i]==1]
	class2_y = [x[1] for i,x in enumerate(x_vals) if iris.target[i]==1]
	class3_x = [x[0] for i,x in enumerate(x_vals) if iris.target[i]==2]
	class3_y = [x[1] for i,x in enumerate(x_vals) if iris.target[i]==2]

	# Declare batch size
	batch_size = 50

	# Initialize placeholders
	x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
	y_target = tf.placeholder(shape=[3, None], dtype=tf.float32)
	prediction_grid = tf.placeholder(shape=[None, 2], dtype=tf.float32)

	# Create variables for svm
	b = tf.Variable(tf.random_normal(shape=[3,batch_size]))

	# Gaussian (RBF) kernel
	gamma = tf.constant(-10.0)
	dist = tf.reduce_sum(tf.square(x_data), 1)
	dist = tf.reshape(dist, [-1,1])
	sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))), tf.transpose(dist))
	my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

	# Declare function to do reshape/batch multiplication
	def reshape_matmul(mat):
		v1 = tf.expand_dims(mat, 1)
		v2 = tf.reshape(v1, [3, batch_size, 1])
		return(tf.matmul(v2, v1))

	# Compute SVM Model
	model_output = tf.matmul(b, my_kernel)
	first_term = tf.reduce_sum(b)
	b_vec_cross = tf.matmul(tf.transpose(b), b)
	y_target_cross = reshape_matmul(y_target)

	second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)),[1,2])
	loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))

	# Gaussian (RBF) prediction kernel
	rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1),[-1,1])
	rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1),[-1,1])
	pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(rB))
	pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

	prediction_output = tf.matmul(tf.multiply(y_target,b), pred_kernel)
	prediction = tf.argmax(prediction_output-tf.expand_dims(tf.reduce_mean(prediction_output,1), 1), 0)
	accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_target,0)), tf.float32))

	# Declare optimizer
	my_opt = tf.train.GradientDescentOptimizer(0.01)
	train_step = my_opt.minimize(loss)

	# Initialize variables
	init = tf.global_variables_initializer()
	sess.run(init)

	# Training loop
	loss_vec = []
	batch_accuracy = []
	for i in range(100):
		rand_index = np.random.choice(len(x_vals), size=batch_size)
		rand_x = x_vals[rand_index]
		rand_y = y_vals[:,rand_index]
		sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
		
		temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
		loss_vec.append(temp_loss)
		
		acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x,
												 y_target: rand_y,
												 prediction_grid:rand_x})
		batch_accuracy.append(acc_temp)
		
		if (i+1)%25==0:
			print('Step #' + str(i+1))
			print('Loss = ' + str(temp_loss))

	# Create a mesh to plot points in
	x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
	y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
						 np.arange(y_min, y_max, 0.02))
	grid_points = np.c_[xx.ravel(), yy.ravel()]
	grid_predictions = sess.run(prediction, feed_dict={x_data: rand_x,
													   y_target: rand_y,
													   prediction_grid: grid_points})
	grid_predictions = grid_predictions.reshape(xx.shape)

	# Plot points and grid
	plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha=0.8)
	plt.plot(class1_x, class1_y, 'ro', label='I. setosa')
	plt.plot(class2_x, class2_y, 'kx', label='I. versicolor')
	plt.plot(class3_x, class3_y, 'gv', label='I. virginica')
	plt.title('Gaussian SVM Results on Iris Data')
	plt.xlabel('Pedal Length')
	plt.ylabel('Sepal Width')
	plt.legend(loc='lower right')
	plt.ylim([-0.5, 3.0])
	plt.xlim([3.5, 8.5])
	plt.show()

	# Plot batch accuracy
	plt.plot(batch_accuracy, 'k-', label='Accuracy')
	plt.title('Batch Accuracy')
	plt.xlabel('Generation')
	plt.ylabel('Accuracy')
	plt.legend(loc='lower right')
	plt.show()

	# Plot loss over time
	plt.plot(loss_vec, 'k-')
	plt.title('Loss per Generation')
	plt.xlabel('Generation')
	plt.ylabel('Loss')
	plt.show()






























	
	
#--------------main()-------------

if __name__=='__main__':
	_tf405CalSvmMulti()
