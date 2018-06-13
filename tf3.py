# Tensors 线性回归
'''
tensorflow求逆矩阵
tensorflow实现矩阵分解
tensorflow实现线性回归
tensorflow损失函数
tensorflow实现戴明回归（Deming Regression）
tensorflow实现Lasso回归和岭回归（Ridge Regression）
tensorflow实现弹性回归（Elastic NEt Regression）
tensorflow实现逻辑回归
'''

import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()
import matplotlib.pyplot as plt
import numpy as np
# Introduce tensors in tf
#--------------------3.1逆矩阵 方程求解------------------
#求公式  A*x+b=Y 的A和b值
# Linear Regression: Inverse Matrix Method
#----------------------------------
#
# This function shows how to use Tensorflow to
# solve linear regression via the matrix inverse.
#
# Given Ax=b, solving for x:
#  x = (t(A) * A)^(-1) * t(A) * b
#  where t(A) is the transpose of A
def _tf301Calinverse():
	# Create graph
	sess = tf.Session()
		
	# Create the data
	x_vals = np.linspace(0, 10, 100)
	y_vals = x_vals + np.random.normal(0, 1, 100)

	# Create design matrix
	x_vals_column = np.transpose(np.matrix(x_vals))#矩阵转置
	ones_column = np.transpose(np.matrix(np.repeat(1, 100)))
	A = np.column_stack((x_vals_column, ones_column))#矩阵横向拼接

	# Create b matrix
	b = np.transpose(np.matrix(y_vals))

	# Create tensors
	A_tensor = tf.constant(A)
	b_tensor = tf.constant(b)

	# Matrix inverse solution
	tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)#A转置后乘以自身，生成2*2矩阵
	tA_A_inv = tf.matrix_inverse(tA_A)#矩阵求逆
	product = tf.matmul(tA_A_inv, tf.transpose(A_tensor))#矩阵相乘
	solution = tf.matmul(product, b_tensor)

	solution_eval = sess.run(solution)

	# Extract coefficients
	slope = solution_eval[0][0]
	y_intercept = solution_eval[1][0]

	print('slope: ' + str(slope))
	print('y_intercept: ' + str(y_intercept))

	# Get best fit line
	best_fit = []
	for i in x_vals:
	  best_fit.append(slope*i+y_intercept)

	# Plot the results
	plt.plot(x_vals, y_vals, 'o', label='Data')
	plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
	plt.legend(loc='upper left')
	plt.show()

#--------------------3.2逆矩阵解决回归问题------------------
#对比L1正则损失函数和L2正则损失函数区别
# This function shows how to use Tensorflow to
# solve linear regression via the matrix inverse.
from sklearn import datasets
def _tf302Calreg():	
	ops.reset_default_graph()
	# Create graph
	sess = tf.Session()

	# Load the data
	# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
	iris = datasets.load_iris()
	x_vals = np.array([x[3] for x in iris.data])
	y_vals = np.array([y[0] for y in iris.data])

	# Declare batch size and number of iterations
	batch_size = 25
	learning_rate = 0.1 # Will not converge with learning rate at 0.4
	iterations = 50 #定义学习次数

	# Initialize placeholders
	x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
	y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

	# Create variables for linear regression
	A = tf.Variable(tf.random_normal(shape=[1,1]))
	b = tf.Variable(tf.random_normal(shape=[1,1]))

	# Declare model operations  定义操作公式
	model_output = tf.add(tf.matmul(x_data, A), b)

	# Declare loss functions  定义L1正则损失函数
	loss_l1 = tf.reduce_mean(tf.abs(y_target - model_output))

	# Initialize variables
	init = tf.global_variables_initializer()
	sess.run(init)

	# Declare optimizers  定义学习方法
	my_opt_l1 = tf.train.GradientDescentOptimizer(learning_rate)
	train_step_l1 = my_opt_l1.minimize(loss_l1)

	# Training loop
	loss_vec_l1 = []
	for i in range(iterations):
		rand_index = np.random.choice(len(x_vals), size=batch_size)
		rand_x = np.transpose([x_vals[rand_index]])
		rand_y = np.transpose([y_vals[rand_index]])
		sess.run(train_step_l1, feed_dict={x_data: rand_x, y_target: rand_y})
		temp_loss_l1 = sess.run(loss_l1, feed_dict={x_data: rand_x, y_target: rand_y})
		loss_vec_l1.append(temp_loss_l1)
		if (i+1)%25==0:
			print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))


	# L2 Loss
	# Reinitialize graph
	ops.reset_default_graph()

	# Create graph
	sess = tf.Session()

	# Initialize placeholders
	x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
	y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

	# Create variables for linear regression
	A = tf.Variable(tf.random_normal(shape=[1,1]))
	b = tf.Variable(tf.random_normal(shape=[1,1]))

	# Declare model operations
	model_output = tf.add(tf.matmul(x_data, A), b)

	# Declare loss functions  定义损失函数为L2
	loss_l2 = tf.reduce_mean(tf.square(y_target - model_output))

	# Initialize variables
	init = tf.global_variables_initializer()
	sess.run(init)

	# Declare optimizers
	my_opt_l2 = tf.train.GradientDescentOptimizer(learning_rate)
	train_step_l2 = my_opt_l2.minimize(loss_l2)

	loss_vec_l2 = []
	for i in range(iterations):
		rand_index = np.random.choice(len(x_vals), size=batch_size)
		rand_x = np.transpose([x_vals[rand_index]])
		rand_y = np.transpose([y_vals[rand_index]])
		sess.run(train_step_l2, feed_dict={x_data: rand_x, y_target: rand_y})
		temp_loss_l2 = sess.run(loss_l2, feed_dict={x_data: rand_x, y_target: rand_y})
		loss_vec_l2.append(temp_loss_l2)
		if (i+1)%25==0:
			print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))


	# Plot loss over time
	plt.plot(loss_vec_l1, 'k-', label='L1 Loss')
	plt.plot(loss_vec_l2, 'r--', label='L2 Loss')
	plt.title('L1 and L2 Loss per Generation')
	plt.xlabel('Generation')
	plt.ylabel('L1 Loss')
	plt.legend(loc='upper right')
	plt.show()
	
#--------------------3.3解决回归问题------------------
# Linear Regression: Tensorflow Way
#----------------------------------
#
# This function shows how to use Tensorflow to
# solve linear regression.
# y = Ax + b
#
# We will use the iris data, specifically:
#  y = Sepal Length
#  x = Petal Width	
def _tf303Calreg():		
	ops.reset_default_graph()

	# Create graph
	sess = tf.Session()

	# Load the data
	# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
	iris = datasets.load_iris()
	x_vals = np.array([x[3] for x in iris.data])
	y_vals = np.array([y[0] for y in iris.data])

	# Declare batch size
	batch_size = 25

	# Initialize placeholders
	x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
	y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

	# Create variables for linear regression
	A = tf.Variable(tf.random_normal(shape=[1,1]))
	b = tf.Variable(tf.random_normal(shape=[1,1]))

	# Declare model operations
	model_output = tf.add(tf.matmul(x_data, A), b)

	# Declare loss function (L2 loss) L2损失函数
	loss = tf.reduce_mean(tf.square(y_target - model_output))

	# Initialize variables
	init = tf.global_variables_initializer()
	sess.run(init)

	# Declare optimizer  设置学习速率为0.05
	my_opt = tf.train.GradientDescentOptimizer(0.05)
	train_step = my_opt.minimize(loss)

	# Training loop
	loss_vec = []
	for i in range(100):
		rand_index = np.random.choice(len(x_vals), size=batch_size)
		rand_x = np.transpose([x_vals[rand_index]])
		rand_y = np.transpose([y_vals[rand_index]])
		sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
		temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
		loss_vec.append(temp_loss)
		if (i+1)%25==0:
			print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
			print('Loss = ' + str(temp_loss))

	# Get the optimal coefficients
	[slope] = sess.run(A)
	[y_intercept] = sess.run(b)

	# Get best fit line
	best_fit = []
	for i in x_vals:
	  best_fit.append(slope*i+y_intercept)

	# Plot the result
	plt.plot(x_vals, y_vals, 'o', label='Data Points')
	plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
	plt.legend(loc='upper left')
	plt.title('Sepal Length vs Pedal Width')
	plt.xlabel('Pedal Width')
	plt.ylabel('Sepal Length')
	plt.show()

	# Plot loss over time
	plt.plot(loss_vec, 'k-')
	plt.title('L2 Loss per Generation')
	plt.xlabel('Generation')
	plt.ylabel('L2 Loss')
	plt.show()
	


#-----------------3.4 矩阵方式解决戴明回归问题---------------
# Linear Regression: Decomposition Method
#----------------------------------
#如果最小二乘线性回归使到线的垂直距离最小，则Deming回归使到线的总距离最小化。 这种类型的回归使y值和x值的误差最小化。
# This function shows how to use Tensorflow to
# solve linear regression via the matrix inverse.
#
# Given Ax=b, and a Cholesky decomposition such that
#  A = L*L' then we can get solve for x via
# 1) L*y=t(A)*b
# 2) L'*x=y

def _tf304CalDMreg():	
	ops.reset_default_graph()

	# Create graph
	sess = tf.Session()

	# Create the data
	x_vals = np.linspace(0, 10, 100)
	y_vals = x_vals + np.random.normal(0, 1, 100)

	# Create design matrix
	x_vals_column = np.transpose(np.matrix(x_vals))
	ones_column = np.transpose(np.matrix(np.repeat(1, 100)))
	A = np.column_stack((x_vals_column, ones_column))

	# Create b matrix
	b = np.transpose(np.matrix(y_vals))

	# Create tensors
	A_tensor = tf.constant(A)
	b_tensor = tf.constant(b)

	# Find Cholesky Decomposition
	tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
	# TensorFlow的cholesky（）函数仅仅返回矩阵分解的下三角矩阵，
	# 因为上三角矩阵是下三角矩阵的转置矩阵。
	#实数界来类比的话，此分解就好像求平方根
	#与一般的矩阵分解求解方程的方法比较，Cholesky分解效率很高
	L = tf.cholesky(tA_A)

	# Solve L*y=t(A)*b
	tA_b = tf.matmul(tf.transpose(A_tensor), b)
	sol1 = tf.matrix_solve(L, tA_b)
	'''
	设x1=2, x2=3，列方程组：  3x + 2y = 12  x + y = 5  得到系数矩阵[[3.,2.],[1.,1.]]和值矩阵[[12.],[5.]]。 
	import tensorflow as tf  
	sess = tf.InteractiveSession()  
	a = tf.constant([[3.,2.],  [1.,1.]]) 
	print(tf.matrix_solve(a, [[12.],[5.]]).eval())  走起，得到输出：  [[ 1.99999988]  [ 3.00000024]]  这不就是x1=2和x2=3嘛
	'''
	
	# Solve L' * y = sol1
	sol2 = tf.matrix_solve(tf.transpose(L), sol1)

	solution_eval = sess.run(sol2)

	# Extract coefficients
	slope = solution_eval[0][0]
	y_intercept = solution_eval[1][0]

	print('slope: ' + str(slope))
	print('y_intercept: ' + str(y_intercept))

	# Get best fit line
	best_fit = []
	for i in x_vals:
	  best_fit.append(slope*i+y_intercept)

	# Plot the results
	plt.plot(x_vals, y_vals, 'o', label='Data')
	plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
	plt.legend(loc='upper left')
	plt.show()

#-----------------3.5 TF解决戴明回归问题---------------
# Deming Regression
#----------------------------------
#
# This function shows how to use Tensorflow to
# solve linear Deming regression.
# y = Ax + b
#
# We will use the iris data, specifically:
#  y = Sepal Length
#  x = Petal Width
def _tf305CalDMregtf():	
	ops.reset_default_graph()

	# Create graph
	sess = tf.Session()

	# Load the data
	# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
	iris = datasets.load_iris()
	x_vals = np.array([x[3] for x in iris.data])
	y_vals = np.array([y[0] for y in iris.data])

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

	# Declare Demming loss function
	demming_numerator = tf.abs(tf.subtract(y_target, tf.add(tf.matmul(x_data, A), b)))
	demming_denominator = tf.sqrt(tf.add(tf.square(A),1))
	loss = tf.reduce_mean(tf.truediv(demming_numerator, demming_denominator))

	# Initialize variables
	init = tf.global_variables_initializer()
	sess.run(init)

	# Declare optimizer
	my_opt = tf.train.GradientDescentOptimizer(0.1)
	train_step = my_opt.minimize(loss)

	# Training loop
	loss_vec = []
	for i in range(250):
		rand_index = np.random.choice(len(x_vals), size=batch_size)
		rand_x = np.transpose([x_vals[rand_index]])
		rand_y = np.transpose([y_vals[rand_index]])
		sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
		temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
		loss_vec.append(temp_loss)
		if (i+1)%50==0:
			print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
			print('Loss = ' + str(temp_loss))

	# Get the optimal coefficients
	[slope] = sess.run(A)
	[y_intercept] = sess.run(b)

	# Get best fit line
	best_fit = []
	for i in x_vals:
	  best_fit.append(slope*i+y_intercept)

	# Plot the result
	plt.plot(x_vals, y_vals, 'o', label='Data Points')
	plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
	plt.legend(loc='upper left')
	plt.title('Sepal Length vs Pedal Width')
	plt.xlabel('Pedal Width')
	plt.ylabel('Sepal Length')
	plt.show()

	# Plot loss over time
	plt.plot(loss_vec, 'k-')
	plt.title('L2 Loss per Generation')
	plt.xlabel('Generation')
	plt.ylabel('L2 Loss')
	plt.show()

#----3.6 tensorflow实现Lasso回归和岭回归（Ridge Regression）---
# Lasso and Ridge Regression
#----------------------------------
#
# This function shows how to use Tensorflow to
# solve lasso or ridge regression.
# y = Ax + b
#
# We will use the iris data, specifically:
#  y = Sepal Length
#  x = Petal Width
def _tf306CalLassR():	
	ops.reset_default_graph()

	# Create graph
	sess = tf.Session()

	# Load the data
	# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
	iris = datasets.load_iris()
	x_vals = np.array([x[3] for x in iris.data])
	y_vals = np.array([y[0] for y in iris.data])

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

	# Declare Lasso loss function
	# Lasso Loss = L2_Loss + heavyside_step,
	# Where heavyside_step ~ 0 if A < constant, otherwise ~ 99
	#lasso_param = tf.constant(0.9)
	#heavyside_step = tf.truediv(1., tf.add(1., tf.exp(tf.mul(-100., tf.sub(A, lasso_param)))))
	#regularization_param = tf.mul(heavyside_step, 99.)
	#loss = tf.add(tf.reduce_mean(tf.square(y_target - model_output)), regularization_param)

	# Declare the Ridge loss function
	# Ridge loss = L2_loss + L2 norm of slope
	ridge_param = tf.constant(1.)
	ridge_loss = tf.reduce_mean(tf.square(A))
	loss = tf.expand_dims(tf.add(tf.reduce_mean(tf.square(y_target - model_output)), tf.multiply(ridge_param, ridge_loss)), 0)
	'''
	TensorFlow中，想要维度增加一维，可以使用tf.expand_dims(input, dim, name=None)函数。当然，我们常用tf.reshape(input, shape=[])也可以达到相同效果，但是有些时候在构建图的过程中，placeholder没有被feed具体的值，这时就会包下面的错误：TypeError: Expected binary or unicode string, got 1 
	在这种情况下，我们就可以考虑使用expand_dims来将维度加1。比如我自己代码中遇到的情况，在对图像维度降到二维做特定操作后，要还原成四维[batch, height, width, channels]，前后各增加一维。如果用reshape，则因为上述原因报错
	'''
	# Initialize variables
	init = tf.global_variables_initializer()
	sess.run(init)

	# Declare optimizer
	my_opt = tf.train.GradientDescentOptimizer(0.001)
	train_step = my_opt.minimize(loss)

	# Training loop
	loss_vec = []
	for i in range(1500):
		rand_index = np.random.choice(len(x_vals), size=batch_size)
		rand_x = np.transpose([x_vals[rand_index]])
		rand_y = np.transpose([y_vals[rand_index]])
		sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
		temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
		loss_vec.append(temp_loss[0])
		if (i+1)%300==0:
			print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
			print('Loss = ' + str(temp_loss))

	# Get the optimal coefficients
	[slope] = sess.run(A)
	[y_intercept] = sess.run(b)

	# Get best fit line
	best_fit = []
	for i in x_vals:
	  best_fit.append(slope*i+y_intercept)

	# Plot the result
	plt.plot(x_vals, y_vals, 'o', label='Data Points')
	plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
	plt.legend(loc='upper left')
	plt.title('Sepal Length vs Pedal Width')
	plt.xlabel('Pedal Width')
	plt.ylabel('Sepal Length')
	plt.show()

	# Plot loss over time
	plt.plot(loss_vec, 'k-')
	plt.title('L2 Loss per Generation')
	plt.xlabel('Generation')
	plt.ylabel('L2 Loss')
	plt.show()

#----3.7 tensorflow实现Lasso回归和岭回归（Ridge Regression）---
# Elastic Net Regression
#----------------------------------
#
# This function shows how to use Tensorflow to
# solve elastic net regression.
# y = Ax + b
#
# We will use the iris data, specifically:
#  y = Sepal Length
#  x = Pedal Length, Petal Width, Sepal Width
def _tf307CalElassR():	
	ops.reset_default_graph()

	# Create graph
	sess = tf.Session()

	# Load the data
	# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
	iris = datasets.load_iris()
	x_vals = np.array([[x[1], x[2], x[3]] for x in iris.data])
	y_vals = np.array([y[0] for y in iris.data])

	# Declare batch size
	batch_size = 50

	# Initialize placeholders
	x_data = tf.placeholder(shape=[None, 3], dtype=tf.float32)
	y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

	# Create variables for linear regression
	A = tf.Variable(tf.random_normal(shape=[3,1]))
	b = tf.Variable(tf.random_normal(shape=[1,1]))

	# Declare model operations
	model_output = tf.add(tf.matmul(x_data, A), b)

	# Declare the elastic net loss function
	elastic_param1 = tf.constant(1.)
	elastic_param2 = tf.constant(1.)
	l1_a_loss = tf.reduce_mean(tf.abs(A))
	l2_a_loss = tf.reduce_mean(tf.square(A))
	e1_term = tf.multiply(elastic_param1, l1_a_loss)
	e2_term = tf.multiply(elastic_param2, l2_a_loss)
	loss = tf.expand_dims(tf.add(tf.add(tf.reduce_mean(tf.square(y_target - model_output)), e1_term), e2_term), 0)

	# Initialize variables
	init = tf.global_variables_initializer()
	sess.run(init)

	# Declare optimizer
	my_opt = tf.train.GradientDescentOptimizer(0.001)
	train_step = my_opt.minimize(loss)

	# Training loop
	loss_vec = []
	for i in range(1000):
		rand_index = np.random.choice(len(x_vals), size=batch_size)
		rand_x = x_vals[rand_index]
		rand_y = np.transpose([y_vals[rand_index]])
		sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
		temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
		loss_vec.append(temp_loss[0])
		if (i+1)%250==0:
			print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
			print('Loss = ' + str(temp_loss))

	# Get the optimal coefficients
	[[sw_coef], [pl_coef], [pw_ceof]] = sess.run(A)
	[y_intercept] = sess.run(b)

	# Plot loss over time
	plt.plot(loss_vec, 'k-')
	plt.title('Loss per Generation')
	plt.xlabel('Generation')
	plt.ylabel('Loss')
	plt.show()


#----3.8 tensorflow实现逻辑回归---
# Logistic Regression
#----------------------------------
#
# This function shows how to use Tensorflow to
# solve logistic regression.
# y = sigmoid(Ax + b)
#
# We will use the low birth weight data, specifically:
#  y = 0 or 1 = low birth weight
#  x = demographic and medical history data
def _tf308Callogic():	
	ops.reset_default_graph()

	# Create graph
	sess = tf.Session()

	# Load the data
	# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
	iris = datasets.load_iris()
	x_vals = np.array([[x[1], x[2], x[3]] for x in iris.data])
	y_vals = np.array([y[0] for y in iris.data])

	# Declare batch size
	batch_size = 50

	# Initialize placeholders
	x_data = tf.placeholder(shape=[None, 3], dtype=tf.float32)
	y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

	# Create variables for linear regression
	A = tf.Variable(tf.random_normal(shape=[3,1]))
	b = tf.Variable(tf.random_normal(shape=[1,1]))

	# Declare model operations
	model_output = tf.add(tf.matmul(x_data, A), b)

	# Declare the elastic net loss function
	elastic_param1 = tf.constant(1.)
	elastic_param2 = tf.constant(1.)
	l1_a_loss = tf.reduce_mean(tf.abs(A))
	l2_a_loss = tf.reduce_mean(tf.square(A))
	e1_term = tf.multiply(elastic_param1, l1_a_loss)
	e2_term = tf.multiply(elastic_param2, l2_a_loss)
	loss = tf.expand_dims(tf.add(tf.add(tf.reduce_mean(tf.square(y_target - model_output)), e1_term), e2_term), 0)

	# Initialize variables
	init = tf.global_variables_initializer()
	sess.run(init)

	# Declare optimizer
	my_opt = tf.train.GradientDescentOptimizer(0.001)
	train_step = my_opt.minimize(loss)

	# Training loop
	loss_vec = []
	for i in range(1000):
		rand_index = np.random.choice(len(x_vals), size=batch_size)
		rand_x = x_vals[rand_index]
		rand_y = np.transpose([y_vals[rand_index]])
		sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
		temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
		loss_vec.append(temp_loss[0])
		if (i+1)%250==0:
			print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
			print('Loss = ' + str(temp_loss))

	# Get the optimal coefficients
	[[sw_coef], [pl_coef], [pw_ceof]] = sess.run(A)
	[y_intercept] = sess.run(b)

	# Plot loss over time
	plt.plot(loss_vec, 'k-')
	plt.title('Loss per Generation')
	plt.xlabel('Generation')
	plt.ylabel('Loss')
	plt.show()



















	
	
	
	
	
	
	
	
	
	
	
	
#--------------main()-------------

if __name__=='__main__':
	_tf308Callogic()
























