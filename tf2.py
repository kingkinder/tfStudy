# Tensors 进阶
'''
tensorflow嵌入layer
tensorflow多层layer
tensorflow实现损失函数
tensorflow反向传播
tensorflow实现随机训练和批量训练
tensorflow实现创建分类器
tensorflow实现模型评估
'''
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()
import matplotlib.pyplot as plt
import numpy as np
# Introduce tensors in tf
#--------------------2.1简单的计算图实现------------------
# Create graph
sess = tf.Session()

# Create tensors

# Create data to feed in
x_vals = np.array([1., 3., 5., 7., 9.])
x_data = tf.placeholder(tf.float32)
m = tf.constant(3.)

# Multiplication
prod = tf.multiply(x_data, m)
for x_val in x_vals:
    print(sess.run(prod, feed_dict={x_data: x_val}))

#--------------------2.2 嵌入layer------------------
# Create data to feed in
my_array = np.array([[1., 3., 5., 7., 9.],
                   [-2., 0., 2., 4., 6.],
                   [-6., -3., 0., 3., 6.]])
x_vals = np.array([my_array, my_array + 1])
x_data = tf.placeholder(tf.float32, shape=(3, 5))
m1 = tf.constant([[1.],[0.],[-1.],[2.],[4.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])

# 1st Operation Layer = Multiplication
prod1 = tf.matmul(x_data, m1)

# 2nd Operation Layer = Multiplication
prod2 = tf.matmul(prod1, m2)

# 3rd Operation Layer = Addition
add1 = tf.add(prod2, a1)

for x_val in x_vals:
    print(sess.run(add1, feed_dict={x_data: x_val}))

#--------------------2.3 多层layer------------------
# Create a small random 'image' of size 4x4
x_shape = [1, 4, 4, 1]
x_val = np.random.uniform(size=x_shape)

x_data = tf.placeholder(tf.float32, shape=x_shape)

# Create a layer that takes a spatial moving window average
# Our window will be 2x2 with a stride of 2 for height and width
# The filter value will be 0.25 because we want the average of the 2x2 window
my_filter = tf.constant(0.25, shape=[2, 2, 1, 1])
my_strides = [1, 2, 2, 1]
mov_avg_layer= tf.nn.conv2d(x_data, my_filter, my_strides,
                            padding='SAME', name='Moving_Avg_Window')

'''
卷积层说明
（旧版本）
input: 一个4维Tensor（N,H,W,C). 类型必须是以下几种类型之一: half, float32, float64.
filter: 卷积核（滤波器）. 类型和input必须相同，4维tensor, [filter_height, filter_width, in_channels, out_channels],如[5,5,3,32]，分别表示 卷积核的高度，宽度，输入通道数，输出通道数
[filter.size,filter.size,depth,num]，前三个参数定义了单个卷积核的尺寸，其中的depth应该等于需要卷积的层的depth，即[batch,image.size,image.size,depth]中的第四个参数。而num可以理解为卷积核的个数，它决定了下一层输出的通道数。
strides:  在input上切片采样时，每个方向上的滑窗步长，必须和format指定的维度同阶,如[1, 2, 2, 1]
padding: 指定边缘填充类型: "SAME", "VALID". SAME表示卷积后图片保持不变，VALID则会缩小。
use_cudnn_on_gpu: 可选项，bool型。表示是否在GPU上用cudnn进行加速，默认为True.
data_format: 可选项，指定输入数据的格式: "NHWC"或 "NCHW"， 默认为"NHWC"。
NHWC格式指[batch, in_height, in_width, in_channels]
NCHW格式指[batch, in_channels, in_height, in_width]
name: 操作名，可选.

conv2d(
    inputs,
    filters,
    kernel_size,
    strides=(1, 1),
    padding='valid',
    data_format='channels_last',
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    trainable=True,
    name=None,
    reuse=None
)

参数多了很多，但实际用起来，却更简单。
（新版本）
inputs: 输入数据，4维tensor.
filters: 卷积核个数。
kernel_size:卷积核大小，如【5，5】。如果长宽相等，也可以直接设置为一个数，如kernel_size=5
strides: 卷积过程中的滑动步长，默认为[1,1]. 也可以直接设置为一个数，如strides=2
padding: 边缘填充，'same' 和'valid‘选其一。默认为valid
data_format: 输入数据格式，默认为channels_last ，即 (batch, height, width, channels),也可以设置为channels_first 对应 (batch, channels, height, width).
dilation_rate: 微步长卷积，这个比较复杂一些，请百度.
activation: 激活函数.
use_bias: Boolean型，是否使用偏置项.
kernel_initializer: 卷积核的初始化器.
bias_initializer: 偏置项的初始化器，默认初始化为0.
kernel_regularizer: 卷积核化的正则化，可选.
bias_regularizer: 偏置项的正则化，可选.
activity_regularizer: 输出的正则化函数.
trainable: Boolean型，表明该层的参数是否参与训练。如果为真则变量加入到图集合中 GraphKeys.TRAINABLE_VARIABLES (see tf.Variable).
name: 层的名字.
reuse: Boolean型, 是否重复使用参数.
'''							
							
# Define a custom layer which will be sigmoid(Ax+b) where
# x is a 2x2 matrix and A and b are 2x2 matrices
def custom_layer(input_matrix):
    input_matrix_sqeezed = tf.squeeze(input_matrix)
    A = tf.constant([[1., 2.], [-1., 3.]])
    b = tf.constant(1., shape=[2, 2])
    temp1 = tf.matmul(A, input_matrix_sqeezed)
    temp = tf.add(temp1, b) # Ax + b
    return(tf.sigmoid(temp))

# Add custom layer to graph
with tf.name_scope('Custom_Layer') as scope:
    custom_layer1 = custom_layer(mov_avg_layer)

# The output should be an array that is 2x2, but size (1,2,2,1)
#print(sess.run(mov_avg_layer, feed_dict={x_data: x_val}))

# After custom operation, size is now 2x2 (squeezed out size 1 dims)
print(sess.run(custom_layer1, feed_dict={x_data: x_val}))
	
#--------------------2.4 实现损失函数------------------	
# Create graph
sess = tf.Session()

###### Numerical Predictions ######
x_vals = tf.linspace(-1., 1., 500)#0到1之间的等差数列
target = tf.constant(0.)

# L2 loss，正则损失函数（欧拉损失函数），差值评分和
# L = (pred - actual)^2
#tensorflow有内建的L2正则形式，nn.l2_loss，上式的一半
l2_y_vals = tf.square(target - x_vals)
l2_y_out = sess.run(l2_y_vals)

# L1 loss ，绝对差值，不能很好的收敛
# L = abs(pred - actual)
l1_y_vals = tf.abs(target - x_vals)
l1_y_out = sess.run(l1_y_vals)

# Pseudo-Huber loss
# L = delta^2 * (sqrt(1 + ((pred - actual)/delta)^2) - 1)
#Pseudo-Huber函数是Huber函数的连续、平滑估计，表达式依赖于delta参数，delta越大，越平滑
delta1 = tf.constant(0.25)
phuber1_y_vals = tf.multiply(tf.square(delta1), tf.sqrt(1. + tf.square((target - x_vals)/delta1)) - 1.)
phuber1_y_out = sess.run(phuber1_y_vals)

delta2 = tf.constant(5.)
phuber2_y_vals = tf.multiply(tf.square(delta2), tf.sqrt(1. + tf.square((target - x_vals)/delta2)) - 1.)
phuber2_y_out = sess.run(phuber2_y_vals)

# Plot the output:
x_array = sess.run(x_vals)
plt.plot(x_array, l2_y_out, 'b-', label='L2 Loss')
plt.plot(x_array, l1_y_out, 'r--', label='L1 Loss')
plt.plot(x_array, phuber1_y_out, 'k-.', label='P-Huber Loss (0.25)')
plt.plot(x_array, phuber2_y_out, 'g:', label='P-Huber Loss (5.0)')
plt.ylim(-0.2, 0.4)
plt.legend(loc='lower right', prop={'size': 11})
plt.show()
'''
以上都是用于回归分析的常用损失函数
'''



###### Categorical Predictions ######
x_vals = tf.linspace(-3., 5., 500)
target = tf.constant(1.)
targets = tf.fill([500,], 1.)

# Hinge loss，主要用于支持向量机的算法，也可以用来评估神经网络
# Use for predicting binary (-1, 1) classes
# L = max(0, 1 - (pred * actual))
hinge_y_vals = tf.maximum(0., 1. - tf.multiply(target, x_vals))
hinge_y_out = sess.run(hinge_y_vals)

# Cross entropy loss，两类交叉熵损失函数，也叫逻辑损失函数
# L = -actual * (log(pred)) - (1-actual)(log(1-pred))
xentropy_y_vals = - tf.multiply(target, tf.log(x_vals)) - tf.multiply((1. - target), tf.log(1. - x_vals))
xentropy_y_out = sess.run(xentropy_y_vals)

# Sigmoid entropy loss，交叉熵损失函数，与上一个不同的是通过sigmod函数进行变换
# L = -actual * (log(sigmoid(pred))) - (1-actual)(log(1-sigmoid(pred)))
# or
# L = max(actual, 0) - actual * pred + log(1 + exp(-abs(actual)))
xentropy_sigmoid_y_vals = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=x_vals)
xentropy_sigmoid_y_out = sess.run(xentropy_sigmoid_y_vals)
'''
For brevity, let x = logits, z = labels. The logistic loss is
z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
= z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
= z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
= z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
= (1 - z) * x + log(1 + exp(-x))
= x - x * z + log(1 + exp(-x))

For x < 0, to avoid overflow in exp(-x), we reformulate the above
x - x * z + log(1 + exp(-x))
= log(exp(x)) - x * z + log(1 + exp(-x))
= - x * z + log(1 + exp(x))

Hence, to ensure stability and avoid overflow, the implementation uses this equivalent formulation
max(x, 0) - x * z + log(1 + exp(-abs(x)))
logits and labels must have the same type and shape.

'''


#加权交叉熵损失函数，是SIGMOD函数加权，比如0.5进行加权
# Weighted (softmax) cross entropy loss
# L = -actual * (log(pred)) * weights - (1-actual)(log(1-pred))
# or
# L = (1 - pred) * actual + (1 + (weights - 1) * pred) * log(1 + exp(-actual))
weight = tf.constant(0.5)
xentropy_weighted_y_vals = tf.nn.weighted_cross_entropy_with_logits(x_vals, targets, weight)
xentropy_weighted_y_out = sess.run(xentropy_weighted_y_vals)

# Plot the output
x_array = sess.run(x_vals)
plt.plot(x_array, hinge_y_out, 'b-', label='Hinge Loss')
plt.plot(x_array, xentropy_y_out, 'r--', label='Cross Entropy Loss')
plt.plot(x_array, xentropy_sigmoid_y_out, 'k-.', label='Cross Entropy Sigmoid Loss')
plt.plot(x_array, xentropy_weighted_y_out, 'g:', label='Weighted Cross Entropy Loss (x0.5)')
plt.ylim(-1.5, 3)
#plt.xlim(-1, 3)
plt.legend(loc='lower right', prop={'size': 11})
plt.show()

# Softmax entropy loss，softmax交叉熵损失函数，针对单个目标分类计算损失，将输出结果转化为概率分布，然后计算概率分布的损失
# L = -actual * (log(softmax(pred))) - (1-actual)(log(1-softmax(pred)))
unscaled_logits = tf.constant([[1., -3., 10.]])
target_dist = tf.constant([[0.1, 0.02, 0.88]])
softmax_xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=unscaled_logits,logits=target_dist)
print(sess.run(softmax_xentropy))

# Sparse entropy loss。稀疏softmax交叉熵损失函数，把目标分类为true的转化为index然后进行交叉熵计算
# Use when classes and targets have to be mutually exclusive
# L = sum( -actual * log(pred) )
unscaled_logits = tf.constant([[1., -3., 10.]])
sparse_target_dist = tf.constant([2])
sparse_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(unscaled_logits, sparse_target_dist)
print(sess.run(sparse_xentropy))


#--------------------2.5 实现反向传播------------------	
# Create data，生产数据，创建占位符和变量A
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)

# Create variable (one model parameter = A)
A = tf.Variable(tf.random_normal(shape=[1]))

# Add operation to graph，增加乘法操作
my_output = tf.multiply(x_data, A)

# Add L2 loss operation to graph，L2正则损失函数
loss = tf.square(my_output - y_target)

# Initialize variables 初始化所有变量
init = tf.initialize_all_variables()
sess.run(init)

# Create Optimizer  声明优化器 标准梯度下降法，学习速度为0.02
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)

# Run Loop
for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1)%25==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))

# Classification Example
# We will create sample data as follows:
# x-data: sample 50 random values from a normal = N(-1, 1)
#         + sample 50 random values from a normal = N(1, 1)
# target: 50 values of 0 + 50 values of 1.
#         These are essentially 100 values of the corresponding output index
# We will fit the binary classification model:
# If sigmoid(x+A) < 0.5 -> 0 else 1
# Theoretically, A should be -(mean1 + mean2)/2

#重置计算图
ops.reset_default_graph()

# Create graph
sess = tf.Session()

# Create data，np.random.normal(3, 1, 50)表示均值三，方差1的50个个随机变量
#concatenate是数组拼接方法，如果要是水平链接，就设置axis=1;就可以了，a=np.array([1,2]),s=np.concatenate((a,a,),axis=1)
x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)

# Create variable (one model parameter = A)
#初始化变量A为0附近的值
A = tf.Variable(tf.random_normal(mean=10, shape=[1]))

# Add operation to graph
# Want to create the operstion sigmoid(x + A)
# Note, the sigmoid() part is in the loss function
my_output = tf.add(x_data, A)

#expand_dims()函数增加维度
# Now we have to add another dimension to each (batch size of 1)
my_output_expanded = tf.expand_dims(my_output, 0)
y_target_expanded = tf.expand_dims(y_target, 0)

# Initialize variables
init = tf.initialize_all_variables()
sess.run(init)

# Add classification loss (cross entropy)，SIGMOD交叉熵
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=my_output_expanded, logits=y_target_expanded)

# Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)

# Run loop，训练1400次
for i in range(1400):
    rand_index = np.random.choice(100)#从0到100随机选择一个数
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1)%200==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(xentropy, feed_dict={x_data: rand_x, y_target: rand_y})))
        
# Evaluate Predictions
predictions = []
for i in range(len(x_vals)):
    x_val = [x_vals[i]]
    prediction = sess.run(tf.round(tf.sigmoid(my_output)), feed_dict={x_data: x_val})
    predictions.append(prediction[0])
    
accuracy = sum(x==y for x,y in zip(predictions, y_vals))/100.
print('Ending Accuracy = ' + str(np.round(accuracy, 2)))
	
#------------------2.6 实现随机训练和批量训练----------------	
	# Create data
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)

# Create variable (one model parameter = A)
A = tf.Variable(tf.random_normal(shape=[1]))

# Add operation to graph
my_output = tf.multiply(x_data, A)

# Add L2 loss operation to graph
loss = tf.square(my_output - y_target)

# Initialize variables
init = tf.initialize_all_variables()
sess.run(init)

# Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)

loss_stochastic = []
# Run Loop
for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1)%5==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        print('Loss = ' + str(temp_loss))
        loss_stochastic.append(temp_loss)
        

# Batch Training:
# Re-initialize graph
ops.reset_default_graph()
sess = tf.Session()

# Declare batch size,批量大小
batch_size = 20

# Create data
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variable (one model parameter = A)
A = tf.Variable(tf.random_normal(shape=[1,1]))

# Add operation to graph
my_output = tf.matmul(x_data, A)

# Add L2 loss operation to graph
loss = tf.reduce_mean(tf.square(my_output - y_target))

# Initialize variables
init = tf.initialize_all_variables()
sess.run(init)

# Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)

loss_batch = []
# Run Loop
for i in range(100):
    rand_index = np.random.choice(100, size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1)%5==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        print('Loss = ' + str(temp_loss))
        loss_batch.append(temp_loss)
        
plt.plot(range(0, 100, 5), loss_stochastic, 'b-', label='Stochastic Loss')
plt.plot(range(0, 100, 5), loss_batch, 'r--', label='Batch Loss, size=20')
plt.legend(loc='upper right', prop={'size': 11})
plt.show()
	
	
#------------------2.7 分类器实战----------------		
from sklearn import datasets
from tensorflow.python.framework import ops
ops.reset_default_graph()	
	
# Load the iris data
# iris.target = {0, 1, 2}, where '0' is setosa
# iris.data ~ [sepal.width, sepal.length, pedal.width, pedal.length]
iris = datasets.load_iris()
binary_target = np.array([1. if x==0 else 0. for x in iris.target])
iris_2d = np.array([[x[2], x[3]] for x in iris.data])	
# Declare batch size
batch_size = 20

# Create graph
sess = tf.Session()

# Declare placeholders 第一位占位符应该声明为None
x1_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
x2_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variables A and b (0 = x1 - A*x2 + b)
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# Add model to graph:
# x1 - A*x2 + b
my_mult = tf.matmul(x2_data, A)
my_add = tf.add(my_mult, b)
my_output = tf.subtract(x1_data, my_add)
#my_output = tf.sub(x_data[0], tf.add(tf.matmul(x_data[1], A), b))

# Add classification loss (cross entropy)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=my_output, logits=y_target)

# Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Run Loop
for i in range(1000):
    rand_index = np.random.choice(len(iris_2d), size=batch_size)
    #rand_x = np.transpose([iris_2d[rand_index]])
    rand_x = iris_2d[rand_index]
    rand_x1 = np.array([[x[0]] for x in rand_x])
    rand_x2 = np.array([[x[1]] for x in rand_x])
    #rand_y = np.transpose([binary_target[rand_index]])
    rand_y = np.array([[y] for y in binary_target[rand_index]])
    sess.run(train_step, feed_dict={x1_data: rand_x1, x2_data: rand_x2, y_target: rand_y})
    if (i+1)%200==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ', b = ' + str(sess.run(b)))
        

# Visualize Results
# Pull out slope/intercept
[[slope]] = sess.run(A)
[[intercept]] = sess.run(b)

# Create fitted line
x = np.linspace(0, 3, num=50)
ablineValues = []
for i in x:
  ablineValues.append(slope*i+intercept)

# Plot the fitted line over the data
setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i]==1]
setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i]==1]
non_setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i]==0]
non_setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i]==0]
plt.plot(setosa_x, setosa_y, 'rx', ms=10, mew=2, label='setosa')
plt.plot(non_setosa_x, non_setosa_y, 'ro', label='Non-setosa')
plt.plot(x, ablineValues, 'b-')
plt.xlim([0.0, 2.7])
plt.ylim([0.0, 7.1])
plt.suptitle('Linear Separator For I.setosa', fontsize=20)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend(loc='lower right')
plt.show()	
	
	
	
#------------------2.8 模型评估----------------	
ops.reset_default_graph()

# Create graph
sess = tf.Session()

# Regression Example:
# We will create sample data as follows:
# x-data: 100 random samples from a normal ~ N(1, 0.1)
# target: 100 values of the value 10.
# We will fit the model:
# x-data * A = target
# Theoretically, A = 10.

# Declare batch size
batch_size = 25

# Create data
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Split data into train/test = 80%/20%
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# Create variable (one model parameter = A)
A = tf.Variable(tf.random_normal(shape=[1,1]))

# Add operation to graph
my_output = tf.matmul(x_data, A)

# Add L2 loss operation to graph
loss = tf.reduce_mean(tf.square(my_output - y_target))

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)

# Run Loop
for i in range(100):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = np.transpose([x_vals_train[rand_index]])
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1)%25==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))

# Evaluate accuracy (loss) on test set
mse_test = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_test]), y_target: np.transpose([y_vals_test])})
mse_train = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_train]), y_target: np.transpose([y_vals_train])})
print('MSE on test:' + str(np.round(mse_test, 2)))
print('MSE on train:' + str(np.round(mse_train, 2)))

# Classification Example
# We will create sample data as follows:
# x-data: sample 50 random values from a normal = N(-1, 1)
#         + sample 50 random values from a normal = N(1, 1)
# target: 50 values of 0 + 50 values of 1.
#         These are essentially 100 values of the corresponding output index
# We will fit the binary classification model:
# If sigmoid(x+A) < 0.5 -> 0 else 1
# Theoretically, A should be -(mean1 + mean2)/2

ops.reset_default_graph()

# Create graph
sess = tf.Session()

# Declare batch size
batch_size = 25

# Create data
x_vals = np.concatenate((np.random.normal(-1, 0.5, 50), np.random.normal(2, 0.5, 50)))
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))
x_data = tf.placeholder(shape=[1, None], dtype=tf.float32)
y_target = tf.placeholder(shape=[1, None], dtype=tf.float32)

# Split data into train/test = 80%/20%
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# Create variable (one model parameter = A)
A = tf.Variable(tf.random_normal(mean=10, shape=[1]))

# Add operation to graph
# Want to create the operstion sigmoid(x + A)
# Note, the sigmoid() part is in the loss function
my_output = tf.add(x_data, A)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Add classification loss (cross entropy)
xentropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_target, logits=my_output))

# Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)

# Run loop
for i in range(1800):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = [x_vals_train[rand_index]]
    rand_y = [y_vals_train[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1)%200==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(xentropy, feed_dict={x_data: rand_x, y_target: rand_y})))
        
# Evaluate Predictions on test set，模型比较和评估
y_prediction = tf.squeeze(tf.round(tf.nn.sigmoid(tf.add(x_data, A))))
correct_prediction = tf.equal(y_prediction, y_target)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc_value_test = sess.run(accuracy, feed_dict={x_data: [x_vals_test], y_target: [y_vals_test]})
acc_value_train = sess.run(accuracy, feed_dict={x_data: [x_vals_train], y_target: [y_vals_train]})
print('Accuracy on train set: ' + str(acc_value_train))
print('Accuracy on test set: ' + str(acc_value_test))

# Plot classification result
A_result = -sess.run(A)
bins = np.linspace(-5, 5, 50)
plt.hist(x_vals[0:50], bins, alpha=0.5, label='N(-1,1)', color='green')
plt.hist(x_vals[50:100], bins[0:50], alpha=0.5, label='N(2,1)', color='red')
plt.plot((A_result, A_result), (0, 8), 'k--', linewidth=3, label='A = '+ str(np.round(A_result, 2)))
plt.legend(loc='upper right')
plt.title('Binary Classifier, Accuracy=' + str(np.round(acc_value_test, 2)))
plt.show()















































