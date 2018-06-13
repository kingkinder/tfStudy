# Tensors

import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()
import matplotlib.pyplot as plt
import numpy as np
# Introduce tensors in tf
#--------------------1.1变量设置------------------
# Get graph handle
sess = tf.Session()
#创建指定维度的0张量
my_tensor = tf.zeros([1,20])

# 封装张量作为变量，Variable是可变的张量
my_var = tf.Variable(tf.zeros([1,20]))
#也可以通过tf.convert_to_tensor()将任意numpy数组转换为python列表

# Different kinds of variables
row_dim = 2
col_dim = 3 

# Zero initialized variable，0张量
zero_var = tf.Variable(tf.zeros([row_dim, col_dim]))

# One initialized variable，1张量
ones_var = tf.Variable(tf.ones([row_dim, col_dim]))

# shaped like other variable
sess.run(zero_var.initializer)
sess.run(ones_var.initializer)
zero_similar = tf.Variable(tf.zeros_like(zero_var))
ones_similar = tf.Variable(tf.ones_like(ones_var))

sess.run(ones_similar.initializer)
sess.run(zero_similar.initializer)

# Fill shape with a constant，创建指定维度的常数边量
fill_var = tf.Variable(tf.fill([row_dim, col_dim], -1))

# Create a variable from a constant
const_var = tf.Variable(tf.constant([8, 6, 7, 5, 3, 0, 9]))
# This can also be used to fill an array:
const_fill_var = tf.Variable(tf.constant(-1, shape=[row_dim, col_dim]))

# Sequence generation，创建序列变量，包含最后一个值
linear_var = tf.Variable(tf.linspace(start=0.0, stop=1.0, num=3)) # Generates [0.0, 0.5, 1.0] includes the end

#创建序列变量，不包含最后一个值
sequence_var = tf.Variable(tf.range(start=6, limit=15, delta=3)) # Generates [6, 9, 12] doesn't include the end

# Random Numbers

# Random Normal创建随机变量
rnorm_var = tf.random_normal([row_dim, col_dim], mean=0.0, stddev=1.0)

# Initialize operation，一次性初始化所有的变量
initialize_op = tf.global_variables_initializer()

# Add summaries to tensorboard
#merged = tf.merge_all_summaries()

# Initialize graph writer:
#writer = tf.train.SummaryWriter("/tmp/variable_logs", sess.graph_def)

# Run initialization of variable
sess.run(initialize_op)

#------------1.2占位符-----------------
import numpy as np

x = tf.placeholder(tf.float32, shape=(4, 4))
#identity返回占位符输入的数据本身
y = tf.identity(x)

rand_array = np.random.rand(4, 4)

print(sess.run(y, feed_dict={x: rand_array}))



#--------------1.3基本运算-------------------
# Declaring Operations
import matplotlib.pyplot as plt
'''基本运算
add()加法
subtract()
multiply()
matmul()矩阵相乘
div()除法

'''
# div() vs truediv() vs floordiv()
print(sess.run(tf.div(3,4)))
print(sess.run(tf.truediv(3,4)))#浮点运算
print(sess.run(tf.floordiv(3.0,4.0)))#对浮点数进行整数除法运算
# Mod function取模运算
print(sess.run(tf.mod(22.0,5.0)))
# Cross Product，点积运算，只有三维向量可用
print(sess.run(tf.cross([1.,0.,0.],[0.,1.,0.])))

#三角函数运算.
# Trig functions
print(sess.run(tf.sin(3.1416)))
print(sess.run(tf.cos(3.1416)))
# Tangemt
print(sess.run(tf.div(tf.sin(3.1416/4.), tf.cos(3.1416/4.))))

#自定义函数运算
# Custom operation
test_nums = range(15)
#from tensorflow.python.ops import math_ops
#print(sess.run(tf.equal(test_num, 3)))，y=3*x^2-x+10
def custom_polynomial(x_val):
    # Return 3x^2 - x + 10
    return(tf.subtract(3 * tf.square(x_val), x_val) + 10)

print(sess.run(custom_polynomial(11)))
# What should we get with list comprehension
expected_output = [3*x*x-x+10 for x in test_nums]
print(expected_output)

# Tensorflow custom function output
for num in test_nums:
    print(sess.run(custom_polynomial(num)))


#--------------1.4向量运算-------------------
# Identity matrix 对角矩阵
identity_matrix = tf.diag([1.0,1.0,1.0])
print(sess.run(identity_matrix))

# 2x3 random norm matrix  随机矩阵，分配后固定不变
A = tf.truncated_normal([2,3])
print(sess.run(A))

# 2x3 constant matrix 常量矩阵
B = tf.fill([2,3], 5.0)
print(sess.run(B))

# 3x2 random uniform matrix  随机变量，每次运算的时候都会改变值
C = tf.random_uniform([3,2])
print(sess.run(C))
print(sess.run(C)) 

# Create matrix from np array
D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
print(sess.run(D))

# Matrix addition/subtraction  矩阵加法和减法
print(sess.run(A+B))
print(sess.run(B-B))


# Matrix Multiplication  矩阵乘法
print(sess.run(tf.matmul(B, identity_matrix)))

# Matrix Transpose  矩阵转置
print(sess.run(tf.transpose(C))) # Again, new random variables

# Matrix Determinant  计算矩阵行列式
print(sess.run(tf.matrix_determinant(D)))

# Cholesky Decomposition  计算逆矩阵
print(sess.run(tf.cholesky(identity_matrix)))

# Eigenvalues and Eigenvectors 计算特征值和特征向量，第一行是特征值
print(sess.run(tf.self_adjoint_eig(D)))


#--------------1.5 下载数据集-------------------
# Iris Data
from sklearn import datasets
#鸢尾花卉数据集
iris = datasets.load_iris()
print(len(iris.data))
print(len(iris.target))
print(iris.data[0])
print(set(iris.target))

#出生体重数据集
import requests

birthdata_url = 'https://www.umass.edu/statdata/statdata/data/lowbwt.dat'
birth_file = requests.get(birthdata_url)
birth_data = birth_file.text.split('\r\n')[5:]
birth_header = [x for x in birth_data[0].split(' ') if len(x)>=1]
birth_data = [[float(x) for x in y.split(' ') if len(x)>=1] for y in birth_data[1:] if len(y)>=1]
print(len(birth_data))
print(len(birth_data[0]))

# Housing Price Data
import requests

housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housing_file = requests.get(housing_url)
housing_data = [[float(x) for x in y.split(' ') if len(x)>=1] for y in housing_file.text.split('\n') if len(y)>=1]
print(len(housing_data))
print(len(housing_data[0]))


# MNIST Handwriting Data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(len(mnist.train.images))
print(len(mnist.test.images))
print(len(mnist.validation.images))
print(mnist.train.labels[1,:])


# Ham/Spam Text Data
import requests
import io
from zipfile import ZipFile

# Get/read zip file  垃圾短信文本数据集
zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
r = requests.get(zip_url)
z = ZipFile(io.BytesIO(r.content))
file = z.read('SMSSpamCollection')
# Format Data
text_data = file.decode()
text_data = text_data.encode('ascii',errors='ignore')
text_data = text_data.decode().split('\n')
text_data = [x.split('\t') for x in text_data if len(x)>=1]
[text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]
print(len(text_data_train))
print(set(text_data_target))
print(text_data_train[1])


# Movie Review Data  影评样本数据集
import requests
import io
import tarfile

movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
r = requests.get(movie_data_url)
# Stream data into temp object
stream_data = io.BytesIO(r.content)
tmp = io.BytesIO()
while True:
    s = stream_data.read(16384)
    if not s:  
        break
    tmp.write(s)
stream_data.close()
tmp.seek(0)
# Extract tar file
tar_file = tarfile.open(fileobj=tmp, mode="r:gz")
pos = tar_file.extractfile('rt-polaritydata/rt-polarity.pos')
neg = tar_file.extractfile('rt-polaritydata/rt-polarity.neg')
# Save pos/neg reviews
pos_data = []
for line in pos:
    pos_data.append(line.decode('ISO-8859-1').encode('ascii',errors='ignore').decode())
neg_data = []
for line in neg:
    neg_data.append(line.decode('ISO-8859-1').encode('ascii',errors='ignore').decode())
tar_file.close()

print(len(pos_data))
print(len(neg_data))
print(neg_data[0])


# The Works of Shakespeare Data 莎士比亚文本集
import requests

shakespeare_url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'
# Get Shakespeare text
response = requests.get(shakespeare_url)
shakespeare_file = response.content
# Decode binary into string
shakespeare_text = shakespeare_file.decode('utf-8')
# Drop first few descriptive paragraphs.
shakespeare_text = shakespeare_text[7675:]
print(len(shakespeare_text))


# English-German Sentence Translation Data
import requests
import io
from zipfile import ZipFile
sentence_url = 'http://www.manythings.org/anki/deu-eng.zip'
r = requests.get(sentence_url)
z = ZipFile(io.BytesIO(r.content))
file = z.read('deu.txt')
# Format Data
eng_ger_data = file.decode()
eng_ger_data = eng_ger_data.encode('ascii',errors='ignore')
eng_ger_data = eng_ger_data.decode().split('\n')
eng_ger_data = [x.split('\t') for x in eng_ger_data if len(x)>=1]
[english_sentence, german_sentence] = [list(x) for x in zip(*eng_ger_data)]
print(len(english_sentence))
print(len(german_sentence))
print(eng_ger_data[10])


#--------------1.6 激活函数-------------------
# X range
x_vals = np.linspace(start=-10., stop=10., num=100)

# ReLU activation  ReLU函数，映射到0以上
print(sess.run(tf.nn.relu([-3., 3., 10.])))
y_relu = sess.run(tf.nn.relu(x_vals))

# ReLU-6 activation ReLU-6函数，映射到0到6之间
print(sess.run(tf.nn.relu6([-3., 3., 10.])))
y_relu6 = sess.run(tf.nn.relu6(x_vals))

# Sigmoid activation 公司 1/(1+exp(-x))
print(sess.run(tf.nn.sigmoid([-1., 0., 1.])))
y_sigmoid = sess.run(tf.nn.sigmoid(x_vals))

# Hyper Tangent activation  双曲正切函数,((exp(x)-exp(-x))/((exp(x)+exp(-x))
print(sess.run(tf.nn.tanh([-1., 0., 1.])))
y_tanh = sess.run(tf.nn.tanh(x_vals))

# Softsign activation   x/(abs(x)+1)
print(sess.run(tf.nn.softsign([-1., 0., 1.])))
y_softsign = sess.run(tf.nn.softsign(x_vals))

# Softplus activation  Relue函数的平滑版，log(exp(x)+1)
print(sess.run(tf.nn.softplus([-1., 0., 1.])))
y_softplus = sess.run(tf.nn.softplus(x_vals))

# Exponential linear activation (exp(x)+1) if x<0 else x
print(sess.run(tf.nn.elu([-1., 0., 1.])))
y_elu = sess.run(tf.nn.elu(x_vals))

# Plot the different functions
plt.plot(x_vals, y_softplus, 'r--', label='Softplus', linewidth=2)
plt.plot(x_vals, y_relu, 'b:', label='ReLU', linewidth=2)
plt.plot(x_vals, y_relu6, 'g-.', label='ReLU6', linewidth=2)
plt.plot(x_vals, y_elu, 'k-', label='ExpLU', linewidth=0.5)
plt.ylim([-1.5,7])
plt.legend(loc='top left')
plt.show()

plt.plot(x_vals, y_sigmoid, 'r--', label='Sigmoid', linewidth=2)
plt.plot(x_vals, y_tanh, 'b:', label='Tanh', linewidth=2)
plt.plot(x_vals, y_softsign, 'g-.', label='Softsign', linewidth=2)
plt.ylim([-2,2])
plt.legend(loc='top left')
plt.show()




































