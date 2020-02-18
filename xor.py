from skimage import io, transform
import tensorflow as tf
import numpy as np
import os  # os 处理文件和目录的模块
import glob  # glob 文件通配符模块
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from matplotlib.colors import colorConverter, ListedColormap
from sklearn.preprocessing import OneHotEncoder


learning_rate=1e-4
n_input = 2 #number of nodes in input layer
n_label=1 #number of nodes in output layer
n_hidden=2 #number of nodes in hidden layers

x = tf.placeholder(tf.float32,[None,n_input])
y = tf.placeholder(tf.float32,[None,n_label])

weight = {
    'h1':tf.Variable(tf.truncated_normal([n_input,n_hidden], stddev=0.1)),
    'h2':tf.Variable(tf.truncated_normal([n_hidden,n_label], stddev=0.1))
}
biases = {
    'h1': tf.Variable(tf.zeros([n_hidden])),
    'h2': tf.Variable(tf.zeros([n_label]))
}

layer_1 = tf.nn.relu(tf.add(tf.matmul(x,weight['h1']),biases['h1']))
#layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1,weight['h2']),bias=bias['h2']))
# y_pred=tf.maximum(layer_2,layer_2*0.01)
y_pred = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weight['h2']),biases['h2']))

loss=tf.reduce_mean((y_pred-y)**2)
train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss)

#generate DATA
X=[[0,0],[0,1],[1,0],[1,1]]
Y=[[0],[1],[1],[0]]
X=np.array(X).astype('float32')
Y=np.array(Y).astype('int16')

#set up session
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

#train
for i in range(20000):
    sess.run(train_step,feed_dict={x:X,y:Y})

#calculate prediction value
print(sess.run(y_pred,feed_dict={x:X}))

#hidden layer
print(sess.run(layer_1,feed_dict={x:X}))