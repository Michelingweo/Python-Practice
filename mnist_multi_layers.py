from skimage import io, transform
import tensorflow as tf
import numpy as np
import os  # os 处理文件和目录的模块
import glob  # glob 文件通配符模块
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from matplotlib.colors import colorConverter, ListedColormap
from sklearn.preprocessing import OneHotEncoder
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("E:\python practice\MNIST_data", one_hot=True)


learning_rate=0.001
training_epochs=25
batch_size=100
display_step=1

#network parameter
n_hidden_1 = 256
n_hidden_2 = 256
n_input=784 #MNIST 784 dimension
n_classes=10 #MNIST 10 categories

#network framework
#placeholder
x=tf.placeholder('float',[None,n_input])
y=tf.placeholder('float',[None,n_classes])

#model
def multilayer_perceptron(x,weight,biases):
    #first hidden layer1
    layer_1=tf.add(tf.matmul(x,weight['h1']),biases['h1'])
    layer_1=tf.nn.relu(layer_1)
    #second hidden layer
    layer_2=tf.add(tf.matmul(layer_1,weight['h2']),biases['h2'])
    layer_2=tf.nn.relu(layer_2)
    #output layer
    output_layer=tf.matmul(layer_2,weight['out'])+biases['out']
    
    return output_layer

#learning parameter
weight={
    'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
}
biases={
    'h1': tf.Variable(tf.random_normal([n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#output value
pred=multilayer_perceptron(x,weight,biases)

#loss and optimizer
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#initialization
init=tf.global_variables_initializer()

#set up session
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost=0.
        total_batch=int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _,c=sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})

            avg_cost += c/total_batch
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                    "{:.9f}".format(avg_cost))
    print(" Finished!")

    #test the model
    correct_prediction= tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    #calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


