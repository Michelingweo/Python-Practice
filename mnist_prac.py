import tensorflow as tf
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import os
import glob
import time
import pylab
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print('input data:',mnist.train.images)
print('input the shape of data:',mnist.train.images.shape)
print('validation shape:',mnist.validation.images.shape)
im=mnist.train.images[1]
im=im.reshape(-1,28)
pylab.imshow(im)
pylab.show()

tf.reset_default_graph()
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])


#Definition of training parameter
W=tf.Variable(tf.random_normal([784,10]))
b=tf.Variable(tf.zeros([10]))

pred=tf.nn.softmax(tf.matmul(x,W)+b)#softmax classification

#cost function
cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))
learning_rate=0.01
optimizer=tf.train.GradientDescentOptimizer(learning_rate,name="GradientDescent").minimize(cost)


train_epochs=25
batch_size=100
display_step=1
saver=tf.train.Saver()
model_path="log/mnist_model.ckpt"
#set up session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(train_epochs):
        avg_cost=0.
        total_batch=int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            _,c=sess.run([optimizer,cost],feed_dict={x: batch_xs, y: batch_ys})
            avg_cost+=c/total_batch
        if(epoch+1)%display_step==0:
            print('Epoch:','%04d'%(epoch+1),'cost=',"{:.9f}".format(avg_cost))

    print('Finished')
    correct_prediction=tf.equal(tf.arg_max(pred,1), tf.argmax(y, 1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    save_path=saver.save(sess,model_path)
    print('model saved in file:%s'%save_path)
    