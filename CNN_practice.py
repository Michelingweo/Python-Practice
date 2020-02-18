# -*- coding: utf-8 -*-
import tensorflow as tf
#[batch,in_height,in_width,in_channels]

input = tf.Variable(tf.constant(1.0,shape=[1,5,5,1]))
input2 = tf.Variable(tf.constant(1.0,shape=[1,5,5,2]))
input3 = tf.Variable(tf.constant(1.0,shape=[1,4,4,1]))

#[filter_height,filter_width,in_channels,out_channels]

filter1= tf.Variable(tf.constant([-1.0,0,0,-1],shape=[2,2,1,1]))
filter2= tf.Variable(tf.constant([-1.0,0,0,-1,-1.0,0,0,-1],shape=[2,2,1,2]))
filter3= tf.Variable(tf.constant([-1.0,0,0,-1,-1.0,0,0,-1,-1.0,0,0,-1],shape=[2,2,1,3]))
filter4= tf.Variable(tf.constant([-1.0,0,0,-1,
                                  -1.0,0,0,-1,
                                  -1.0,0,0,-1,
                                  -1.0,0,0,-1],shape=[2,2,2,2]))
filter5= tf.Variable(tf.constant([-1.0,0,0,-1,-1.0,0,0,-1],shape=[2,2,2,1]))

#padding: "VALID" edge not filled; "SAME" convolution kernals have access to the edge
op1 = tf.nn.conv2d(input, filter1, strides=[1, 2, 2, 1], padding='SAME') #1个通道输入，生成1个feature map
op2 = tf.nn.conv2d(input, filter2, strides=[1, 2, 2, 1], padding='SAME') #1个通道输入，生成2个feature map
op3 = tf.nn.conv2d(input, filter3, strides=[1, 2, 2, 1], padding='SAME') #1个通道输入，生成3个feature map

op4 = tf.nn.conv2d(input2, filter4, strides=[1, 2, 2, 1], padding='SAME') # 2个通道输入，生成2个feature
op5 = tf.nn.conv2d(input2, filter5, strides=[1, 2, 2, 1], padding='SAME') # 2个通道输入，生成一个feature map

vop1 = tf.nn.conv2d(input, filter1, strides=[1, 2, 2, 1], padding='VALID') # 5*5 对于padding不同而不同
op6 = tf.nn.conv2d(input3, filter1, strides=[1, 2, 2, 1], padding='SAME')
vop6 = tf.nn.conv2d(input3, filter1, strides=[1, 2, 2, 1], padding='VALID')  #4*

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    print('op1:\n',sess.run([op1,filter1]))#1-1 with 0 padding
    print('----------------------------')
    print('op2:\n',sess.run([op2,filter2]))#1-2
    print('----------------------------')
    print('op3:\n',sess.run([op2,filter3]))#1-3
    print('----------------------------')
    print('op4:\n',sess.run([op4,filter4]))#2-2
    print('----------------------------')
    print('op5:\n',sess.run([op5,filter5]))#2-1
    print('----------------------------')
    print('vop1:\n',sess.run([vop1,filter1]))
    print("op6:\n", sess.run([op6, filter1]))
    print("vop6:\n", sess.run([vop6, filter1]))
    print('----------------------------')


