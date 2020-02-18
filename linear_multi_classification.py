from skimage import io, transform
import tensorflow as tf
import numpy as np
import os  # os 处理文件和目录的模块
import glob  # glob 文件通配符模块
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from matplotlib.colors import colorConverter, ListedColormap
from sklearn.preprocessing import OneHotEncoder


def onehot(y, start, end):
    ohe = OneHotEncoder(categories='auto')
    a = np.linspace(start, end - 1, end - start)
    b = np.reshape(a, [-1, 1]).astype(np.int32)
    ohe.fit(b)
    c = ohe.transform(y).toarray()
    return c


#

def generate(sample_size, num_classes, diff, regression=False):
    np.random.seed(10)
    mean = np.random.randn(2)
    cov = np.eye(2)

    # len(diff)
    samples_per_class = int(sample_size / num_classes)

    X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    Y0 = np.zeros(samples_per_class)

    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean + d, cov, samples_per_class)
        Y1 = (ci + 1) * np.ones(samples_per_class)

        X0 = np.concatenate((X0, X1))
        Y0 = np.concatenate((Y0, Y1))
        # print(X0, Y0)

    if regression == False:  # one-hot  0 into the vector "1 0
        Y0 = np.reshape(Y0, [-1, 1])
        # print(Y0.astype(np.int32))
        Y0 = onehot(Y0.astype(np.int32), 0, num_classes)
        # print(Y0)
    X, Y = shuffle(X0, Y0)
    # print(X, Y)
    return X, Y

np.random.seed(10)
input_dim = 2
num_classes = 3

X, Y = generate(2000, num_classes, [[3.0],[3.0,0]], False)
aa = [np.argmax(l) for l in Y]
colors = ['r' if l == 0 else 'b' if l == 1 else 'y' for l in aa[:]]
#plot the actual points with different colors
plt.scatter(X[:,0], X[:,1], c=colors)
plt.xlabel('Scaled age (in yrs)')
plt.ylabel('Tumor size(in cm')
plt.show()

lab_dim=num_classes

input_features=tf.placeholder(tf.float32,[None,input_dim])
input_label=tf.placeholder(tf.float32,[None,lab_dim])
W=tf.Variable(tf.random_normal([input_dim,lab_dim]),name='weight')
b=tf.Variable(tf.zeros([lab_dim]),name='bias')
output=tf.matmul(input_features,W)+b

z=tf.nn.softmax(output)

a1=tf.argmax(tf.nn.softmax(output),axis=1)
b1=tf.argmax(input_label,axis=1)
err=tf.count_nonzero(a1-b1)

cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=input_label,logits=output)
loss=tf.reduce_mean(cross_entropy)

optimizer=tf.train.AdamOptimizer(0.04)
train=optimizer.minimize(loss)

maxEpochs=50
minibatchSize=25


#setup session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(maxEpochs):
        sumerr=0
        for i in range(np.int32(len(Y)/minibatchSize)):
            x1=X[i*minibatchSize:(i+1)*minibatchSize,:]
            y1=Y[i*minibatchSize:(i+1)*minibatchSize,:]

            _,lossval, outputval,errval=sess.run([train,loss,output,err], feed_dict={input_features: x1, input_label:y1})
            sumerr=sumerr+(errval/minibatchSize)
        print('epoch:',"%04d"%(epoch+1),"cost=","{:.9f}".format(lossval),"err=",sumerr/minibatchSize)
    train_X,train_Y=generate(200,num_classes,[[3.0],[3.0,0]],False)
    aa=[np.argmax(l) for l in train_Y]
    colors=['r' if l == 0 else 'b' if l==1 else 'y' for l in aa[:]]
    plt.scatter(train_X[:,0],train_X[:,1],c=colors)

    x=np.linspace(-1,8,200)

    y=-x*(sess.run(W)[0][0]/sess.run(W)[1][0])-sess.run(b)[0]/sess.run(W)[1][0]
    plt.plot(x,y, label='first line',lw=3)

    y=-x*(sess.run(W)[0][1]/sess.run(W)[1][1])-sess.run(b)[1]/sess.run(W)[1][1]
    plt.plot(x,y, label='second line',lw=2)

    y=-x*(sess.run(W)[0][2]/sess.run(W)[1][2])-sess.run(b)[2]/sess.run(W)[1][2]
    plt.plot(x,y, label='third line',lw=1)

    plt.legend()
    plt.show()
    print(sess.run(W),sess.run(b))
    train_X, train_Y = generate(200, num_classes, [[3.0], [3.0, 0]], False)
    aa = [np.argmax(l) for l in train_Y]
    colors = ['r' if l == 0 else 'b' if l == 1 else 'y' for l in aa[:]]
    plt.scatter(train_X[:, 0], train_X[:, 1], c=colors)

    nb_of_xs=200
    xs1=np.linspace(-1,8,num=nb_of_xs)
    xs2=np.linspace(-1,8,num=nb_of_xs)
    xx,yy=np.meshgrid(xs1,xs2)

    #classification plane
    classification_plane=np.zeros((nb_of_xs,nb_of_xs))
    for i in range(nb_of_xs):
        for j in range(nb_of_xs):
            classification_plane[i][j]=sess.run(a1,feed_dict={input_features:[[xx[i,j],yy[i,j] ]]})

    #Creat a color map to show the classification colors for every grid point
    cmap=ListedColormap([
            colorConverter.to_rgba('r',alpha=0.30),
            colorConverter.to_rgba('b',alpha=0.30),
            colorConverter.to_rgba('y',alpha=0.30),
    ])
    plt.contourf(xx,yy,classification_plane,cmap=cmap)
    plt.show()
