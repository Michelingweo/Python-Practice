import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#loss visualization function
plotdata={"batchsize":[],"loss":[]}
def moving_average(a,w=10):
  if len(a)<w:
    return a[:]
  return [val if idx < w else sum(a[(idx-w):idx])/w for idx,val in enumerate(a)]

train_X=np.linspace(-1,1,100)
train_Y=2*train_X+np.random.randn(*train_X.shape)

plt.plot(train_X,train_Y,'ro',label='Original data')
plt.legend()
plt.show()
tf.reset_default_graph()
X=tf.placeholder('float')
Y=tf.placeholder('float')
#parameter
W=tf.Variable(tf.random.normal([1]),name="weight")
b=tf.Variable(tf.zeros([1]),name="bias")
#forwardfeed
z=tf.multiply(X,W)+b

cost=tf.reduce_mean(tf.square(Y-z))
learning_rate=0.1
optimizer=tf.train.GradientDescentOptimizer(learning_rate,name="GradientDescent").minimize(cost)
init=tf.global_variables_initializer()
train_epochs=20
display_step=2
saver=tf.train.Saver(max_to_keep=1)
savedir='log/'
with tf.Session() as sess:
  sess.run(init)
  plotdata={"batchsize":[],"loss":[]}

  for epoch in range(train_epochs):
    for (x,y) in zip(train_X,train_Y):
      sess.run(optimizer, feed_dict={X:x,Y:y})

    if epoch % display_step == 0:
      loss=sess.run(cost,feed_dict={X:train_X,Y:train_Y})
      print("Epoch:",epoch+1,"cost:",loss,"W=",sess.run(W),"b=",sess.run(b))
      if not (loss=="NA"):
        plotdata["batchsize"].append(epoch)
        plotdata["loss"].append(loss)
  print("Finished")
  print("cost=",sess.run(cost,feed_dict={X:train_X,Y:train_Y}),"W=",sess.run(W),"b=",sess.run(b))

  plt.plot(train_X,train_Y,'ro',label='Original data')
  plt.plot(train_X,sess.run(W)*train_X+sess.run(b),label="fitted line")
  plt.legend()
  plt.show()

  plotdata["avgloss"]=moving_average(plotdata['loss'])
  plt.figure(1)
  plt.subplot(211)
  plt.plot(plotdata['batchsize'],plotdata['avgloss'],'b')
  plt.xlabel('minibatch number')
  plt.ylabel('loss')
  plt.title('minibatch run VS. Training loss')
  plt.show()

load_epoch=18
with tf.Session() as sess2:
  sess2.run(tf.global_variables_initializer())
  saver.restore(sess2,savedir+'linearmodel.cpkt-'+str(load_epoch))
  print('x=0.2,z=',sess2.run(z,feed_dict={X:0.2}))