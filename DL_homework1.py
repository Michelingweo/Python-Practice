import numpy as np

import matplotlib.pyplot as plt


# definition of sigmoid function


def sigmoid(x):
    return 1/(1+np.exp(x))


# definition of SGD
# def DeltaSGD(W,X,D,alpha=0.9,N=4,b=0):
#     #set up the weighted sum v
#     v=[]
#     #set up the output value list y
#     y=[]
#     #list of error
#     Error=[]
#     #list of Delta
#     Delta=[]
#
#     for i in range(N):
#
#         DeltaW = []
#         #weight value after add DeltaW
#         W_new = []
#         v.append(np.dot(X[i],W)+b)
#         y.append(sigmoid(v[i]))
#         Error.append(D[i]-y[i])
#         Delta.append(sigmoid(v[i])*(1-sigmoid(v[i]))*Error[i])
#         for i in range(N):
#             DeltaW.append(alpha*Delta[i]*X[i])
#             W_new.append(DeltaW[i] + W[i])
#
#     return Error


def SGD(X, Y, W, alpha=0.8, b=0):
    v = float(np.dot(X,W)+b)
    d = sigmoid(v)
    error = d - Y
    delta = sigmoid(v)*(1-sigmoid(v))*error
    dW = []
    for i in range(len(X)):
        xi = float(X[i])
        dW.append(alpha*delta*xi)
        W[i] = W[i]+dW[i]

    return error, d


if __name__ == "__main__":

    #Data initialization
    x=[
      [0,0,1],
      [0,1,1],
      [1,0,1],
      [1,1,1]
    ]
    #correct value
    Y = [0,0,1,1]
    #weights initialization
    W = [0,0,0]
    epoch = 10000
    Error_history = [None]*epoch
    Output_history = [None]*epoch
    ####################test
    # for elem in range(len(x)):
    #     a,b=SGD(x[elem],Y[elem],W)
    #     print('error；',a,'output',b,'\n')




    ##################
    # training
    for ep in range(epoch):
        if ep%100==0:
            print('No.',ep,':',W,'\n')
        for elem in range(len(x)):
            Error_history[ep],Output_history[ep]=SGD(x[elem],Y[elem],W)

    #set training epoch as x axis
    x_axis=np.arange(0,epoch,1)

    output4=[]
    for i in range(len(Output_history)):
        if i%2==0 and i%4!=0:
            output4.append(Output_history[i])
    x4_axis = np.arange(0, len(output4),1)
    #plot error changes
    plt.plot(x_axis,Error_history,label='Error')
    plt.title('Change of the Errors')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()
    #plot output changes
    plt.plot(x4_axis,output4,label='output')
    plt.title('Change of the output')
    plt.xlabel('Epoch')
    plt.ylabel('output')
    plt.show()

    print('The final weights are:',W)
    for i in range(20):
        print(Output_history[i],'\n')
    print(Output_history[0])
