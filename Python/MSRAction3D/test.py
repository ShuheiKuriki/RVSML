import numpy as np

def distL1(X, Y):
    x = np.shape(X)
    m = x[0]
    n = np.shape(Y)[0]
    Z = np.zeros(x)
    D = np.zeros((m,n))
    print(vars())
    for i in range(n):
        yi = Y[i,:]
        for j in range(m):
            Z[j] = yi
        print(Z)
        print(X)
        D[:,i] = np.sum(np.abs(X-Z),axis=1)
        print(np.abs(X-Z))
    return D

def distCosine(X,Y):
    # print(X.dtype)
    # if( ~isa(X,'double') or ~isa(Y,'double')):
    #   error( 'Inputs must be of type double')

    x = np.shape(X)
    p = x[1]
    X1 = np.zeros(x)
    XX = np.sqrt(np.sum(X*X,axis=1))
    for i in range(p):
        X1[:,i] = XX
    X = X/X1
    y=np.shape(Y)
    Y1 = np.zeros(y)
    YY = np.sqrt(np.sum(Y*Y,axis=1))
    for i in range(p):
        Y1[:,i] = YY
    Y = Y/Y1
    D = 1 - np.dot(X,Y.T)
    return D

def distEmd(X,Y):
    Xcdf = np.cumsum(X,axis=1)
    Ycdf = np.cumsum(Y,axis=1)
    x = np.shape(X)
    m = x[0]
    n = np.shape(Y)[0]
    ycdfRep = np.zeros(x)
    D = np.zeros((m,n))
    for i in range(n):
      ycdf = Ycdf[i,:]
      print(vars())
      for j in range(m):
          ycdfRep[j] = ycdf
      D[:,i] = np.sum(np.abs(Xcdf - ycdfRep),axis=1)
    return D

def distChiSq(X,Y,x,y):
# supposedly it's possible to implement this without a loop!
    yiRep = np.zeros(x)
    D = np.zeros((x[0],y[0]))
    for i in range(y[0]):
        yi = Y[i,:]
        for j in range(x[0]):
            yiRep[j] = yi
        s = yiRep + X
        d = yiRep - X
        D[:,i] = np.sum( d**2 / (s+2**(-52)), axis=1)/2
    return D

def distEucSq(X,Y,x,y):

    #if( ~isa(X,'double') or ~isa(Y,'double'))
     # error( 'Inputs must be of type double') end
    YYRep = np.zeros((x[0],y[0]))
    XXRep = np.zeros((x[0],y[0]))
    #Yt = Y'
    XX = np.sum(X*X,axis=1)
    YY = np.sum(Y*Y,axis=1).T
    print(vars())
    for j in range(y[0]):
        XXRep[:,j] = XX
    for j in range(x[0]):
        YYRep[j] = YY
    D = XXRep + YYRep - 2*np.dot(X,Y.T)

    return D

# X = np.array([[1,0,4],[2,3,5],[6,4,2]])
# Y = np.array([[0,1,2],[1,0,2]])
# x = np.shape(X)
# y = np.shape(Y)
# print("L1:",distL1(X,Y,x,y))
# print("Cosine:",distCosine(X,Y,x,y))
# print("Emd:",distEmd(X,Y,x,y))
# print("ChiSq:",distChiSq(X,Y,x,y))
# print("EucSq:",distEucSq(X,Y,x,y))
import numpy as np
from scipy.io import loadmat
data = loadmat("MSR_Python_ori.mat")
trainset = data["trainset"][0]
trainsetdata = data["trainsetdata"]
trainsetdatanum = data["trainsetdatanum"][0][0]
trainsetdatalabel = data["trainsetdatalabel"][0]
trainsetnum = data["trainsetnum"][0]

testsetdata = data["testsetdata"][0]
testsetdatanum = data["testsetdatanum"][0][0]
testsetdatalabel = data["testsetdatalabel"][0]
print("trainset:",trainset.shape)
print("trainset:",trainset[0][0].shape)
print("trainsetdata:",trainsetdata.shape)
print("trainsetdatanum:",trainsetdatanum)
print("trainsetdatalabel:",trainsetdatalabel.shape)
print("trainsetnum:",trainsetnum.shape)
print("testsetdata:",testsetdata.shape)
print("testsetdatanum:",testsetdatanum)
print("testsetdatalabel:",testsetdatalabel.shape)

# trainset: (20,)
# trainset: (15,)
# trainsetdata: (284, 284)
# trainsetdatanum: 284
# trainsetdatalabel: (284,)
# trainsetnum: (20,)
# testsetdata: (273,)
# testsetdatanum: 273
# testsetdatalabel: (273,)