import numpy as np
from scipy.io import loadmat
import time
from NNClassifier import *
from RVSML_OT_Learning import *
import logging

charnum = 4
classnum = charnum
dim = 2
nperclass = 10
CVAL = 1

 # add path
# addpath('/usr/local/Cellar/vlfeat-0.9.21/toolbox')
# vl_setup()
# addpath('libsvm-3.20/matlab')

delta = 1
lambda1 = 50
lambda2 = 0.1
max_iters = 10
err_limit = 10**(-6)

class Options:
    def __init__(self, max_iters, err_limit, lambda1, lambda2, delta):
        self.max_iters = max_iters
        self.err_limit = err_limit
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.delta = delta

options = Options(max_iters,err_limit,lambda1,lambda2,delta)

# data = loadmat("MSR_Python_ori.mat")

# trainset = data["trainset"][0]
trainset = [0]*classnum
trainset[0] = [np.array([np.linspace(0,100,10+i), np.linspace(0,100,10+i)]).reshape(dim,10+i).T for i in range(nperclass)]
trainset[1] = [np.array([np.linspace(0,-100,10+i), np.linspace(0,100,10+i)]).reshape(dim,10+i).T for i in range(nperclass)]
trainset[2] = [np.array([np.linspace(0,-100,10+i), np.linspace(0,-100,10+i)]).reshape(dim,10+i).T for i in range(nperclass)]
trainset[3] = [np.array([np.linspace(0,100,10+i), np.linspace(0,-100,10+i)]).reshape(dim,10+i).T for i in range(nperclass)]

# trainsetdatanum = data["trainsetdatanum"][0][0]
trainsetdatanum = 40
# trainsetdatalabel = data["trainsetdatalabel"][0]
trainsetdatalabel = [1]*nperclass+[2]*nperclass+[3]*nperclass+[4]*nperclass
# trainsetnum = data["trainsetnum"][0]
trainsetnum = [10]*4
# testsetdata = data["testsetdata"][0]
# testsetdata = [0]*classnum
testsetdata = [np.array([np.linspace(0,100,20+i), np.linspace(0,100,20+i)]).reshape(dim,20+i).T for i in range(nperclass)]
testsetdata += [np.array([np.linspace(0,-100,20+i), np.linspace(0,100,20+i)]).reshape(dim,20+i).T for i in range(nperclass)]
testsetdata += [np.array([np.linspace(0,-100,20+i), np.linspace(0,-100,20+i)]).reshape(dim,20+i).T for i in range(nperclass)]
testsetdata += [np.array([np.linspace(0,100,20+i), np.linspace(0,-100,20+i)]).reshape(dim,20+i).T for i in range(nperclass)]

# testsetdatanum = data["testsetdatanum"][0][0]
testsetdatanum = 40
# testsetdatalabel = data["testsetdatalabel"][0]
testsetdatalabel = [1]*10+[2]*10+[3]*10+[4]*10
trainset_m = trainset
testsetdata_m = testsetdata
testsetlabel = testsetdatalabel


print("data lode done")

print("OPW start")
templatenum = 8
lambda0 = 0.01
tic = time.time()
L, v_s_opw = RVSML_OT_Learning(trainset,templatenum,lambda0,options)
RVSML_opw_time = time.time() - tic

# v_s_opw = np.array([v[0] for v in v_s_opw])
# real_v_opw = np.linalg.solve(L.T,v_s_opw.T)

print("OPW lerning done")
## classification with the learned metric
# print("Classification start")
traindownset = [0]*classnum
testdownsetdata = [0]*testsetdatanum
for j in range(classnum):
    traindownset[j] = [0]*trainsetnum[j]
    for m in range(trainsetnum[j]):
        traindownset[j][m] = np.dot(trainset[j][m] ,L)

for j in range(testsetdatanum):
    testdownsetdata[j] = np.dot(testsetdata[j], L)

RVSML_opw_macro,RVSML_opw_micro,RVSML_opw_acc,opw_knn_average_time = NNClassifier(classnum,traindownset,trainsetnum,testdownsetdata,testsetdatanum,testsetlabel,options)
RVSML_opw_acc_1 = RVSML_opw_acc[0]
print("OPW Classification done")

print("OPW done")
print("DTW start")

templatenum = 8
lambda0 = 0.1
tic = time.time()
L, v_s_dtw = RVSML_OT_Learning_dtw(trainset,templatenum,lambda0,options)
RVSML_dtw_time = time.time() - tic

# v_s_dtw = np.array([v[0] for v in v_s_dtw])
# real_v_dtw = np.linalg.solve(L.T,v_s_dtw)
print("dtw learning done")
## classification with the learned metric
traindownset = [0]*classnum
testdownsetdata = [0]*testsetdatanum
for j in range(classnum):
    traindownset[j] = [0]*trainsetnum[j]
    for m in range(trainsetnum[j]):
        traindownset[j][m] = np.dot(trainset[j][m] ,L)

for j in range(testsetdatanum):
    testdownsetdata[j] = np.dot(testsetdata[j], L)

RVSML_dtw_macro,RVSML_dtw_micro,RVSML_dtw_acc,dtw_knn_average_time = NNClassifier_dtw(classnum,traindownset,trainsetnum,testdownsetdata,testsetdatanum,testsetlabel,options)
RVSML_dtw_acc_1 = RVSML_dtw_acc[0]
# logger.debug(vars())

print('Training time of RVSML instantiated by DTW is {:.4f} \n'.format(RVSML_dtw_time))
print('Classification using 1 nearest neighbor classifier with DTW distance:\n')
print('MAP macro is {:.4f}, micro is {:.4f} \n'.format(RVSML_dtw_macro, RVSML_dtw_micro))
print('dtw_knn_average_time is {:.4f} \n'.format(dtw_knn_average_time))
print('Accuracy is {:.4f} \n'.format(RVSML_dtw_acc_1))
for i,v in enumerate(v_s_dtw):
    print('ラベル{}:{}'.format(i+1,v))

print('Training time of RVSML instantiated by OPW is {:.4f} \n'.format(RVSML_opw_time))
print('Classification using 1 nearest neighbor classifier with OPW distance:\n')
print('MAP macro is {:.4f}, MAP micro is {:.4f} \n'.format(RVSML_opw_macro, RVSML_opw_micro))
# print('Accuracy is .4f \n',RVSML_opw_acc_1)
print('opw_knn_average_time is {:.4f} \n'.format(opw_knn_average_time))
print('Accuracy is {:.4f} \n'.format(RVSML_opw_acc_1))
for i,v in enumerate(v_s_opw):
    print('ラベル{}:{}'.format(i+1,v))


# print("debug")