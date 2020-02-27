import numpy as np
from scipy.io import loadmat
import time
from NNClassifier import *
from RVSML_OT_Learning import *
import logging
import h5py
from array import array

charnum = 20
classnum = charnum
dim = 100
#rankdim = 58
CVAL = 1


# addpath('E:/BING/ActionRecognition/FrameWideFeatures/libsvm-3.20/matlab')

delta = 1
lambda1 = 50
lambda2 = 0.1
max_iters = 10
err_limit = 10**(-2)

class Options:
    def __init__(self, max_iters, err_limit, lambda1, lambda2, delta):
        self.max_iters = max_iters
        self.err_limit = err_limit
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.delta = delta

options = Options(max_iters,err_limit,lambda1,lambda2,delta)

matlist = ["trainset", "trainsetnum","testset",
            "testsetnum","testsetdata","testsetlabel",
            "testsetdatanum","traindatamean"]

# for name in matlist:
#     filepath = './datamat/{}.mat'.format(name)
#     arrays = {}
#     with h5py.File(filepath, 'r') as f:
#         for k, v in f.items():
#             arrays[k] = np.array(v)
#     exec("%s = %s" % (name, arrays))

for name in matlist:
    filepath = './datamat/{}.mat'.format(name)
    exec("%s = loadmat('%s')['%s']" % (name, filepath,name))

testsetlabel = testsetlabel.T

for name in matlist:
    exec("%s = %s[0]" % (name, name))

testsetdatanum = testsetdatanum[0]

trainset_m = trainset

shape=trainset_m[0].shape

for c in range(classnum):
    for m in range(trainsetnum[c]):
        trainset_m[c][0][m] = trainset[c][0][m] - traindatamean

testsetdata_m = testsetdata

for m in range(testsetdatanum):
    testsetdata_m[m] = testsetdata[m] - traindatamean

## RVSML-DTW
print("data lode done")

print(testsetdatanum)
# for name in matlist:
#     exec("print('%s:',%s.shape)" % (name,name))

# print("DTW start")
# templatenum = 4
# lambda0 = 0.0005
# tic = time.time()
# L = RVSML_OT_Learning_dtw(trainset_m,templatenum,lambda0,options)
# RVSML_dtw_time = time.time() - tic
# print("dtw learning done")
# ## classification with the learned metric
# traindownset = [0]*classnum
# testdownsetdata = [0]*testsetdatanum
# for j in range(classnum):
#     traindownset[j] = [0]*trainsetnum[j]
#     for m in range(trainsetnum[j]):
#         traindownset[j][m] = np.dot(trainset[j][0][m] ,L)
#
# for j in range(testsetdatanum):
#     testdownsetdata[j] = np.dot(testsetdata[j], L)
#
# RVSML_dtw_macro,RVSML_dtw_micro,RVSML_dtw_acc,dtw_knn_average_time = NNClassifier_dtw(classnum,traindownset,trainsetnum,testdownsetdata,testsetdatanum,testsetlabel,options)
# RVSML_dtw_acc_1 = RVSML_dtw_acc[0]
#
# ## RVSML-OPW
# templatenum = 4
# lambda0 = 0.00005
# tic = time.time()
# L = RVSML_OT_Learning(trainset_m,templatenum,lambda0,options)
# RVSML_opw_time = time.time() - tic
# print("OPW lerning done")
#
# ## classification with the learned metric
# # print("Classification start")
# traindownset = [0]*classnum
# testdownsetdata = [0]*testsetdatanum
# for j in range(classnum):
#     traindownset[j] = [0]*trainsetnum[j]
#     for m in range(trainsetnum[j]):
#         traindownset[j][m] = np.dot(trainset[j][0][m] ,L)
#
# for j in range(testsetdatanum):
#     testdownsetdata[j] = np.dot(testsetdata[j], L)
#
# RVSML_opw_macro,RVSML_opw_micro,RVSML_opw_acc,opw_knn_average_time = NNClassifier(classnum,traindownset,trainsetnum,testdownsetdata,testsetdatanum,testsetlabel,options)
# # RVSML_opw_acc_1 = RVSML_opw_acc[0]
#
#
# print('Training time of RVSML instantiated by DTW is {:.4f} \n'.format(RVSML_dtw_time))
# print('Classification using 1 nearest neighbor classifier with DTW distance:\n')
# print('MAP macro is {:.4f}, micro is {:.4f} \n'.format(RVSML_dtw_macro, RVSML_dtw_micro))
#
# for acc in RVSML_dtw_acc:
#     print('Accuracy is {:.4f} \n'.format(acc))
#
# print('Training time of RVSML instantiated by OPW is {:.4f} \n'.format(RVSML_opw_time))
# print('Classification using 1 nearest neighbor classifier with OPW distance:\n')
# print('MAP macro is {:.4f}, MAP micro is {:.4f} \n'.format(RVSML_opw_macro, RVSML_opw_micro))
# # print('Accuracy is .4f \n',RVSML_opw_acc_1)
#
# for acc in RVSML_opw_acc:
#     print('Accuracy is {:.4f} \n'.format(acc))