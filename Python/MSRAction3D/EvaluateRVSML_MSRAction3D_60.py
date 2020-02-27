import numpy as np
from scipy.io import loadmat
import time
from NNClassifier import *
from RVSML_OT_Learning import *
import logging

# ログの出力名を設定（1）
# logger = logging.getLogger('LoggingTest')

# ログレベルの設定（2）
# logger.setLevel(20)

# ログのコンソール出力の設定（3）
# sh = logging.StreamHandler()
# logger.addHandler(sh)

# ログのファイル出力先を設定（4）
# fh = logging.FileHandler('test.log')
# logger.addHandler(fh)

charnum = 20
classnum = charnum
dim = 60
CVAL = 1

 # add path
# addpath('/usr/local/Cellar/vlfeat-0.9.21/toolbox')
# vl_setup()
# addpath('libsvm-3.20/matlab')

delta = 1
lambda1 = 50
lambda2 = 0.1
max_iters = 200
err_limit = 10**(-6)

class Options:
    def __init__(self, max_iters, err_limit, lambda1, lambda2, delta):
        self.max_iters = max_iters
        self.err_limit = err_limit
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.delta = delta

options = Options(max_iters,err_limit,lambda1,lambda2,delta)

data = loadmat("MSR_Python_ori.mat")

trainset = data["trainset"][0]
trainsetdata = data["trainsetdata"]
trainsetdatanum = data["trainsetdatanum"][0][0]
trainsetdatalabel = data["trainsetdatalabel"][0]
trainsetnum = data["trainsetnum"][0]

testsetdata = data["testsetdata"][0]
testsetdatanum = data["testsetdatanum"][0][0]
testsetdatalabel = data["testsetdatalabel"][0]

trainset_m = trainset
testsetdata_m = testsetdata
testsetlabel = testsetdatalabel

print("data lode done")

print("OPW start")
templatenum = 4
lambda0 = 0.01
tic = time.time()
L = RVSML_OT_Learning(trainset,templatenum,lambda0,options)
RVSML_opw_time = time.time() - tic
print("OPW lerning done")
## classification with the learned metric
# print("Classification start")
traindownset = [0]*classnum
testdownsetdata = [0]*testsetdatanum
for j in range(classnum):
    traindownset[j] = [0]*trainsetnum[j]
    for m in range(trainsetnum[j]):
        traindownset[j][m] = np.dot(trainset[j][0][m] ,L)

for j in range(testsetdatanum):
    testdownsetdata[j] = np.dot(testsetdata[j], L)

RVSML_opw_macro,RVSML_opw_micro,RVSML_opw_acc,opw_knn_average_time = NNClassifier(classnum,traindownset,trainsetnum,testdownsetdata,testsetdatanum,testsetlabel,options)
RVSML_opw_acc_1 = RVSML_opw_acc[0]
print("OPW Classification done")

print("OPW done")
print("DTW start")

templatenum = 4
lambda0 = 0.1
tic = time.time()
L = RVSML_OT_Learning_dtw(trainset,templatenum,lambda0,options)
RVSML_dtw_time = time.time() - tic
print("dtw learning done")
## classification with the learned metric
traindownset = [0]*classnum
testdownsetdata = [0]*testsetdatanum
for j in range(classnum):
    traindownset[j] = [0]*trainsetnum[j]
    for m in range(trainsetnum[j]):
        traindownset[j][m] = np.dot(trainset[j][0][m] ,L)

for j in range(testsetdatanum):
    testdownsetdata[j] = np.dot(testsetdata[j], L)

RVSML_dtw_macro,RVSML_dtw_micro,RVSML_dtw_acc,dtw_knn_average_time = NNClassifier_dtw(classnum,traindownset,trainsetnum,testdownsetdata,testsetdatanum,testsetlabel,options)
RVSML_dtw_acc_1 = RVSML_dtw_acc[0]
# logger.debug(vars())

print('Training time of RVSML instantiated by DTW is {:.4f} \n'.format(RVSML_dtw_time))
print('Classification using 1 nearest neighbor classifier with DTW distance:\n')
print('MAP macro is {:.4f}, micro is {:.4f} \n'.format(RVSML_dtw_macro, RVSML_dtw_micro))
print('dtw_knn_average_time is {:.4f} \n'.format(dtw_knn_average_time))

for acc in RVSML_dtw_acc:
    print('Accuracy is {:.4f} \n'.format(acc))

print('Training time of RVSML instantiated by OPW is {:.4f} \n'.format(RVSML_opw_time))
print('Classification using 1 nearest neighbor classifier with OPW distance:\n')
print('MAP macro is {:.4f}, MAP micro is {:.4f} \n'.format(RVSML_opw_macro, RVSML_opw_micro))
# print('Accuracy is .4f \n',RVSML_opw_acc_1)
print('opw_knn_average_time is {:.4f} \n'.format(opw_knn_average_time))

for acc in RVSML_opw_acc:
    print('Accuracy is {:.4f} \n'.format(acc))

# print("debug")