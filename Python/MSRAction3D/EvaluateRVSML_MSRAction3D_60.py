import numpy as np
from scipy.io import loadmat
import time
from NNClassifier import *
from RVSML_OT_Learning import *
import logging

# ログの出力名を設定（1）
logger = logging.getLogger('MSRAction3DLog')

# ログレベルの設定（2）
logger.setLevel(20)

# ログのコンソール出力の設定（3）
sh = logging.StreamHandler()
logger.addHandler(sh)

# ログのファイル出力先を設定（4）
logging.basicConfig(filename='MSRAction3D.log', format="%(message)s", filemode='w')

charnum = 20
classnum = charnum
dim = 60
CVAL = 1

delta = 1
lambda0 = 0.01
lambda1 = 50
lambda2 = 0.5
max_iters = 100
err_limit = 10**(-6)

class Options:
    def __init__(self, max_iters, err_limit, lambda0, lambda1, lambda2, delta):
        self.max_iters = max_iters
        self.err_limit = err_limit
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.delta = delta

class Dataset:
    def __init__(self,data):
        self.trainset = data["trainset"]
        self.trainsetnum = data["trainsetnum"]
        self.trainsetdata = data["trainsetdata"]
        self.trainsetdatanum = int(data["trainsetdatanum"])
        self.trainsetdatalabel = data["trainsetdatalabel"]
        self.trainsetlabelfull = self.getLabel(self.trainsetdatalabel)
        self.testsetdata = data["testsetdata"]
        self.testsetdatanum = int(data["testsetdatanum"])
        self.testsetdatalabel = data["testsetdatalabel"]
        self.testsetlabelfull = self.getLabel(self.testsetdatalabel)
        self.classnum = int(data["classnum"])
        self.dim = int(data["dim"])
        self.ClassLabel = data["ClassLabel"]

    def getLabel(self,classid):
        p = int(max(classid))
        # logger.info(p)
        X = np.zeros((np.size(classid),p))-1
        for i in range(p):
            indx = np.nonzero(classid == i+1)
            X[indx,i] = 1
        return X

options = Options(max_iters,err_limit,lambda0,lambda1,lambda2,delta)

# data = loadmat("MSR_Python_ori.mat")

# trainset = data["trainset"][0]
# trainsetdata = data["trainsetdata"]
# trainsetdatanum = data["trainsetdatanum"][0][0]
# trainsetdatalabel = data["trainsetdatalabel"][0]
# trainsetnum = data["trainsetnum"][0]

# testsetdata = data["testsetdata"][0]
# testsetdatanum = data["testsetdatanum"][0][0]
# testsetdatalabel = data["testsetdatalabel"][0]

# trainset_m = trainset
# testsetdata_m = testsetdata


# ClassLabel = np.arange(classnum).T+1

# trainset_py = [0]*classnum
# trainsetdata_py = [0]*trainsetdatanum
# testsetdata_py = [0]*testsetdatanum
# trainsetdatanum = int(trainsetdatanum)
# testsetdatanum = int(testsetdatanum)


# for i in range(classnum):
#     trainset_py[i] = trainset[i][0]
# for i in range(trainsetdatanum):
#     trainsetdata_py[i] = trainsetdata[i][0]
# for i in range(testsetdatanum):
#     testsetdata_py[i] = testsetdata[i]


# np.savez('MSRAction3D',trainset=trainset_py,trainsetdata=trainsetdata_py,trainsetdatanum=trainsetdatanum,trainsetdatalabel=trainsetdatalabel,trainsetnum=trainsetnum,testsetdata=testsetdata_py,testsetdatanum=testsetdatanum,testsetdatalabel=testsetdatalabel,ClassLabel=ClassLabel,dim=dim,classnum=classnum)
data = np.load('MSRAction3D.npz',allow_pickle=True)

dataset = Dataset(data)
logger.info("data load done")

# logger.info("OPW start")

# options.method='opw'
# templatenum = 4
# tic = time.time()
# L = RVSML_OT_Learning(dataset,templatenum,options,method='opw')
# RVSML_opw_time = time.time() - tic
# logger.info("OPW lerning done")
# ## classification with the learned metric
# # print("Classification start")
# traindownset = [0]*classnum
# traindownsetdata = []
# testdownsetdata = [0]*dataset.testsetdatanum
# for j in range(classnum):
#     traindownset[j] = [0]*dataset.trainsetnum[j]
#     for m in range(dataset.trainsetnum[j]):
#         downdata = np.dot(dataset.trainset[j][m],L)
#         traindownset[j][m] = downdata
#         traindownsetdata.append(downdata)

# for j in range(dataset.testsetdatanum):
#     testdownsetdata[j] = np.dot(dataset.testsetdata[j], L)

# dataset.traindownset = traindownset
# dataset.traindownsetdata = traindownsetdata
# dataset.testdownsetdata = testdownsetdata

# RVSML_opw_macro,RVSML_opw_micro,RVSML_opw_acc,opw_knn_average_time = NNClassifier(dataset,options, method='opw')
# # RVSML_opw_acc_1 = RVSML_opw_acc[0]

# logger.info("OPW Classification done")

# logger.info("OPW done")
logger.info("DTW start")

options.method='dtw'
templatenum = 60
tic = time.time()
L = RVSML_OT_Learning(dataset,templatenum,options,method='dtw')
RVSML_dtw_time = time.time() - tic
logger.info("dtw learning done")
## classification with the learned metric
traindownset = [0]*classnum
traindownsetdata = []
testdownsetdata = [0]*dataset.testsetdatanum
for j in range(classnum):
    traindownset[j] = [0]*dataset.trainsetnum[j]
    for m in range(dataset.trainsetnum[j]):
        downdata = np.dot(dataset.trainset[j][m],L)
        traindownset[j][m] = downdata
        traindownsetdata.append(downdata)

for j in range(dataset.testsetdatanum):
    testdownsetdata[j] = np.dot(dataset.testsetdata[j], L)

dataset.traindownset = traindownset
dataset.traindownsetdata = traindownsetdata
dataset.testdownsetdata = testdownsetdata

RVSML_dtw_macro,RVSML_dtw_micro,RVSML_dtw_acc,dtw_knn_average_time = NNClassifier(dataset,options,method='dtw')
RVSML_dtw_acc_1 = RVSML_dtw_acc[0]

logger.info('Training time of RVSML instantiated by DTW is {:.4f} \n'.format(RVSML_dtw_time))
logger.info('Classification using 1 nearest neighbor classifier with DTW distance:\n')
logger.info('MAP macro is {:.4f}, micro is {:.4f} \n'.format(RVSML_dtw_macro, RVSML_dtw_micro))
logger.info('dtw_knn_average_time is {:.4f} \n'.format(dtw_knn_average_time))
logger.info('dtw_knn_total_time is {:.4f} \n'.format(dtw_knn_average_time*dataset.testsetdatanum))

for acc in RVSML_dtw_acc:
    logger.info('Accuracy is {:.4f} \n'.format(acc))

# logger.info('Training time of RVSML instantiated by OPW is {:.4f} \n'.format(RVSML_opw_time))
# logger.info('Classification using 1 nearest neighbor classifier with OPW distance:\n')
# logger.info('MAP macro is {:.4f}, MAP micro is {:.4f} \n'.format(RVSML_opw_macro, RVSML_opw_micro))
# # logger.info('Accuracy is .4f \n',RVSML_opw_acc_1)
# logger.info('opw_knn_average_time is {:.4f} \n'.format(opw_knn_average_time))
# logger.info('opw_knn_total_time is {:.4f} \n'.format(opw_knn_average_time*dataset.testsetdatanum))

# for acc in RVSML_opw_acc:
#     logger.info('Accuracy is {:.4f} \n'.format(acc))

# print("debug")