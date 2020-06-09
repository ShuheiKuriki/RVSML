import numpy as np
from scipy.io import loadmat
import time
from NNClassifier import *
from RVSML_OT_Learning import *
import logging
from array import array

charnum = 20
classnum = charnum
dim = 100
#rankdim = 58
CVAL = 1

method = 'opw'
logger = logging.getLogger('ChaLearn{}Log'.format(method))

logger.setLevel(10)

sh = logging.StreamHandler()
logger.addHandler(sh)

logging.basicConfig(filename='ChaLearn_{}.log'.format(method), format="%(message)s", filemode='w')

delta = 1
lambda1 = 50
lambda2 = 0.5
lambda0 = 0.0005
max_iters = 1000
err_limit = 10**(-2)

class Options:
    def __init__(self, max_iters, err_limit, lambda0, lambda1, lambda2, delta, method):
        self.max_iters = max_iters
        self.err_limit = err_limit
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.delta = delta
        self.method = method

options = Options(max_iters,err_limit,lambda0,lambda1,lambda2,delta,method)

class Datainfo:
    def __init__(self,train,test,classnum,dim):
        self.train = train
        self.test = test
        self.classnum = classnum
        self.dim = dim
        self.ClassLabel = np.arange(classnum).T+1

    def centrize(self,traindatamean):
        trainset_m = self.train.dset
        for c in range(self.classnum):
            for m in range(self.train.setnum[c]):
                trainset_m[c][0][m] = self.train.dset[c][0][m] - traindatamean
        trainsetdata_m = self.train.setdata
        for m in range(self.train.setdatanum):
            trainsetdata_m[m] = self.train.setdata[m] - traindatamean
        testsetdata_m = testsetdata
        for m in range(self.test.setdatanum):
            testsetdata_m[m] = self.test.setdata[m] - traindatamean
        self.train.set_m = trainset_m
        self.train.setdata_m = trainsetdata_m
        self.test.setdata_m = testsetdata_m

    def getdown(self,L):
        traindownsetdata = [0]*self.train.setdatanum
        for m in range(self.train.setdatanum):
            traindownsetdata[m] = np.dot(self.train.setdata_m[m],L)
        self.train.downsetdata = traindownsetdata
        testdownsetdata = [0]*self.test.setdatanum
        for j in range(self.test.setdatanum):
            testdownsetdata[j] = np.dot(self.test.setdata_m[j], L)
        self.test.downsetdata = testdownsetdata

class Dataset:
    def __init__(self,dset=None,setnum=None,setlabel=None,setdata=None,setdatanum=None,setdatalabel=None):
        #setはclass別に分かれている,setdataは全部まとまっている
        self.dset = dset
        self.setnum = setnum
        self.setlabel = setlabel
        if setdata is None or setdatalabel is None:
            setdata,setdatalabel = [],[]
            for c in range(len(dset)):
                for i in range(len(dset[c][0])): 
                    setdata.append(dset[c][0][i])
                    setdatalabel.append(c+1)
        self.setdata = setdata
        self.setdatalabel = setdatalabel
        if setdatanum is None:
            setdatanum = np.sum(setnum)
        self.setdatanum = setdatanum
        self.setlabelfull = self.getLabel(setdatalabel)

    def getLabel(self,classid):
        p = int(max(classid))
        # logger.info(p)
        X = np.zeros((np.size(classid),p))-1
        for i in range(p):
            indx = np.nonzero(classid == i+1)
            X[indx,i] = 1
        return X

#matlab -v7で保村しなおさないと読み込めない
trainset = loadmat('./datamat/trainset.mat')['trainset'][0]
trainsetnum = loadmat('./datamat/trainsetnum.mat')['trainsetnum'][0]
testset = loadmat('./datamat/testset.mat')['testset'][0]
testsetnum = loadmat('./datamat/testsetnum.mat')['testsetnum'][0]
testsetdata = loadmat('./datamat/testsetdata.mat')['testsetdata'][0]
testsetdatalabel = loadmat('./datamat/testsetlabel.mat')['testsetlabel'].T[0]
testsetdatanum = loadmat('./datamat/testsetdatanum.mat')['testsetdatanum'][0][0]
traindatamean = loadmat('./datamat/traindatamean.mat')['traindatamean'][0]

train = Dataset(dset=trainset,setnum=trainsetnum)
test = Dataset(dset=testset,setnum=testsetnum,setdata=testsetdata,setdatanum=testsetdatanum,setdatalabel=testsetdatalabel)
datainfo = Datainfo(train=train,test=test,classnum=classnum,dim=dim)

datainfo.centrize(traindatamean)
logger.info("learning start")

options.templatenum = 4
tic = time.time()
L = RVSML_OT_Learning(datainfo,options)
RVSML_time = time.time() - tic

logger.info("learning done")

datainfo.getdown(L)

macro,micro,acc,knn_time,knn_average_time = NNClassifier(datainfo,options)

logger.info('Training time is {:.4f} \n'.format(RVSML_time))
logger.info('Classification using 1 nearest neighbor classifier:\n')
logger.info('MAP macro is {:.4f}, micro is {:.4f} \n'.format(macro, micro))
logger.info('knn_average_time is {:.4f} \n'.format(knn_average_time))
logger.info('knn_total_time is {:.4f} \n'.format(knn_time))

for ac in acc:
    logger.info('Accuracy is {:.4f} \n'.format(ac))