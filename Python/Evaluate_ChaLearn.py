import numpy as np
from scipy.io import loadmat
import time,logging,pickle
from rvsml.run_RVSML import run_RVSML

name = 'ChaLearn'

# ログの出力名を設定（1）
logger = logging.getLogger('{}Log'.format(name))

# ログレベルの設定（2）
logger.setLevel(20)

# ログのコンソール出力の設定（3）
sh = logging.StreamHandler()
logger.addHandler(sh)

# ログのファイル出力先を設定（4）
logging.basicConfig(filename='{}.log'.format(name), format="%(message)s", filemode='w')

class Options:
    def __init__(self):
        self.max_iters = 1000
        self.err_limit = 10**(-8)
        self.lambda0 = 0.0005
        self.lambda1 = 50
        self.lambda2 = 0.5
        self.delta = 1
        self.method = 'opw'
        self.classify = 'knn'
        self.templatenum = 4

class Dataset:
    def __init__(self):
        self.dataname = 'ChaLearn'
        self.classnum = 20
        self.trainsetdatanum = 6850
        self.testsetdatanum = 3454
        self.dim = 100
        self.ClassLabel = np.arange(self.classnum).T+1

    def getlabelfull(self):
        self.trainsetlabelfull = self.getLabel(self.trainsetdatalabel)
        self.testsetlabelfull = self.getLabel(self.testsetdatalabel)
        return

    def getLabel(self,classid):
        p = int(max(classid))
        # logger.info(p)
        X = np.zeros((np.size(classid),p))-1
        for i in range(p):
            indx = np.nonzero(classid == i+1)
            X[indx,i] = 1
        return X

    def centrize(self,traindatamean):
        trainset_m = self.trainset
        for c in range(self.classnum):
            for m in range(self.trainsetnum[c]):
                trainset_m[c][m] = self.trainset[c][m] - traindatamean
        trainsetdata_m = self.trainsetdata
        for m in range(self.trainsetdatanum):
            trainsetdata_m[m] = self.trainsetdata[m] - traindatamean
        testsetdata_m = self.testsetdata
        for m in range(self.testsetdatanum):
            testsetdata_m[m] = self.testsetdata[m] - traindatamean
        self.trainset_m = trainset_m
        self.trainsetdata_m = trainsetdata_m
        self.testsetdata_m = testsetdata_m

options = Options()
dataset = Dataset()


pickle_list = ['trainsetnum','trainset','trainsetdatalabel','trainsetdatanum','trainsetdata','testsetdata','testsetdatalabel','traindatamean']

for v in pickle_list:
    exec("with open('ChaLearn/data/{}.bin','rb') as f: dataset.{} = pickle.load(f)".format(v,v))
dataset.getlabelfull()
dataset.centrize(dataset.traindatamean)

options.templatenum = 4

run_RVSML(dataset,options)

#matlab -v7で保村しなおさないと読み込めない
# trainset = loadmat('ChaLearn/datamat/trainset.mat')['trainset'][0]
# trainsetnum = loadmat('ChaLearn/datamat/trainsetnum.mat')['trainsetnum'][0]
# testset = loadmat('ChaLearn/datamat/testset.mat')['testset'][0]
# testsetnum = loadmat('ChaLearn/datamat/testsetnum.mat')['testsetnum'][0]
# testsetdata = loadmat('ChaLearn/datamat/testsetdata.mat')['testsetdata'][0]
# testsetdatalabel = loadmat('ChaLearn/datamat/testsetlabel.mat')['testsetlabel'].T[0]
# testsetdatanum = loadmat('ChaLearn/datamat/testsetdatanum.mat')['testsetdatanum'][0][0]
# traindatamean = loadmat('ChaLearn/datamat/traindatamean.mat')['traindatamean'][0]
# trainsetdatanum = sum(trainsetnum)

# trainset_py = [0]*20
# trainsetdata_py = []
# trainsetdatalabel = []
# for c in range(20):
#     trainset_py[c] = [0]*trainsetnum[c]
#     for m in range(trainsetnum[c]):
#         trainset_py[c][m] = trainset[c][0][m]
#         trainsetdata_py.append(trainset[c][0][m])
#         trainsetdatalabel.append(c+1)
# testsetdata_py = [0]*testsetdatanum
# for m in range(testsetdatanum):
#     testsetdata_py[m] = testsetdata[m]

# trainset = trainset_py
# trainsetdata = trainsetdata_py
# testsetdata = testsetdata_py
# for v in pickle_list:
#     exec("with open('ChaLearn/data/{}.bin','wb') as f: pickle.dump({},f)".format(v,v))