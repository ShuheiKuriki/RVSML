import numpy as np
import time,logging,pickle
from rvsml.run_RVSML import run_RVSML

class Options:
    def __init__(self):
        self.max_iters, self.err_limit = 1000, 10**(-8)
        self.lambda0, self.lambda1, self.lambda2 = 0.0005, 50, 0.5
        self.delta = 1
        self.method, self.classify = 'opw', 'knn'
        self.templatenum = 4

class Dataset:
    def __init__(self):
        self.dataname = 'ChaLearn'
        self.classnum, self.dim = 20, 100
        self.trainsetdatanum, self.testsetdatanum = 6850, 3454
        self.ClassLabel = np.arange(self.classnum).T+1
        self.trainsetdatalabel, self.testsetdatalabel = [0],[0]
        self.trainset, self.trainsetdata = [0], [0]
        self.trainsetnum, self.traindatamean = [0], 0
        self.testsetdata = [0]

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
        trainsetdata_m = self.trainsetdata
        testsetdata_m = self.testsetdata
        for c in range(self.classnum):
            for m in range(self.trainsetnum[c]):
                trainset_m[c][m] = self.trainset[c][m] - traindatamean
        for m in range(self.trainsetdatanum):
            trainsetdata_m[m] = self.trainsetdata[m] - traindatamean
        for m in range(self.testsetdatanum):
            testsetdata_m[m] = self.testsetdata[m] - traindatamean
        self.trainset_m = trainset_m
        self.trainsetdata_m = trainsetdata_m
        self.testsetdata_m = testsetdata_m

options = Options()
dataset = Dataset()

logger = logging.getLogger('{}Log'.format(dataset.dataname)) # ログの出力名を設定
logger.setLevel(20) # ログレベルの設定
logger.addHandler(logging.StreamHandler()) # ログのコンソール出力の設定
logging.basicConfig(filename='CharLearn/{}.log'.format(dataset.dataname), format="%(message)s", filemode='w') # ログのファイル出力先を設定

pickle_list = ['trainsetnum','trainset','trainsetdatalabel','trainsetdatanum','trainsetdata','testsetdata','testsetdatalabel','traindatamean']
for v in pickle_list:
    exec("with open('ChaLearn/data/{}.bin','rb') as f: dataset.{} = pickle.load(f)".format(v,v))

dataset.getlabelfull()
dataset.centrize(dataset.traindatamean)

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