import numpy as np
from scipy.io import loadmat
import time
from rvsml.run_RVSML import run_RVSML
import logging,pickle

name = 'MSRAction3D'

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
        self.lambda0 = 0.01
        self.lambda1 = 50
        self.lambda2 = 0.1
        self.delta = 1
        self.method = 'opw'
        self.classify = 'knn'
        self.templatenum = 4

class Dataset:
    def __init__(self):
        self.dataname = 'MSRAction3D'
        self.classnum = 20
        self.trainsetdatanum = 284
        self.testsetdatanum = 273
        self.dim = 60
        self.ClassLabel = np.arange(self.classnum).T+1
        self.trainsetdatalabel = 0
        self.testsetdatalabel = 0

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

options = Options()
dataset = Dataset()

pickle_list = ['trainsetnum','trainset','trainsetdata','trainsetdatalabel','testsetdata','testsetdatalabel']

for v in pickle_list:
    exec("with open('MSRAction3D/data/{}.bin','rb') as f: dataset.{} = pickle.load(f)".format(v,v))
dataset.getlabelfull()

run_RVSML(dataset,options)