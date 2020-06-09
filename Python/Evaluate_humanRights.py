import numpy as np
from rvsml.run_RVSML import run_RVSML
import logging,pickle,time
import os

class Options:
    def __init__(self):
        self.max_iters, self.err_limit = 1000, 10**(-5)
        self.lambda0, self.lambda1, self.lambda2 = 0.01, 50, 0.1
        self.delta = 1
        self.method, self.classify  = 'opw', 'virtual'
        self.templatenum = 10
        self.cpu_count = min(os.cpu_count(),2)

class Dataset:
    def __init__(self,data=None):
        self.dataname = 'humanRights'
        self.langs = ['en','es']
        self.langnum, self.classnum, self.dim = 2, 30, 300
        self.trainsetdatanum = self.langnum * self.classnum
        self.trainsetnum = [self.langnum]*self.classnum
        self.testsetdatanum = self.trainsetdatanum
        # self.ClassLabel = np.arange(self.classnum).T+1
        self.trainsetdatalabel = [1+ i//self.langnum for i in range(self.trainsetdatanum)]
        self.testsetdatalabel = self.trainsetdatalabel
        if data is not None:
            self.trainsetdata,self.testsetdata = data,data
            trainset = [[0]*self.langnum for _ in range(self.classnum)]
            for c in range(self.classnum):
                for l in range(self.langnum):
                    trainset[c][l] = data[c*self.langnum+l]
            self.trainset = trainset

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

logger = logging.getLogger('{}Log'.format(dataset.dataname)) # ログの出力名を設定
logger.setLevel(20) # ログレベルの設定
logger.addHandler(logging.StreamHandler()) # ログのコンソール出力の設定
logging.basicConfig(filename='{}/{}.log'.format(dataset.dataname,dataset.dataname), format="%(message)s", filemode='w') # ログのファイル出力先を設定

data = []
for c in range(dataset.classnum):
    for l in dataset.langs:
        data.append(np.load('{}/vectorized_texts/{}/article{}.npy'.format(dataset.dataname,l,c+1)))

dataset = Dataset(data)
dataset.getlabelfull()

dataset = run_RVSML(dataset,options)
