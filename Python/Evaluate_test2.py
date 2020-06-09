import numpy as np
from scipy.io import loadmat
import time,os
from rvsml.run_RVSML import run_RVSML
import logging
from random import sample,randint
from itertools import product
import copy

class Options:
    def __init__(self):
        self.max_iters = 100
        self.err_limit = 10**(-3)
        self.lambda0 = 0.01
        self.lambda1 = 50
        self.lambda2 = 0.5
        self.delta = 1
        self.method = 'dtw'
        self.classify = 'knn'
        self.cpu_count = os.cpu_count()

class Dataset:
    def __init__(self,N=1,V=1,T=1):
        self.dataname = 'test'
        self.classnum = 2
        self.trainsetnum = [V]*self.classnum
        self.trainsetdatanum = sum(self.trainsetnum)
        self.testsetdatanum = T
        self.dim = N
        self.ClassLabel = np.arange(self.classnum).T+1
        self.getdata(N)

    def getLabel(self,classid):
        p = int(max(classid))
        # logger.info(p)
        X = np.zeros((np.size(classid),p))-1
        for i in range(p):
            indx = np.nonzero(classid == i+1)
            X[indx,i] = 1
        return X

    def getdata(self,N):
        self.trainset = [[0]*self.trainsetnum[i] for i in range(2)]
        self.trainsetdata = []
        self.trainsetdatalabel = []
        for c in range(self.classnum):
            for v in range(self.trainsetnum[c]):
                self.trainset[c][v] = np.zeros((N*(c+1),N))
                for i in range(N):
                    for j in range(c+1):
                        self.trainset[c][v][(c+1)*i+j,i]=1
                self.trainsetdata.append(self.trainset[c][v])
                self.trainsetdatalabel.append(c+1)

        self.testsetdata = copy.deepcopy(self.trainsetdata)
        self.testsetdatalabel = copy.deepcopy(self.trainsetdatalabel)
        self.testsetdatanum = self.trainsetdatanum

        # for i in range(T):
        #     cla = randint(1,2)
        #     lis = sample(range(M),k=M)
        #     for j,l in enumerate(lis):
        #         self.testsetdata[i][l,j+N*(cla-1)]=1
        #     self.testsetdatalabel[i] = cla
        
        self.trainsetlabelfull = self.getLabel(self.trainsetdatalabel)
        self.testsetlabelfull = self.getLabel(self.testsetdatalabel)    
        return

options = Options()
dataset = Dataset()

logger = logging.getLogger('{}Log'.format(dataset.dataname)) # ログの出力名を設定
logger.setLevel(20) # ログレベルの設定
logger.addHandler(logging.StreamHandler()) # ログのコンソール出力の設定
logging.basicConfig(filename='test/{}.log'.format(dataset.dataname), format="%(message)s", filemode='w') # ログのファイル出力先を設定

T=5 #テストデータ数
N=2 #次元数
# M=50 #時系列長
V=5 #学習データ数

dataset = Dataset(N,V,T)
logger.info("data load done")

options.templatenum = 5
run_RVSML(dataset,options)
