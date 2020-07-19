import numpy as np
from rvsml.run_RVSML import run_RVSML
import logging,pickle,time,os
np.set_printoptions(precision=3,suppress=True)
class Options:
    def __init__(self):
        self.max_iters, self.err_limit = 100, 10**(-4)
        self.method, self.classify  = 'dtw', 'knn'
        self.init = 'uniform'
        if self.method == 'dtw':
            self.lambda0 = 0.1
        if self.method == 'opw':
            self.lambda0, self.lambda1, self.lambda2 = 0.01, 50, 1
            self.delta = 1
        self.templatenum = 4
        self.cpu_count = os.cpu_count()//2

class Dataset:
    def __init__(self):
        self.dataname = 'MSRAction3D'
        self.classnum, self.dim = 20, 60
        self.trainsetdatanum, self.testsetdatanum  = 284, 273
        self.ClassLabel = np.arange(self.classnum).T+1
        self.trainsetdatalabel, self.testsetdatalabel = [0], [0]

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
logging.basicConfig(filename='MSRAction3D/{}.log'.format(dataset.dataname), format="%(message)s", filemode='w') # ログのファイル出力先を設定

pickle_list = ['trainsetnum','trainset','trainsetdata','trainsetdatalabel','testsetdata','testsetdatalabel']
for v in pickle_list:
    exec("with open('MSRAction3D/data/{}.bin','rb') as f: dataset.{} = pickle.load(f)".format(v,v))

dataset.getlabelfull()

run_RVSML(dataset,options)