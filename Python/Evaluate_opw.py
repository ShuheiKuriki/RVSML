from rvsml.align import OPW_w
import argparse,logging,os
import numpy as np

np.set_printoptions(precision=3,suppress=True)
parser = argparse.ArgumentParser()
parser.add_argument("--lambda1", type=float, default=1, help="the parameter of the inverse difference moment")
parser.add_argument("--lambda2", type=float, default=1, help="the parameter of the standard distribution")
parser.add_argument("--delta", type=float, default=1, help="variance of the standard distribution")
params = parser.parse_args()

class Options:
    def __init__(self):
        self.max_iters, self.err_limit = 1000, 10**(-4)
        self.method = 'opw'
        self.lambda1, self.lambda2, self.delta = params.lambda1, params.lambda2, params.delta

class Dataset:
    def __init__(self):
        self.dataname = 'opw'
        self.dim = 5
        nums1 = [1,2,3,1,1]
        nums2 = [1,1,3,2,1]
        self.data1 = np.zeros((sum(nums1),self.dim))
        t = 0
        for i in range(self.dim):
            for _ in range(nums1[i]):
                self.data1[t,i] = 1
                t += 1
        self.data2 = np.zeros((sum(nums2),self.dim))
        t = 0
        for i in range(self.dim):
            for _ in range(nums2[i]):
                self.data2[t,i] = 1
                t += 1

options = Options()
data = Dataset()

logger = logging.getLogger('{}Log'.format(data.dataname)) # ログの出力名を設定
logger.setLevel(20) # ログレベルの設定
logger.addHandler(logging.StreamHandler()) # ログのコンソール出力の設定
dirname = 'log/{}/'.format(data.dataname)
if not os.path.isdir(dirname):
    os.mkdir(dirname)
if not os.path.isdir(dirname+'double'):
    os.mkdir(dirname+'double')
logging.basicConfig(filename='log/{}/double/{}_l2-{}_l1-{}.log'.format(data.dataname,data.dim,options.lambda2,options.lambda1), format="%(message)s", filemode='w') # ログのファイル出力先を設定

dist, T,D = OPW_w(data.data1, data.data2,[],[],options,1)
logger.info(T)
logger.info(D)
logger.info(dist)