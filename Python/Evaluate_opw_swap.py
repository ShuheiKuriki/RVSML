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
        self.dataname = 'opw_swap'
        self.dim = 6
        self.data1 = np.eye(self.dim)
        self.data2 = np.zeros((self.dim,self.dim))
        for i in range(0,self.dim-1,2):
            if np.random.rand()<0.5:
                self.data2[i,i+1],self.data2[i+1,i] = 1,1
            else:
                self.data2[i,i],self.data2[i+1,i+1] = 1,1
                
options = Options()
data = Dataset()

logger = logging.getLogger('{}Log'.format(data.dataname)) # ログの出力名を設定
logger.setLevel(20) # ログレベルの設定
logger.addHandler(logging.StreamHandler()) # ログのコンソール出力の設定
dirname = 'log/{}/'.format(data.dataname)
if not os.path.isdir(dirname):
    os.mkdir(dirname)
logging.basicConfig(filename='log/{}/{}_l2-{}_l1-{}.log'.format(data.dataname,data.dim,options.lambda2,options.lambda1), format="%(message)s", filemode='w') # ログのファイル出力先を設定

dist, T = OPW_w(data.data1, data.data2,[],[],options,1)
logger.info(data.data2)
logger.info(T)
logger.info(dist)