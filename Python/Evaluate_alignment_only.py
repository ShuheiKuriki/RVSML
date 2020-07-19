from rvsml.align import dtw2,OPW_w,greedy,OT,sinkhorn
import argparse,logging,os
import numpy as np

np.set_printoptions(precision=3,suppress=True)
parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default='opw', help="alignment method")
parser.add_argument("--lambda1", type=float, default=0.1, help="the parameter of the inverse difference moment")
parser.add_argument("--lambda2", type=float, default=0.1, help="the parameter of the standard distribution")
parser.add_argument("--delta", type=float, default=1, help="variance of the standard distribution")
parser.add_argument("--reg", type=float, default=1, help="regularization parameter of sinkhorn distance")
params = parser.parse_args()

class Options:
    def __init__(self):
        self.max_iters, self.err_limit = 1000, 10**(-4)
        self.method = params.method
        if self.method == 'opw':
            self.lambda1, self.lambda2, self.delta = params.lambda1, params.lambda2, params.delta
        if self.method == 'sinkhorn':
            self.reg = params.reg

class Dataset:
    def __init__(self):
        self.dataname = 'double'
        self.dim = 5
        nums1 = [1,1,1,1,1]
        nums2 = [1,5,1,1,1]
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

logger = logging.getLogger('{}Log'.format(options.method)) # ログの出力名を設定
logger.setLevel(20) # ログレベルの設定
logger.addHandler(logging.StreamHandler()) # ログのコンソール出力の設定
dirname = 'log/{}/'.format(options.method)
if not os.path.isdir(dirname):
    os.mkdir(dirname)
if not os.path.isdir(dirname+data.dataname+str(data.dim)):
    os.mkdir(dirname+data.dataname+str(data.dim))

if options.method == 'opw':
    filename = 'log/{}/{}{}/l2-{}_l1-{}.log'.format(options.method,data.dataname,data.dim,options.lambda2,options.lambda1)
elif options.method == 'sinkhorn':
    filename = 'log/{}/{}{}/reg{}.log'.format(options.method,data.dataname,data.dim,options.reg)
else:
    filename = 'log/{}/{}{}/{}.log'.format(options.method,data.dataname,data.dim,options.method)
logging.basicConfig(filename=filename, format="%(message)s", filemode='w') # ログのファイル出力先を設定

if options.method == 'dtw':
    dist, T, D = dtw2(data.data1, data.data2)
elif options.method == 'opw':
    dist, T = OPW_w(data.data1, data.data2,options,0)
elif options.method == 'greedy':
    dist, T = greedy(data.data1, data.data2)
elif options.method == 'OT':
    dist, T = OT(data.data1, data.data2)
elif options.method == 'sinkhorn':
    dist, T = sinkhorn(data.data1, data.data2)
logger.info(T)
# logger.info(D)
logger.info(dist)
logger.info(filename)