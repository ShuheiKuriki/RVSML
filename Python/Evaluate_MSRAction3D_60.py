import numpy as np
from rvsml_torch.run_RVSML import run_RVSML
import logging,pickle,time,os,argparse,torch
np.set_printoptions(precision=3,suppress=True)

if 'args':
    parser = argparse.ArgumentParser(description='MSRAction3D')
    # highpara
    parser.add_argument("--method", type=str, default='dtw', help="alignment method")
    parser.add_argument("--v_length", type=int, default=4, help="the rate of the templatenum")
    # parser.add_argument("--lambda0", type=float, default=0.1, help="the parameter of the rotation matrix")
    parser.add_argument("--lambda1", type=float, default=0.001, help="the parameter of the inverse difference moment")
    parser.add_argument("--lambda2", type=float, default=0.01, help="the parameter of the standard distribution")
    parser.add_argument("--delta", type=float, default=5,help="variance of the standard distribution")
    parser.add_argument("--init_delta", type=float, default=1, help="variance of the standard distribution")
    parser.add_argument("--reg", type=float, default=0.003, help="regularization parameter of sinkhorn distance")
    parser.add_argument("--init", type=str, default='normal', help="initial by random")
    parser.add_argument("--distance", type=str, default='sqeuclidean', help="type of distnce")

params = parser.parse_args()
class Options:
    def __init__(self):
        self.max_iters, self.err_limit = 100, 10**(-4)
        self.method = params.method
        self.init = params.init
        self.init_delta = params.init_delta
        self.classify = 'knn'
        if self.method == 'opw':
            self.lambda1, self.lambda2, self.delta = params.lambda1, params.lambda2, params.delta
        self.lambda0 = 0.1
        self.distance = params.distance
        self.regularize = params.reg

class Dataset:
    def __init__(self):
        self.dataname = 'MSRAction3D'
        self.classnum, self.dim = 20, 60
        self.trainsetdatanum, self.testsetdatanum  = 284, 273
        self.ClassLabel = torch.arange(self.classnum).t()+1
        self.trainsetdatalabel, self.testsetdatalabel = [0], [0]
        self.templatenums = [params.v_length]*self.classnum

    def getlabelfull(self):
        self.trainsetlabelfull = self.getLabel(self.trainsetdatalabel)
        self.testsetlabelfull = self.getLabel(self.testsetdatalabel)
        return

    def getLabel(self,classid):
        p = int(max(classid))
        # logger.info(p)
        X = np.zeros((len(classid),p))-1
        for i in range(p):
            indx = np.nonzero(classid == i+1)
            X[indx,i] = 1
        return X

options = Options()
dataset = Dataset()

if 'logger':
    logger = logging.getLogger('{}Log'.format(dataset.dataname)) # ログの出力名を設定
    logger.setLevel(20) # ログレベルの設定
    logger.addHandler(logging.StreamHandler()) # ログのコンソール出力の設定
    dirname = 'MSRAction3D/log/{}/'.format(params.method)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    if params.method == 'opw':
        filename='MSRAction3D/log/{}/v{}_l2-{}_del{}_init-{}_init-del{}_l1-{}.log'.format(params.method,params.v_length,params.lambda2,params.delta,params.init,params.init_delta,params.lambda1)
    elif params.method in ['dtw','greedy','OT']:
        filename = 'MSRAction3D/log/{}/v{}_init-{}_init-del{}.log'.format(params.method,params.v_length,params.init,params.init_delta)
    elif params.method == 'sinkhorn':
        filename='MSRAction3D/log/{}/v{}_reg{}_init-{}_init-del{}.log'.format(params.method,params.v_length,params.reg,params.init,params.init_delta)
    logging.basicConfig(filename=filename, format="%(message)s", filemode='w') # ログのファイル出力先を設定

pickle_list = ['trainsetnum','trainsetdatalabel','testsetdatalabel','trainsetdata','testsetdata','trainset']

for v in pickle_list:
    exec("with open('MSRAction3D/data/{}.bin','rb') as f: dataset.{} = pickle.load(f)".format(v,v))

for i in range(len(dataset.trainsetdata)):
    dataset.trainsetdata[i] = torch.from_numpy(dataset.trainsetdata[i].astype(np.float32)).clone()
for i in range(len(dataset.testsetdata)):
    dataset.testsetdata[i] = torch.from_numpy(dataset.testsetdata[i].astype(np.float32)).clone()
for i in range(len(dataset.trainset)):
    for j in range(len(dataset.trainset[i])):
        dataset.trainset[i][j] = torch.from_numpy(dataset.trainset[i][j].astype(np.float32)).clone()

dataset.getlabelfull()

accs = []
for l0 in [0.001,0.01,0.1]:
    options.lambda0 = l0
    dataset,knn_acc,virtual_acc = run_RVSML(dataset,options)
    accs.append([knn_acc,virtual_acc])

for i,l0 in enumerate([0.001,0.01,0.1]):
    logger.info('lambda0:{}, knn_acc:{}, virtual_acc:{}'.format(l0,accs[i][0],accs[i][1]))
