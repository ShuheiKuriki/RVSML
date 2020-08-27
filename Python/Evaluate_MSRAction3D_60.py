"""MSRAction3Dの学習・分類を実行"""
import logging, pickle, time, os, torch
import numpy as np
from rvsml_torch.run_RVSML import run_RVSML
from src.parser import get_parser
np.set_printoptions(precision=3, suppress=True, threshold=10000)

parser = get_parser(rvsml=True, muse=False)
params = parser.parse_args()

class Options:
    """オプションを定義"""
    def __init__(self):
        self.max_epoch, self.err_limit = 10000, 10**(-5)
        self.method = params.method
        self.init = params.init
        self.init_delta = params.init_delta
        self.classify = 'knn'
        if self.method == 'opw':
            self.lambda1, self.lambda2, self.delta = params.lambda1, params.lambda2, params.delta
        if self.method == 'dtw':
            self.lambda0 = params.lambda0_for_dtw
        else:
            self.lambda0 = params.lambda0_for_other
        self.metric = params.metric
        self.regularize = params.reg
        self.cuda = params.cuda
        self.dtype = torch.float64
        self.solve_method = params.solve_method
        self.lr = params.lr
        self.alpha = params.alpha

class Dataset:
    """データセットを用意"""
    def __init__(self):
        self.dataname = 'MSRAction3D'
        self.classnum, self.dim = 20, 60
        self.trainsetdatanum, self.testsetdatanum = 284, 273
        self.ClassLabel = torch.arange(self.classnum).t()+1
        self.trainsetdatalabel, self.testsetdatalabel = [0], [0]
        self.templatenums = [params.v_length]*self.classnum

    def getlabelfull(self):
        """今使ってない"""
        self.trainsetlabelfull = getLabel(self.trainsetdatalabel)
        self.testsetlabelfull = getLabel(self.testsetdatalabel)

def getLabel(classid):
    """これも使ってない"""
    p = int(max(classid))
    # logger.info(p)
    X = np.zeros((len(classid), p))-1
    for q in range(p):
        indx = np.nonzero(classid == q+1)
        X[indx, q] = 1
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
        filename = 'MSRAction3D/log/{}/v{}_l2-{}_del{}_init-{}_init-del{}_l1-{}.log'.format(params.method, params.v_length, params.lambda2, params.delta, params.init, params.init_delta, params.lambda1)
    elif params.method in ['dtw', 'greedy', 'OT']:
        filename = 'MSRAction3D/log/{}/v{}_init-{}_init-del{}.log'.format(params.method, params.v_length, params.init, params.init_delta)
    elif params.method == 'sinkhorn':
        filename = 'MSRAction3D/log/{}/v{}_reg{}_init-{}_init-del{}.log'.format(params.method, params.v_length, params.reg, params.init, params.init_delta)
    logging.basicConfig(filename=filename, format="%(message)s", filemode='w') # ログのファイル出力先を設定

pickle_list = ['trainsetnum', 'trainsetdatalabel', 'testsetdatalabel', 'trainsetdata', 'testsetdata', 'trainset']

for v in pickle_list:
    exec("with open('MSRAction3D/data/%s.bin','rb') as f: dataset.%s = pickle.load(f)" %(v, v))

device = 'cuda' if params.cuda else 'cpu'
for i in range(len(dataset.trainsetdata)):
    dataset.trainsetdata[i] = torch.from_numpy(dataset.trainsetdata[i]).clone().to(device)
for i in range(len(dataset.testsetdata)):
    dataset.testsetdata[i] = torch.from_numpy(dataset.testsetdata[i]).clone().to(device)
for i in range(len(dataset.trainset)):
    for j in range(len(dataset.trainset[i])):
        dataset.trainset[i][j] = torch.from_numpy(dataset.trainset[i][j]).clone().to(device)

# dataset.getlabelfull()

dataset, knn_acc, virtual_acc = run_RVSML(dataset, options)
logger.info("method: %s, knn_acc: %.4f, virtual_acc: %.4f", params.method, knn_acc, virtual_acc)
