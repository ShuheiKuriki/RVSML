"""CWLE learning with humanRights"""
import logging, pickle, time, os, torch
import numpy as np
from rvsml_torch.run_RVSML import run_RVSML
from set_text import read_txt_embeddings
from src.evaluation.word_translation import get_word_translation_accuracy
from src.parser import get_parser
np.set_printoptions(precision=3, suppress=True, threshold=10000)

parser = get_parser(rvsml=True, muse=True)
params = parser.parse_args()
assert not params.cuda or torch.cuda.is_available()
assert 0 <= params.dis_dropout < 1
assert 0 <= params.dis_input_dropout < 1
assert 0 <= params.dis_smooth < 0.5
assert params.dis_lambda > 0 and params.dis_steps > 0
assert 0 < params.lr_shrink <= 1
# assert os.path.isfile(params.src_emb)
# assert os.path.isfile(params.tgt_emb)
assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)
assert params.export in ["", "txt", "pth"]

class Options:
    """学習のオプション"""
    def __init__(self):
        self.max_epoch, self.err_limit = 1000, 10**(-4)
        if params.method == 'dtw':
            self.lambda0 = 0.1
        elif params.method == 'greedy':
            self.lambda0 = 0.01
        elif params.method == "OT":
            self.lambda0 = 0.001
        elif params.method == "opw":
            self.lambda0 = 0.0001
        elif params.method == "sinkhorn":
            self.lambda0 = 0.0001
        self.lambda1, self.lambda2 = params.lambda1, params.lambda2
        self.delta = params.delta
        self.method = params.method
        self.init = params.init
        self.init_delta = params.init_delta
        # self.templatenums = 1
        self.metric = params.metric
        self.regularize = params.reg
        self.cuda = params.cuda
        self.alpha = params.alpha
        self.solve_method = params.solve_method

class Dataset:
    """データセット関連"""
    def __init__(self, data=None):
        self.dataname = 'humanRights'
        self.langs = ['en', 'es']
        self.langnum, self.classnum, self.dim = len(self.langs), params.classnum, params.w2v_dim
        self.trainsetdatanum = self.langnum * self.classnum
        self.trainsetnum = [self.langnum] * self.classnum
        self.testsetdatanum = self.trainsetdatanum
        self.ClassLabel = torch.arange(self.classnum).T+1
        self.trainsetdatalabel = [1+ i//self.langnum for i in range(self.trainsetdatanum)]
        self.testsetdatalabel = self.trainsetdatalabel
        self.L = 0
        if data is not None:
            self.trainsetdata, self.testsetdata = data, data
            trainset = [[0]*self.langnum for _ in range(self.classnum)]
            self.templatenums = torch.zeros(self.classnum, dtype=torch.int32)
            for c in range(self.classnum):
                for l in range(self.langnum):
                    trainset[c][l] = data[c*self.langnum+l]
                    self.templatenums[c] += len(trainset[c][l])
                self.templatenums[c] = max(self.templatenums[c]//(self.langnum*params.v_rate), 1)
            self.trainset = trainset

    # def getlabelfull(self):
    #     self.trainsetlabelfull = self.getLabel(self.trainsetdatalabel)
    #     self.testsetlabelfull = self.getLabel(self.testsetdatalabel)

    # def getLabel(self, classid):
    #     p = int(max(classid))
    #     # logger.info(p)
    #     X = torch.zeros((np.size(classid), p))-1
    #     for i in range(p):
    #         indx = np.nonzero(classid == i+1)
    #         X[indx, i] = 1
    #     return X

options = Options()
dataset = Dataset()

logger = logging.getLogger('{}Log'.format(dataset.dataname)) # ログの出力名を設定
logger.setLevel(20) # ログレベルの設定
logger.addHandler(logging.StreamHandler()) # ログのコンソール出力の設定
lang_chr = ''
for i in range(dataset.langnum):
    lang_chr += dataset.langs[i]
dirname = '{}/log/{}_{}/'.format(dataset.dataname, lang_chr, params.method)
if not os.path.isdir(dirname):
    os.mkdir(dirname)
logging.basicConfig(filename='{}/log/{}_{}/init-{}{}.log'.format(dataset.dataname, lang_chr, params.method, params.init, params.init_delta), format="%(message)s", filemode='w') # ログのファイル出力先を設定

device = 'cuda' if params.cuda else 'cpu'
data1 = []
for cla in range(dataset.classnum):
    for lang in dataset.langs:
        d = np.load('{}/vectorized_texts/{}/sentence{}.npy'.format(dataset.dataname, lang, cla+1))
        data1.append(torch.from_numpy(d).clone().to(device))
dataset = Dataset(data1)
# dataset.getlabelfull()

dataset, knn_accs, virtual_acc = run_RVSML(dataset, options)

dicos = [0]*dataset.langnum
embs = [0]*dataset.langnum
for i in range(dataset.langnum):
    dicos[i], embs[i] = read_txt_embeddings(dataset.langs[i])
    embs[i] = torch.from_numpy(np.dot(embs[i], dataset.L).astype(np.float32)).clone().to(device)
for i in range(dataset.langnum):
    for j in range(dataset.langnum):
        if i == j:
            continue
        results = get_word_translation_accuracy(dataset.langs[i], dicos[i].word2id, embs[i], dataset.langs[j], dicos[j].word2id, embs[j], "csls_knn_10", 'default')
        logger.info('%s-%s: %.5f', dataset.langs[i], dataset.langs[j], results)
# logger.info(acc
