"""CWLE learning with humanRights"""
import logging, pickle, time, os, argparse, torch
import numpy as np
from rvsml_torch.run_RVSML import run_RVSML
from set_text import read_txt_embeddings
from src.utils import bool_flag
from src.evaluation.word_translation import get_word_translation_accuracy
np.set_printoptions(precision=3, suppress=True, threshold=10000)

if 'args':
    # main
    parser = argparse.ArgumentParser(description='Unsupervised training')
    parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
    parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
    parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--cuda", type=bool_flag, default=False, help="Run on GPU")
    parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")
    # data
    # parser.add_argument("--langnum", type=int, default=2, help="the number of languages")
    parser.add_argument("--classnum", type=int, default=60, help="the number of classes")
    parser.add_argument("--w2v_dim", type=int, default=300, help="the dimension of word2vec")
    # highpara
    parser.add_argument("--method", type=str, default='dtw', help="alignment method")
    parser.add_argument("--v_rate", type=int, default=8, help="the rate of the templatenum")
    parser.add_argument("--lambda1", type=float, default=0.001, help="the parameter of the inverse difference moment")
    parser.add_argument("--lambda2", type=float, default=0.01, help="the parameter of the standard distribution")
    parser.add_argument("--delta", type=float, default=1, help="variance of the standard distribution")
    parser.add_argument("--init_delta", type=float, default=1, help="variance of the standard distribution")
    parser.add_argument("--reg", type=float, default=0.003, help="regularization parameter of sinkhorn metric")
    parser.add_argument("--init", type=str, default='normal', help="initial by random")
    parser.add_argument("--metric", type=str, default='cosine', help="type of distnce")
    parser.add_argument("--solve_method", type=str, default='analytic', help="how to optimize")
    parser.add_argument("--alpha", type=float, default=0.001, help="map_beta")

    if "mapping":
        parser.add_argument("--map_id_init", type=bool_flag, default=True, help="Initialize the mapping as an identity matrix")
        parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalizan")
    if "discriminator":
        parser.add_argument("--dis_layers", type=int, default=2, help="Discriminator layers")
        parser.add_argument("--dis_hid_dim", type=int, default=2048, help="Discriminator hidden layer dimenns")
        parser.add_argument("--dis_dropout", type=float, default=0., help="Discriminator dropout")
        parser.add_argument("--dis_input_dropout", type=float, default=0.1, help="Discriminator input dropout")
        parser.add_argument("--dis_steps", type=int, default=5, help="Discriminator steps")
        parser.add_argument("--dis_lambda", type=float, default=1, help="Discriminator loss feedback coefficient")
        parser.add_argument("--dis_most_frequent", type=int, default=75000, help="Select embeddings of the k most frequent words for discriminan (0 to disable)")
        parser.add_argument("--dis_smooth", type=float, default=0.1, help="Discriminator smooth predicns")
        parser.add_argument("--dis_clip_weights", type=float, default=0, help="Clip discriminator weights (0 to disable)")
    if "training adversarial":
        parser.add_argument("--adversarial", type=bool_flag, default=True, help="Use adversarial training")
        parser.add_argument("--n_epochs", type=int, default=2, help="Number of epochs")
        parser.add_argument("--epoch_size", type=int, default=200000, help="Iterans per epoch")
        parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
        parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
        parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="Discriminator optimizer")
        parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
        parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
        parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validan metric decreases (1 to disable)")
    if "training refinement":
        parser.add_argument("--n_refinement", type=int, default=2, help="Number of refinement iterans (0 to disable the refinement procedure)")
    if "dicnary crean parameters (for refinement)":
        parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluan dicnary")
        parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dicnary generan (nn/invsm_beta_30/csls_knn_10)")
        parser.add_argument("--dico_build", type=str, default='S2T', help="S2T,T2S,S2T|T2S,S2T&T2S")
        parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dicnary generan")
        parser.add_argument("--dico_max_rank", type=int, default=15000, help="Maximum dicnary words rank (0 to disable)")
        parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dicnary size (0 to disable)")
        parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dicnary size (0 to disable)")
    if "reload pre-trained embeddings":
        parser.add_argument("--src_emb", type=str, default="vectors/wiki.en.vec", help="Reload source embeddings")
        parser.add_argument("--tgt_emb", type=str, default="vectors/wiki.es.vec", help="Reload target embeddings")
        parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")
    if "parse parameters":
        params = parser.parse_args()
    if "check parameters":
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
print(device)
data1 = []
for cla in range(dataset.classnum):
    for lang in dataset.langs:
        d = np.load('{}/vectorized_texts/{}/sentence{}.npy'.format(dataset.dataname, lang, cla+1))
        data1.append(torch.from_numpy(d).clone().to(device))
dataset = Dataset(data1)
dataset.getlabelfull()

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
