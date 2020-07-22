import numpy as np
from rvsml.run_RVSML import run_RVSML
from set_text import read_txt_embeddings
import logging,pickle,time,os,argparse,torch
from src.utils import bool_flag
from src.evaluation.word_translation import get_word_translation_accuracy
np.set_printoptions(precision=3,suppress=True, threshold=10000)

if 'args':
    # main
    parser = argparse.ArgumentParser(description='Unsupervised training')
    parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
    parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
    parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
    parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")
    # data
    parser.add_argument("--langnum", type=int, default=2, help="the number of languages")
    parser.add_argument("--classnum", type=int, default=30, help="the number of classes")
    parser.add_argument("--w2v_dim", type=int, default=300, help="the dimension of word2vec")
    parser.add_argument("--max_vocab", type=int, default=1000, help="the number of vocablaries")
    # highpara
    parser.add_argument("--method", type=str, default='dtw', help="alignment method")
    parser.add_argument("--v_length", type=int, default=7, help="the rate of the templatenum")
    parser.add_argument("--lambda0", type=float, default=0.1, help="the parameter of the rotation matrix")
    parser.add_argument("--lambda1", type=float, default=0.1, help="the parameter of the inverse difference moment")
    parser.add_argument("--lambda2", type=float, default=0.1, help="the parameter of the standard distribution")
    parser.add_argument("--delta", type=float, default=1, help="variance of the standard distribution")
    parser.add_argument("--init_delta", type=float, default=1, help="variance of the standard distribution")
    parser.add_argument("--reg", type=float, default=1, help="regularization parameter of sinkhorn distance")
    parser.add_argument("--init", type=str, default='uniform', help="initial by random")

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
    def __init__(self):
        self.max_iters, self.err_limit = 1000, 10**(-4)
        if params.method in ["dtw","greedy"]:
            self.lambda0 = 0.1
        elif params.method in ["opw","OT","sinkhorn"]:
            self.lambda0 = 0.1
        self.lambda1, self.lambda2 = params.lambda1, params.lambda2
        self.delta = params.delta
        self.method = params.method
        self.init = params.init
        self.init_delta = params.init_delta
        self.templatenum = params.v_length
        self.cpu_count = os.cpu_count()//2
        self.classify = 'knn'

class Dataset:
    def __init__(self,data=None):
        self.dataname = 'humanRights'
        self.langs = ['es','en']
        self.langnum, self.classnum, self.dim = 2, 30, 300
        self.trainsetdatanum = self.langnum * self.classnum
        self.trainsetnum = [self.langnum] * self.classnum
        self.testsetdatanum = self.trainsetdatanum
        self.ClassLabel = np.arange(self.classnum).T+1
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
dirname = '{}/log/{}-{}_{}_v{}/'.format(dataset.dataname,dataset.langs[0],dataset.langs[1],params.method,params.v_length)
if not os.path.isdir(dirname):
    os.mkdir(dirname)
logging.basicConfig(filename='{}/log/{}-{}_{}_v{}/init-{}{}.log'.format(dataset.dataname,dataset.langs[0],dataset.langs[1],params.method,params.v_length,params.init,params.init_delta), format="%(message)s", filemode='w') # ログのファイル出力先を設定

data = []
for c in range(dataset.classnum):
    for l in dataset.langs:
        data.append(np.load('{}/vectorized_texts/{}/article{}.npy'.format(dataset.dataname,l,c+1)))
dataset = Dataset(data)
dataset.getlabelfull()

dataset = run_RVSML(dataset,options)

dico1,emb1 = read_txt_embeddings(dataset.langs[0])
dico2,emb2 = read_txt_embeddings(dataset.langs[1])
emb1 = np.dot(emb1,dataset.L)
emb2 = np.dot(emb2,dataset.L)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
emb1 = torch.from_numpy(emb1.astype(np.float32)).clone().to(device)
emb2 = torch.from_numpy(emb2.astype(np.float32)).clone().to(device)
accs = get_word_translation_accuracy(dataset.langs[0], dico1.word2id, emb1, dataset.langs[1], dico2.word2id, emb2, "nn", 'default')
logger.info(accs)