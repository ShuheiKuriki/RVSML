import numpy as np
from rvsml.RVSML_OT_Learning import RVSML_OT_Learning
import logging,pickle,time,os,argparse,torch
from src.utils import bool_flag
from src.evaluation.word_translation import get_word_translation_accuracy_for_random
from statistics import *
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
    parser.add_argument("--langnum", type=int, default=3, help="the number of languages")
    parser.add_argument("--classnum", type=int, default=10, help="the number of classes")
    parser.add_argument("--seqlen", type=int, default=10, help="the length of the sequence")
    parser.add_argument("--swap", type=float, default=0, help="the rate of swap")
    parser.add_argument("--w2v_dim", type=int, default=50, help="the dimension of word2vec")
    parser.add_argument("--max_vocab", type=int, default=1000, help="the number of vocablaries")
    parser.add_argument("--freq_rate", type=float, default=0, help="the number of vocablaries")
    parser.add_argument("--perturb", type=float, default=0, help="the rate of perturbation")
    # highpara
    parser.add_argument("--method", type=str, default='opw', help="alignment method")
    parser.add_argument("--v_rate", type=float, default=1, help="the rate of the templatenum")
    parser.add_argument("--lambda0", type=float, default=0.1, help="the parameter of the rotation matrix")
    parser.add_argument("--lambda1", type=float, default=0.1, help="the parameter of the inverse difference moment")
    parser.add_argument("--lambda2", type=float, default=0.1, help="the parameter of the standard distribution")
    parser.add_argument("--delta", type=float, default=1, help="variance of the standard distribution")
    parser.add_argument("--init_delta", type=float, default=0.1, help="variance of the standard distribution")
    parser.add_argument("--reg", type=float, default=1, help="regularization parameter of sinkhorn distance")
    parser.add_argument("--init", type=str, default='uniform', help="initial by random")

    # mapping
    parser.add_argument("--map_id_init", type=bool_flag, default=True, help="Initialize the mapping as an identity matrix")
    parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalizan")
    # discriminator
    parser.add_argument("--dis_layers", type=int, default=2, help="Discriminator layers")
    parser.add_argument("--dis_hid_dim", type=int, default=2048, help="Discriminator hidden layer dimenns")
    parser.add_argument("--dis_dropout", type=float, default=0., help="Discriminator dropout")
    parser.add_argument("--dis_input_dropout", type=float, default=0.1, help="Discriminator input dropout")
    parser.add_argument("--dis_steps", type=int, default=5, help="Discriminator steps")
    parser.add_argument("--dis_lambda", type=float, default=1, help="Discriminator loss feedback coefficient")
    parser.add_argument("--dis_most_frequent", type=int, default=75000, help="Select embeddings of the k most frequent words for discriminan (0 to disable)")
    parser.add_argument("--dis_smooth", type=float, default=0.1, help="Discriminator smooth predicns")
    parser.add_argument("--dis_clip_weights", type=float, default=0, help="Clip discriminator weights (0 to disable)")
    # training adversarial
    parser.add_argument("--adversarial", type=bool_flag, default=True, help="Use adversarial training")
    parser.add_argument("--n_epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--epoch_size", type=int, default=200000, help="Iterans per epoch")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
    parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="Discriminator optimizer")
    parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
    parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validan metric decreases (1 to disable)")
    # training refinement
    parser.add_argument("--n_refinement", type=int, default=2, help="Number of refinement iterans (0 to disable the refinement procedure)")
    # dicnary crean parameters (for refinement)
    parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluan dicnary")
    parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dicnary generan (nn/invsm_beta_30/csls_knn_10)")
    parser.add_argument("--dico_build", type=str, default='S2T', help="S2T,T2S,S2T|T2S,S2T&T2S")
    parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dicnary generan")
    parser.add_argument("--dico_max_rank", type=int, default=15000, help="Maximum dicnary words rank (0 to disable)")
    parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dicnary size (0 to disable)")
    parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dicnary size (0 to disable)")
    # reload pre-trained embeddings
    parser.add_argument("--src_emb", type=str, default="vectors/wiki.en.vec", help="Reload source embeddings")
    parser.add_argument("--tgt_emb", type=str, default="vectors/wiki.es.vec", help="Reload target embeddings")
    parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")

    # parse parameters
    params = parser.parse_args()

    # check parameters
    assert not params.cuda or torch.cuda.is_available()
    assert 0 <= params.dis_dropout < 1
    assert 0 <= params.dis_input_dropout < 1
    assert 0 <= params.dis_smooth < 0.5
    assert params.dis_lambda > 0 and params.dis_steps > 0
    assert 0 < params.lr_shrink <= 1
    assert os.path.isfile(params.src_emb)
    assert os.path.isfile(params.tgt_emb)
    assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)
    assert params.export in ["", "txt", "pth"]

class Options:
    def __init__(self):
        self.max_iters, self.err_limit = 1000, 10**(-6)
        self.method = params.method
        if self.method == 'opw':
            self.lambda1, self.lambda2, self.delta = params.lambda1, params.lambda2, params.delta
        self.lambda0 = params.lambda0
        self.templatenum = int(params.seqlen*params.v_rate)
        self.cpu_count = os.cpu_count()//3
        self.init = params.init
        self.init_delta = params.init_delta
        self.classify = 'knn'

class Dataset:
    def __init__(self):
        self.dataname = 'init_'+params.init
        self.langnum,self.classnum,self.seqlen,self.max_vocab,self.w2v_dim = params.langnum,params.classnum,params.seqlen,params.max_vocab,params.w2v_dim
        self.dim = params.w2v_dim*params.langnum
        self.trainsetdatanum = params.langnum * params.classnum
        self.trainsetnum = [params.langnum]*params.classnum
        self.testsetdatanum = self.trainsetdatanum
        self.trainsetdatalabel = [1 + i//params.langnum for i in range(self.trainsetdatanum)]
        self.testsetdatalabel = self.trainsetdatalabel
        self.L = 0
        self.real_vocab = self.max_vocab-int(params.freq_rate * self.max_vocab)
        src_embs = np.zeros((self.real_vocab, self.w2v_dim))
        src_embs = np.random.rand(self.real_vocab, self.w2v_dim)*2-1

        embeddings = [0]*self.langnum
        for l in range(self.langnum):
            embeddings[l] = np.zeros((self.real_vocab,self.dim))
            mapping = np.random.rand(self.w2v_dim, self.w2v_dim)*2-1
            embeddings[l][:,self.w2v_dim*l:self.w2v_dim*(l+1)] = np.dot(src_embs,mapping)+np.random.rand(self.real_vocab, self.w2v_dim)*params.perturb

        self.sentences = [0]*self.classnum
        sents = []
        for c in range(self.classnum):
            ss = np.random.randint(0,self.max_vocab,(self.seqlen))
            for s in range(self.seqlen):
                if ss[s]>=self.real_vocab:
                    ss[s] = 0
            while sum(ss) in sents:
                ss = np.random.randint(0,self.max_vocab,(self.seqlen))
                for s in range(self.seqlen):
                    if ss[s]>=self.real_vocab:
                        ss[s] = 0
            self.sentences[c] = ss
            sents.append(sum(ss))

        data = [[0]*self.langnum for _ in range(self.classnum)]
        orders = [[np.arange(self.seqlen) for _ in range(self.langnum)] for _ in range(self.classnum)]
        for c in range(self.classnum):
            for l in range(self.langnum):
                for s in range(0,self.seqlen,2):
                    if np.random.rand()<params.swap:
                        orders[c][l][s],orders[c][l][s+1] = orders[c][l][s+1],orders[c][l][s]
                data[c][l] = embeddings[l][self.sentences[c][orders[c][l]]]
        self.orders = orders

        self.embeddings = embeddings
        self.trainset = data
        self.trainsetdata = []
        for c in range(self.classnum):
            for l in range(self.langnum):
                self.trainsetdata.append(data[c][l])
        self.testsetdata = self.trainsetdata

dataset = Dataset()
options = Options()

if 'logger':
    logger = logging.getLogger('{}Log'.format(dataset.dataname)) # ログの出力名を設定
    logger.setLevel(20) # ログレベルの設定
    logger.addHandler(logging.StreamHandler()) # ログのコンソール出力の設定
    dirname = 'log/{}/'.format(dataset.dataname)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    dirname = 'log/{}/{}/'.format(dataset.dataname,params.method)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    dirname = 'log/{}/{}/c{}_sl{}_wd{}_f{}_swap{}_per{}/'.format(dataset.dataname,params.method,params.classnum,params.seqlen,params.w2v_dim,params.freq_rate,params.swap,params.perturb)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    if params.method == 'opw':
        filename='log/{}/{}/c{}_sl{}_wd{}_f{}_swap{}_per{}/vr{}_l2-{}_del{}_init-del{}_l0-{}_l1-{}_mv{}.log'.format(dataset.dataname,params.method,params.classnum,params.seqlen,params.w2v_dim,params.freq_rate,params.swap,params.perturb,params.v_rate,params.lambda2,params.delta,params.init_delta,params.lambda0,params.lambda1,params.max_vocab)
        logging.basicConfig(filename=filename, format="%(message)s", filemode='w') # ログのファイル出力先を設定
    elif params.method in ['dtw','greedy','OT']:
        filename = 'log/{}/{}/c{}_sl{}_wd{}_f{}_swap{}_per{}/vr{}_l0-{}_init-del{}_mv{}.log'.format(dataset.dataname,params.method,params.classnum,params.seqlen,params.w2v_dim,params.freq_rate,params.swap,params.perturb,params.v_rate,params.lambda0,params.init_delta,params.max_vocab)
        logging.basicConfig(filename=filename, format="%(message)s", filemode='w') # ログのファイル出力先を設定
    elif params.method == 'sinkhorn':
        filename='log/{}/{}/c{}_sl{}_wd{}_f{}_swap{}_per{}/vr{}_reg{}_l0-{}_init-del{}_mv{}.log'.format(dataset.dataname,params.method,params.classnum,params.seqlen,params.w2v_dim,params.freq_rate,params.swap,params.perturb,params.v_rate,params.reg,params.lambda0,params.init_delta,params.max_vocab)
        logging.basicConfig(filename=filename, format="%(message)s", filemode='w') # ログのファイル出力先を設定

avgs = []
for i in range(10):
    logger.info(i)
    dataset = Dataset()
    dataset = RVSML_OT_Learning(dataset,options)

    learned_emb = [0]*dataset.langnum
    for l in range(dataset.langnum):
        learned_emb[l] = np.dot(dataset.embeddings[l],dataset.L)
    dataset.learned_emb = learned_emb

    avg = get_word_translation_accuracy_for_random(dataset,options)
    avgs.append(avg)

logger.info('mean:{}, std:{}'.format(mean(avgs),stdev(avgs)))
print(filename)

if False:
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from matplotlib.cm import get_cmap
    plt.switch_backend('agg')
    decomp = TSNE(n_components=2,perplexity=30)
    # decomp = PCA(n_components=2)
    cmap = get_cmap("tab10")
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    for l in range(langnum):
        x = embeddings[l]
        x = decomp.fit_transform(x)
        for w in range(max_vocab):
            marker = "$" + str(w+1) + "$"
            ax1.scatter(x[w, 0], x[w, 1], marker=marker, color=cmap(l))
    ax1.set_title(f"before ditributions")

    for l in range(langnum):
        x = learned_emb[l]
        # x = decomp.fit_transform(x)
        for w in range(max_vocab):
            marker = "$" + str(w+1) + "$"
            ax2.scatter(x[w, 0], x[w, 1], marker=marker, color=cmap(l))
    ax2.set_title(f"after ditributions")

    plt.savefig('figure/before_after_{}_{}_{}_{}_{}_{}.png'.format(options.method, langnum, classnum, seqlen, w2v_dim, max_vocab))
