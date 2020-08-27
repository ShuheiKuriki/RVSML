"""ダブりケースを評価する"""
import logging, pickle, time, os, torch
import numpy as np
from rvsml.RVSML_OT_Learning import RVSML_OT_Learning
from set_text import read_txt_embeddings
from src.evaluation.word_translation import get_word_translation_accuracy_for_random
from src.parser import get_parser
from statistics import mean, stdev
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
    """option"""
    def __init__(self):
        self.max_iters, self.err_limit = 1000, 10**(-4)
        self.method = params.method
        if self.method == 'dtw':
            self.lambda0 = params.lambda0
        if self.method == 'opw':
            self.lambda0, self.lambda1, self.lambda2, self.delta = params.lambda0, params.lambda1, params.lambda2, params.delta
        self.templatenum = int(params.seqlen*params.v_rate)
        self.cpu_count = os.cpu_count()//2

class Dataset:
    """dataset関連"""
    def __init__(self):
        self.dataname = 'double'
        self.langnum, self.classnum, self.seqlen, self.max_vocab, self.w2v_dim = params.langnum, params.classnum, params.seqlen, params.max_vocab, params.w2v_dim
        self.dim = params.w2v_dim*params.langnum
        self.trainsetdatanum = params.langnum * params.classnum
        self.trainsetnum = [params.langnum]*params.classnum
        self.testsetdatanum = self.trainsetdatanum
        self.trainsetdatalabel = [1+ i//params.langnum for i in range(self.trainsetdatanum)]
        self.testsetdatalabel = self.trainsetdatalabel
        self.L = 0
        src_embs = np.random.rand(self.max_vocab, self.w2v_dim)*2-1

        embeddings = [0]*self.langnum
        for lang in range(self.langnum):
            embeddings[lang] = np.zeros((self.max_vocab, self.dim))
            mapping = np.random.rand(self.w2v_dim, self.w2v_dim)*2-1
            embeddings[lang][:, self.w2v_dim*lang:self.w2v_dim*(lang+1)] = np.dot(src_embs, mapping)

        sentences = [0]*self.classnum
        sents = []
        for c in range(self.classnum):
            ss = np.random.randint(0, self.max_vocab, (self.seqlen))
            while sum(ss) in sents:
                ss = np.random.randint(0, self.max_vocab, (self.seqlen))
            sentences[c] = ss
            sents.append(sum(ss))

        data = [[0]*self.langnum for _ in range(self.classnum)]

        for c in range(self.classnum):
            for lang in range(self.langnum):
                nums = [1]*self.seqlen
                nums[2] = params.double_num
                order = []
                for ii in range(self.seqlen):
                    for _ in range(nums[ii]):
                        order.append(ii)
                order = np.array(order)
                # print(order)
                data[c][lang] = embeddings[lang][sentences[c][order]]

        self.embeddings = embeddings
        self.trainset = data
        self.trainsetdata = []
        for c in range(self.classnum):
            for lang in range(self.langnum):
                self.trainsetdata.append(data[c][lang])
        self.testsetdata = self.trainsetdata

options = Options()
dataset = Dataset()

if 'logger':
    logger = logging.getLogger('{}Log'.format(dataset.dataname)) # ログの出力名を設定
    logger.setLevel(20) # ログレベルの設定
    logger.addHandler(logging.StreamHandler()) # ログのコンソール出力の設定
    dirname = 'log/{}/'.format(dataset.dataname)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    dirname = 'log/{}/{}/'.format(dataset.dataname, params.method)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    dirname = 'log/{}/{}/double{}/'.format(dataset.dataname, params.method, params.double_num)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    dirname = 'log/{}/{}/double{}/c{}_sl{}_wd{}/'.format(dataset.dataname, params.method, params.double_num, params.classnum, params.seqlen, params.w2v_dim)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    if params.method == 'dtw':
        logging.basicConfig(filename='log/{}/{}/double{}/c{}_sl{}_wd{}/vr{}_l0-{}_mv{}.log'.format(dataset.dataname, params.method, params.double_num, params.classnum, params.seqlen, params.w2v_dim, params.v_rate, params.lambda0, params.max_vocab), format="%(message)s", filemode='w') # ログのファイル出力先を設定
    elif params.method == 'opw':
        logging.basicConfig(filename='log/{}/{}/double{}/c{}_sl{}_wd{}/vr{}_l2-{}_l0-{}_l1-{}_mv{}.log'.format(dataset.dataname, params.method, params.double_num, params.classnum, params.seqlen, params.w2v_dim, params.v_rate, params.lambda2, params.lambda0, params.lambda1, params.max_vocab), format="%(message)s", filemode='w') # ログのファイル出力先を設定

avgs = []
for i in range(5):
    logger.info(i)
    dataset = Dataset()
    dataset = RVSML_OT_Learning(dataset, options)

    learned_emb = [0]*dataset.langnum
    for l in range(dataset.langnum):
        learned_emb[l] = np.dot(dataset.embeddings[l], dataset.L)
    dataset.learned_emb = learned_emb

    avg = get_word_translation_accuracy_for_random(dataset, options)
    avgs.append(avg)
logger.info('mean:%.7f, std:%.7f', mean(avgs), stdev(avgs))

if False:
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from matplotlib.cm import get_cmap
    plt.switch_backend('agg')
    decomp = TSNE(n_components=2, perplexity=30)
    # decomp = PCA(n_components=2)
    cmap = get_cmap("tab10")
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    for l in range(options.langnum):
        x = dataset.embeddings[l]
        x = decomp.fit_transform(x)
        for w in range(options.max_vocab):
            marker = "$" + str(w+1) + "$"
            ax1.scatter(x[w, 0], x[w, 1], marker=marker, color=cmap(l))
    ax1.set_title(f"before ditributions")

    for l in range(options.langnum):
        x = learned_emb[l]
        # x = decomp.fit_transform(x)
        for w in range(options.max_vocab):
            marker = "$" + str(w+1) + "$"
            ax2.scatter(x[w, 0], x[w, 1], marker=marker, color=cmap(l))
    ax2.set_title(f"after ditributions")

    plt.savefig('figure/before_after_{}_{}_{}_{}_{}_{}.png'.format(options.method, options.langnum, options.classnum, options.seqlen, options.w2v_dim, options.max_vocab))
