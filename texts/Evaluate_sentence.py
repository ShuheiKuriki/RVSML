from src.utils import bool_flag, initialize_exp,read_txt_embeddings,loadFile
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator
from learning import learning
from collections import OrderedDict
import logging,io,os,time,json,argparse,torch
import numpy as np
# np.set_printoptions(precision=0, floatmode='fixed',suppress=True)

if True:
    # main
    parser = argparse.ArgumentParser(description='Unsupervised training')
    parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
    parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
    parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
    parser.add_argument("--exp_name", type=str, default="humanRights", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--cuda", type=bool_flag, default=False, help="Run on GPU")
    parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")
    # data
    parser.add_argument("--src_lang", type=str, default='en', help="Source language")
    parser.add_argument("--tgt_lang", type=str, default='es', help="Target language")
    parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
    parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
    # mapping
    parser.add_argument("--map_id_init", type=bool_flag, default=True, help="Initialize the mapping as an identity matrix")
    parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalization")
    # discriminator
    parser.add_argument("--dis_layers", type=int, default=2, help="Discriminator layers")
    parser.add_argument("--dis_hid_dim", type=int, default=2048, help="Discriminator hidden layer dimensions")
    parser.add_argument("--dis_dropout", type=float, default=0., help="Discriminator dropout")
    parser.add_argument("--dis_input_dropout", type=float, default=0.1, help="Discriminator input dropout")
    parser.add_argument("--dis_steps", type=int, default=5, help="Discriminator steps")
    parser.add_argument("--dis_lambda", type=float, default=1, help="Discriminator loss feedback coefficient")
    parser.add_argument("--dis_most_frequent", type=int, default=75000, help="Select embeddings of the k most frequent words for discrimination (0 to disable)")
    parser.add_argument("--dis_smooth", type=float, default=0.1, help="Discriminator smooth predictions")
    parser.add_argument("--dis_clip_weights", type=float, default=0, help="Clip discriminator weights (0 to disable)")
    # training adversarial
    parser.add_argument("--adversarial", type=bool_flag, default=True, help="Use adversarial training")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--epoch_size", type=int, default=200000, help="Iterations per epoch")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
    parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="Discriminator optimizer")
    parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
    parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation metric decreases (1 to disable)")
    # training refinement
    parser.add_argument("--n_refinement", type=int, default=5, help="Number of refinement iterations (0 to disable the refinement procedure)")
    # dictionary creation parameters (for refinement)
    parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
    parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
    parser.add_argument("--dico_build", type=str, default='S2T', help="S2T,T2S,S2T|T2S,S2T&T2S")
    parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
    parser.add_argument("--dico_max_rank", type=int, default=15000, help="Maximum dictionary words rank (0 to disable)")
    parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
    parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
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
        self.max_nIter, self.err_limit = 1000, 10**(-4)
        self.lambda0, self.lambda1, self.lambda2 = 0.01, 50, 0.1
        self.delta = 1
        self.method, self.classify  = 'dtw', 'virtual'
        self.templatenum = 4

class Dataset:
    def __init__(self,words):
        self.dataname = 'text'
        self.classnum, self.dim = 1, 300
        self.trainsetnum, self.trainsetdatanum, self.testsetdatanum = [1], 1, 1
        self.ClassLabel = np.arange(self.classnum).T+1
        self.trainsetdatalabel, self.testsetdatalabel = [1], [1]
        self.trainset, self.trainsetdata = [[words]], [words]
        self.testsetdata = [torch.from_numpy(words.astype(np.float32)).clone()]

name = 'sentence'

logger = logging.getLogger('{}Log'.format(name)) # ログの出力名を設定
logger.setLevel(20) # ログレベルの設定
logger.addHandler(logging.StreamHandler()) # ログのコンソール出力の設定
logging.basicConfig(filename='{}.log'.format(name), format="%(message)s", filemode='w') # ログのファイル出力先を設定

print('loading_text')
text_src_path = 'sentence/English'
dico,embeddings = read_txt_embeddings(params,source=True)
src_words = loadFile(text_src_path+'.txt',embeddings,dico.word2id)
np.save('words_vec/'+text_src_path, src_words)
src_words = np.load('words_vec/'+text_src_path+'.npy')

text_tgt_path = 'sentence/Spanish'
dico,embeddings = read_txt_embeddings(params,source=False)
tgt_words = loadFile(text_tgt_path+'.txt',embeddings,dico.word2id)
np.save('words_vec/'+text_tgt_path, tgt_words)
tgt_words = np.load('words_vec/'+text_tgt_path+'.npy')

M = [len(src_words), len(tgt_words)] #シーケンスの長さ
print(M)
options = Options()
src_data = Dataset(src_words)
tgt_data = Dataset(tgt_words)

src_data, tgt_data = learning(params,src_data,tgt_data,options)