from rvsml.align import dtw2
import argparse,logging,os
import numpy as np

parser = argparse.ArgumentParser()

class Options:
    def __init__(self):
        self.max_iters, self.err_limit = 1000, 10**(-4)
        self.method = 'dtw'

class Dataset:
    def __init__(self):
        self.dataname = 'dtw'
        self.dim = 5
        nums1 = [1,1,1,1,1,1,1,1,1,1]
        nums2 = [1,5,1,1,1,1,1,1,1,1]
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

logger = logging.getLogger('{}Log'.format(data.dataname)) # ログの出力名を設定
logger.setLevel(20) # ログレベルの設定
logger.addHandler(logging.StreamHandler()) # ログのコンソール出力の設定
dirname = 'log/{}/'.format(data.dataname)
if not os.path.isdir(dirname):
    os.mkdir(dirname)
logging.basicConfig(filename='log/{}/double.log'.format(data.dataname), format="%(message)s", filemode='w') # ログのファイル出力先を設定

dist, T = dtw2(data.data1, data.data2)
logger.info(T)
logger.info(dist)