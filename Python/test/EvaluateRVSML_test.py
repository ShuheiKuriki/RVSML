import numpy as np
from scipy.io import loadmat
import time
from NNClassifier import *
from RVSML_OT_Learning import *
import logging
from random import sample,randint
from itertools import product

N = 2 #2N次元
M = 2 #シーケンスの長さ
classnum = 2
dim = 2*N
V = 1 #クラスあたり訓練データ数
T = 10 #テストデータ数
CVAL = 1
 # add path
# addpath('/usr/local/Cellar/vlfeat-0.9.21/toolbox')
# vl_setup()
# addpath('libsvm-3.20/matlab')

delta = 1
lambda1 = 50
lambda2 = 0.1
max_iters = 100
err_limit = 10**(-6)

class Options:
    def __init__(self, max_iters, err_limit, lambda1, lambda2, delta):
        self.max_iters = max_iters
        self.err_limit = err_limit
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.delta = delta

options = Options(max_iters,err_limit,lambda1,lambda2,delta)

for N,M,V in product(range(3,10),repeat=3):
    if M<N:
        continue
    trainset = [[np.zeros((M,2*N)) for _ in range(V)] for _ in range(2)]
    for c in range(2):
        for i in range(V):
            lis = sample(range(M),k=N)
            for j,l in enumerate(lis):
                trainset[c][i][l,j+N*c]=1

    # trainset[0] = [np.array([np.linspace(0,100,10+i), np.linspace(0,100,10+i)]).reshape(dim,10+i).T for i in range(nperclass)]
    # trainset[1] = [np.array([np.linspace(0,-100,10+i), np.linspace(0,100,10+i)]).reshape(dim,10+i).T for i in range(nperclass)]
    # trainset[2] = [np.array([np.linspace(0,-100,10+i), np.linspace(0,-100,10+i)]).reshape(dim,10+i).T for i in range(nperclass)]
    # trainset[3] = [np.array([np.linspace(0,100,10+i), np.linspace(0,-100,10+i)]).reshape(dim,10+i).T for i in range(nperclass)]

    # trainsetdatanum = data["trainsetdatanum"][0][0]
    trainsetdatanum = V*2
    # trainsetdatalabel = data["trainsetdatalabel"][0]
    trainsetdatalabel = [1]*V+[2]*V
    # trainsetnum = data["trainsetnum"][0]
    trainsetnum = [V]*2
    # testsetdata = data["testsetdata"][0]

    testsetdata = [np.zeros((M,2*N)) for _ in range(T)]
    testsetdatalabel = [0]*T

    for i in range(T):
        cla = randint(1,2)
        lis = sample(range(M),k=N)
        for j,l in enumerate(lis):
            testsetdata[i][l,j+N*(cla-1)]=1
        testsetdatalabel[i] = cla

    # testsetdata = [0]*classnum
    # testsetdata = [np.array([np.linspace(0,100,20+i), np.linspace(0,100,20+i)]).reshape(dim,20+i).T for i in range(nperclass)]
    # testsetdata += [np.array([np.linspace(0,-100,20+i), np.linspace(0,100,20+i)]).reshape(dim,20+i).T for i in range(nperclass)]
    # testsetdata += [np.array([np.linspace(0,-100,20+i), np.linspace(0,-100,20+i)]).reshape(dim,20+i).T for i in range(nperclass)]
    # testsetdata += [np.array([np.linspace(0,100,20+i), np.linspace(0,-100,20+i)]).reshape(dim,20+i).T for i in range(nperclass)]

    # testsetdatanum = data["testsetdatanum"][0][0]
    testsetdatanum = T
    # testsetdatalabel = data["testsetdatalabel"][0]
    trainset_m = trainset
    testsetdata_m = testsetdata
    testsetlabel = testsetdatalabel


    # print("data load done")

    for t in range(M,M+1):
        templatenum = t
            # print("DTW start")

            # templatenum = 8
        lambda0 = 0.1
        tic = time.time()
        L_dtw, v_s_dtw, dists_dtw, Ts_dtw = RVSML_OT_Learning_dtw(trainset,templatenum,lambda0,options)
        RVSML_dtw_time = time.time() - tic

            # v_s_dtw = np.array([v[0] for v in v_s_dtw])
            # real_v_dtw = np.linalg.solve(L.T,v_s_dtw)
            # print("dtw learning done")
            ## classification with the learned metric
        traindownset = [0]*classnum
        testdownsetdata = [0]*testsetdatanum
        for j in range(classnum):
            traindownset[j] = [0]*trainsetnum[j]
            for m in range(trainsetnum[j]):
                traindownset[j][m] = np.dot(trainset[j][m] ,L_dtw)

        for j in range(testsetdatanum):
            testdownsetdata[j] = np.dot(testsetdata[j], L_dtw)

        RVSML_dtw_macro,RVSML_dtw_micro,RVSML_dtw_acc,dtw_knn_average_time = NNClassifier_dtw(classnum,traindownset,trainsetnum,testdownsetdata,testsetdatanum,testsetlabel,options)
        RVSML_dtw_acc_1 = RVSML_dtw_acc[0]
            # logger.debug(vars())

        # print("OPW start")
        lambda0 = 0.01
        tic = time.time()
        L_opw , v_s_opw, dists_opw, Ts_opw = RVSML_OT_Learning_opw(trainset,templatenum,lambda0,options)
        RVSML_opw_time = time.time() - tic

            # v_s_opw = np.array([v[0] for v in v_s_opw])
            # real_v_opw = np.linalg.solve(L.T,v_s_opw.T)

            # print("OPW lerning done")
            ## classification with the learned metric
            # print("Classification start")
        traindownset = [0]*classnum
        testdownsetdata = [0]*testsetdatanum
        for j in range(classnum):
            traindownset[j] = [0]*trainsetnum[j]
            for m in range(trainsetnum[j]):
                traindownset[j][m] = np.dot(trainset[j][m] ,L_opw)

        for j in range(testsetdatanum):
            testdownsetdata[j] = np.dot(testsetdata[j], L_opw)

        RVSML_opw_macro,RVSML_opw_micro,RVSML_opw_acc,opw_knn_average_time = NNClassifier_opw(classnum,traindownset,trainsetnum,testdownsetdata,testsetdatanum,testsetlabel,options)
        RVSML_opw_acc_1 = RVSML_opw_acc[0]
            # print("OPW Classification done")

            # print("OPW done")
        if RVSML_dtw_acc_1<1 or RVSML_opw_acc_1 < 1:
        # if True:
            print('N:{},M:{},V:{}'.format(N,M,V), end=' ')
            print('templatenum:{}'.format(templatenum),end=' ')
            # print('Training time of RVSML instantiated by DTW is {:.4f} \n'.format(RVSML_dtw_time))
            # print('Classification using 1 nearest neighbor classifier with DTW distance:\n')
            # print('MAP macro is {:.4f}, micro is {:.4f} \n'.format(RVSML_dtw_macro, RVSML_dtw_micro))
            # print('dtw_knn_average_time is {:.4f} \n'.format(dtw_knn_average_time))
            print('DTW_Accuracy is {:.4f}'.format(RVSML_dtw_acc_1),end=' ')
            # for i,v in enumerate(v_s_dtw):
            #     print('ラベル{} Virtual Sequence:\n{}'.format(i+1,v))
            # for i in range(classnum):
            #     print('ラベル{} 訓練データ:'.format(i+1))
            #     for j in range(trainsetnum[i]):
            #         print('Virtual Sequenceとの距離:', dists_dtw[i][j])
            #         print(*traindownset[i][j].tolist(),sep='\n')
            #         # print('T:')
            #         # print(*Ts_dtw[i][j].tolist(),sep='\n')
            #         print()
            
            # print('Training time of RVSML instantiated by OPW is {:.4f} \n'.format(RVSML_opw_time))
            # print('Classification using 1 nearest neighbor classifier with OPW distance:\n')
            # print('MAP macro is {:.4f}, MAP micro is {:.4f} \n'.format(RVSML_opw_macro, RVSML_opw_micro))
            #     # print('Accuracy is .4f \n',RVSML_opw_acc_1)
            # print('opw_knn_average_time is {:.4f} \n'.format(opw_knn_average_time))
            print('OPW_Accuracy is {:.4f}'.format(RVSML_opw_acc_1))
            # for i,v in enumerate(v_s_opw):
            #     print('ラベル{} Virtual Sequence:\n{}'.format(i+1,v))

            # for i in range(classnum):
            #     print('ラベル{} 訓練データ:'.format(i+1))
            #     for j in range(trainsetnum[i]):
            #         print('Virtual Sequenceとの距離:', dists_opw[i][j])
            #         print(*traindownset[i][j].tolist(),sep='\n')
            #         # print('T:')
            #         # print(*Ts_opw[i][j].tolist(),sep='\n')
            #         print()

            # print('テスト:')
            # for i in range(testsetdatanum):
            #     print(*testdownsetdata[i].tolist(),sep='\n')
            #     print()


    # print("debug")