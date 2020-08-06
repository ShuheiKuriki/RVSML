import numpy as np
import time
from .align import get_alignment
# from sklearn import metrics
import logging,os

def Classifier(dataset,options):
    logger = logging.getLogger('{}Log'.format(dataset.dataname))
    tic = time.time()
    # Macro = np.zeros(testsetdatanum)
    # Micro = np.zeros(testsetdatanum)
    # rightnum = np.zeros(k_num)
    # Macro = Value(ctypes.c_float * dataset.testsetdatanum)
    # Micro = Value(ctypes.c_float * dataset.testsetdatanum)

    if options.classify == 'knn':
        k_pool = [1]
        k_num = len(k_pool)
        rightnum = np.zeros(k_num)
        for j in range(dataset.testsetdatanum):
            print("j:{}".format(j))
            dis_totrain_scores = np.zeros(dataset.trainsetdatanum)
            for m2 in range(dataset.trainsetdatanum):
                Dist,T = get_alignment(dataset.traindownsetdata[m2],dataset.testdownsetdata[j],options)
                # logger.info(T)
                dis_totrain_scores[m2] = Dist
                if np.isnan(Dist):
                    logger.info('NaN distance!')

            index = np.argsort(dis_totrain_scores)
            cnt = [0]*dataset.classnum
            for k_count in range(len(k_pool)):
                for temp_i in range(k_pool[k_count]):
                    ind = np.nonzero(dataset.ClassLabel==dataset.trainsetdatalabel[index[temp_i]])
                    cnt[ind[0][0]] += 1

                predict = np.argmax(cnt)+1
                if predict==dataset.testsetdatalabel[j]:
                    rightnum[k_count] += 1
    elif options.classify == 'virtual':
        rightnum = 0
        for j in range(dataset.testsetdatanum):
            print("j:{}".format(j))
            dis_to_virtual = np.zeros(dataset.classnum)
            for c in range(dataset.classnum):
                Dist,T = get_alignment(dataset.virtual[c],dataset.testdownsetdata[j],options)
            #     logger.info(T)
                dis_to_virtual[c] = Dist
                if np.isnan(Dist):
                    logger.info('NaN distance!')

            predict = np.argmin(dis_to_virtual)+1
            if predict==dataset.testsetdatalabel[j]:
                rightnum += 1
    
    Acc = rightnum/dataset.testsetdatanum
    # Macro = np.ctypeslib.as_array(Macro.get_obj())
    # Micro = np.ctypeslib.as_array(Micro.get_obj())
    # macro = np.mean(Macro)
    # micro = np.mean(Micro)

    knn_time = time.time()-tic
    knn_averagetime = knn_time/dataset.testsetdatanum
    # logger.info(vars())
    return Acc,knn_time,knn_averagetime