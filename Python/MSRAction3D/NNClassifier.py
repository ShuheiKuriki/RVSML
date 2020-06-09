import numpy as np
import time
from align import *
from sklearn import metrics
import logging,os,ctypes
from multiprocessing import Value, Array, Process

logger = logging.getLogger('MSRAction3DLog')

def knn(i,task_per_cpu,k_num,k_pool,dataset,options,Macro,Micro,rightnum):
    start = i*task_per_cpu
    end = (i+1)*task_per_cpu
    if end > dataset.testsetdatanum:
        end = dataset.testsetdatanum
    print(i,start,end-1)
    for j in range(start,end):
        logger.info("j:{}".format(j))
        dis_totrain_scores = np.zeros(dataset.trainsetdatanum)
        for m2 in range(dataset.trainsetdatanum):
            #[Dist,T] = Sinkhorn_distance(trainsetdata[m2],testsetdata[j],lambda,0)
            if options.method == 'dtw':
                Dist,T = dtw2(dataset.traindownsetdata[m2],dataset.testdownsetdata[j])
            elif options.method == 'opw':
                Dist,T = OPW_w(dataset.traindownsetdata[m2],dataset.testdownsetdata[j],[],[],options,0)
            logger.info(T)
            dis_totrain_scores[m2] = Dist
            if np.isnan(Dist):
                logger.info('NaN distance!')

        # distm = np.sort(dis_totrain_scores)
        index = np.argsort(dis_totrain_scores)

        for k_count in range(k_num):
            cnt = np.zeros(dataset.classnum)
            for temp_i in range(k_pool[k_count]):
                ind = np.nonzero(dataset.ClassLabel==dataset.trainsetdatalabel[index[temp_i]])
                cnt[ind] += 1

            # distm2 = np.max(cnt)
            ind = np.argmax(cnt)
            predict = dataset.ClassLabel[ind]
            if predict==dataset.testsetdatalabel[j]:
                rightnum[k_count] += 1

        temp_dis = -dis_totrain_scores
        temp_dis[np.nonzero(np.isnan(temp_dis))] = 0
        # logger.info("{:.1f}%".format(j/testsetdatanum*100))
        Macro[j] = metrics.average_precision_score(dataset.trainsetlabelfull[:,dataset.testsetdatalabel[j]-1],temp_dis, 'macro')
        Micro[j] = metrics.average_precision_score(dataset.trainsetlabelfull[:,dataset.testsetdatalabel[j]-1],temp_dis, 'micro')

def NNClassifier(dataset,options,method='dtw'):
    classnum = dataset.classnum
    k_pool = [1]
    k_num = len(k_pool)
    Acc = np.zeros(k_num)
    #dtw_knn_map = zeros(vidsetnum,)

    tic = time.time()
    # Macro = np.zeros(testsetdatanum)
    # Micro = np.zeros(testsetdatanum)
    # rightnum = np.zeros(k_num)
    Macro = Value(ctypes.c_float * dataset.testsetdatanum)
    Micro = Value(ctypes.c_float * dataset.testsetdatanum)
    rightnum = Value(ctypes.c_uint * k_num)

    cpu_count = os.cpu_count()
    task_per_cpu = dataset.testsetdatanum//(cpu_count-1)+1
    inner_process = [Process(target=knn, args=(i,task_per_cpu,k_num,k_pool,dataset,options,Macro,Micro,rightnum)) for i in range(cpu_count-1)]

    for p in inner_process:
        p.start()
    for p in inner_process:
        p.join()

    rightnum = np.ctypeslib.as_array(rightnum.get_obj())
    Acc = rightnum/dataset.testsetdatanum
    Macro = np.ctypeslib.as_array(Macro.get_obj())
    Micro = np.ctypeslib.as_array(Micro.get_obj())
    macro = np.mean(Macro)
    micro = np.mean(Micro)

    knn_time = time.time()-tic
    knn_averagetime = knn_time/dataset.testsetdatanum
    # logger.info(vars())
    return macro,micro,Acc,knn_averagetime