import numpy as np
import time
from align import *
from sklearn import metrics
import logging
import logging,os,ctypes
from multiprocessing import Value, Array, Process

def knn(i,task_per_cpu,k_num,k_pool,datainfo,options,Macro,Micro,rightnum):
    logger = logging.getLogger('ChaLearn{}Log'.format(options.method))
    start = i*task_per_cpu
    end = (i+1)*task_per_cpu
    if end > datainfo.test.setdatanum:
        end = datainfo.test.setdatanum
    print(i,start,end-1)
    for j in range(start,end):
        logger.info("j:{}".format(j))
        dis_totrain_scores = np.zeros(datainfo.train.setdatanum)
        for m2 in range(datainfo.train.setdatanum):
            #[Dist,T] = Sinkhorn_distance(trainsetdata[m2],testsetdata[j],lambda,0)
            if options.method == 'dtw':
                Dist,T = dtw2(datainfo.train.downsetdata[m2],datainfo.test.downsetdata[j])
            elif options.method == 'opw':
                Dist,T = OPW_w(datainfo.train.downsetdata[m2],datainfo.test.downsetdata[j],[],[],options,0)
            dis_totrain_scores[m2] = Dist

            if np.isnan(Dist):
                logger.info('NaN distance!')

        # distm = np.sort(dis_totrain_scores)
        index = np.argsort(dis_totrain_scores)

        for k_count in range(k_num):
            cnt = np.zeros(datainfo.classnum)
            for temp_i in range(k_pool[k_count]):
                ind = np.nonzero(datainfo.ClassLabel==datainfo.train.setdatalabel[index[temp_i]])
                cnt[ind] += 1

            # distm2 = np.max(cnt)
            ind = np.argmax(cnt)
            predict = datainfo.ClassLabel[ind]
            if predict==datainfo.test.setdatalabel[j]:
                rightnum[k_count] += 1

        temp_dis = -dis_totrain_scores
        temp_dis[np.nonzero(np.isnan(temp_dis))] = 0
        # logger.info("{:.1f}%".format(j/testsetdatanum*100))
        Macro[j] = metrics.average_precision_score(datainfo.train.setlabelfull[:,datainfo.test.setdatalabel[j]-1],temp_dis, 'macro')
        Micro[j] = metrics.average_precision_score(datainfo.train.setlabelfull[:,datainfo.test.setdatalabel[j]-1],temp_dis, 'micro')

def NNClassifier(datainfo,options):
    testsetdatanum = datainfo.test.setdatanum

    k_pool = [1]
    k_num = len(k_pool)
    Acc = np.zeros(k_num)

    tic = time.time()
    Macro = Value(ctypes.c_float * testsetdatanum)
    Micro = Value(ctypes.c_float * testsetdatanum)
    rightnum = Value(ctypes.c_uint * k_num)

    cpu_count = os.cpu_count()
    task_per_cpu = testsetdatanum//(cpu_count-1)+1
    inner_process = [
        Process(target=knn, args=(i,task_per_cpu,k_num,k_pool,datainfo,options,Macro,Micro,rightnum)) for i in range(cpu_count-1)
    ]

    for p in inner_process:
        p.start()
    for p in inner_process:
        p.join()

    rightnum = np.ctypeslib.as_array(rightnum.get_obj())
    Acc = rightnum/testsetdatanum
    Macro = np.ctypeslib.as_array(Macro.get_obj())
    Micro = np.ctypeslib.as_array(Micro.get_obj())
    macro = np.mean(Macro)
    micro = np.mean(Micro)

    knn_time = time.time()-tic
    knn_averagetime = knn_time/testsetdatanum
    # logger.info(vars())
    return macro,micro,Acc,knn_time,knn_averagetime