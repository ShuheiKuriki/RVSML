import numpy as np
import time
from .align import dtw2,OPW_w
# from sklearn import metrics
import logging,os
from multiprocessing import Value, Pool

def knn(i,task_per_cpu,k_num,k_pool,dataset,options,rightnum,Macro=None,Micro=None):
    logger = logging.getLogger('{}Log'.format(dataset.dataname))
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
            # logger.info(T)
            dis_totrain_scores[m2] = Dist
            if np.isnan(Dist):
                logger.info('NaN distance!')

        # distm = np.sort(dis_totrain_scores)
        index = np.argsort(dis_totrain_scores)
        cnt = [0]*dataset.classnum
        for k_count in range(k_num):
            for temp_i in range(k_pool[k_count]):
                ind = np.nonzero(dataset.ClassLabel==dataset.trainsetdatalabel[index[temp_i]])
                cnt[ind[0][0]] += 1

            # distm2 = np.max(cnt)
            ind = np.argmax(cnt)
            predict = ind+1
            if predict==dataset.testsetdatalabel[j]:
                rightnum[k_count] += 1

        # temp_dis = -dis_totrain_scores
        # temp_dis[np.nonzero(np.isnan(temp_dis))] = 0
        # logger.info("{:.1f}%".format(j/testsetdatanum*100))
        # Macro[j] = metrics.average_precision_score(dataset.trainsetlabelfull[:,dataset.testsetdatalabel[j]-1],temp_dis, 'macro')
        # Micro[j] = metrics.average_precision_score(dataset.trainsetlabelfull[:,dataset.testsetdatalabel[j]-1],temp_dis, 'micro')
    return rightnum

def virtual(i,task_per_cpu,dataset,options,rightnum,Macro=None,Micro=None,):
    logger = logging.getLogger('{}Log'.format(dataset.dataname))
    start = i*task_per_cpu
    end = (i+1)*task_per_cpu
    if end > dataset.testsetdatanum:
        end = dataset.testsetdatanum
    print(i,start,end-1)
    for j in range(start,end):
        logger.info("j:{}".format(j))
        dis_to_virtual = np.zeros(dataset.classnum)
        for c in range(dataset.classnum):
            #[Dist,T] = Sinkhorn_distance(trainsetdata[m2],testsetdata[j],lambda,0)
            if options.method == 'dtw':
                print('dtw')
                Dist,T = dtw2(dataset.virtual[c],dataset.testdownsetdata[j])
            elif options.method == 'opw':
                Dist,T = OPW_w(dataset.virtual[c],dataset.testdownsetdata[j],[],[],options,0)
            dis_to_virtual[c] = Dist
            # logger.info(T)
            if np.isnan(Dist):
                logger.info('NaN distance!')

        ind = np.argmin(dis_to_virtual)
        predict = ind+1
        if predict==dataset.testsetdatalabel[j]:
            rightnum += 1

        # temp_dis = -dis_to_virtual
        # temp_dis[np.nonzero(np.isnan(temp_dis))] = 0
        # # logger.info("{:.1f}%".format(j/testsetdatanum*100))
        # Macro[j] = metrics.average_precision_score(dataset.trainsetlabelfull[:,dataset.testsetdatalabel[j]-1],temp_dis, 'macro')
        # Micro[j] = metrics.average_precision_score(dataset.trainsetlabelfull[:,dataset.testsetdatalabel[j]-1],temp_dis, 'micro')
    return rightnum

def Classifier(dataset,options):
    #dtw_knn_map = zeros(vidsetnum,)

    tic = time.time()
    # Macro = np.zeros(testsetdatanum)
    # Micro = np.zeros(testsetdatanum)
    # rightnum = np.zeros(k_num)
    # Macro = Value(ctypes.c_float * dataset.testsetdatanum)
    # Micro = Value(ctypes.c_float * dataset.testsetdatanum)

    cpu_count = 2
    task_per_cpu = dataset.testsetdatanum//(cpu_count-1)+1

    if options.classify == 'knn':
        k_pool = [1]
        k_num = len(k_pool)
        rightnum = np.zeros(k_num)
        multi_args=[(i,task_per_cpu,k_num,k_pool,dataset,options,rightnum) for i in range(cpu_count-1)]
        pool = Pool(cpu_count-1)
        rightnums = pool.starmap(knn, multi_args)
        rightnum = np.zeros(k_num)
        for rs in rightnums:
            for i in range(k_num):
                rightnum[i] += rs[i]
    else:
        rightnum = 0
        multi_args=[(i,task_per_cpu,dataset,options,rightnum) for i in range(cpu_count-1)]
        pool = Pool(cpu_count-1)
        rightnums = pool.starmap(virtual,multi_args)
        rightnum = np.zeros(1)
        for rs in rightnums:
            rightnum[0] += rs
    
    Acc = rightnum/dataset.testsetdatanum
    # Macro = np.ctypeslib.as_array(Macro.get_obj())
    # Micro = np.ctypeslib.as_array(Micro.get_obj())
    # macro = np.mean(Macro)
    # micro = np.mean(Micro)

    knn_time = time.time()-tic
    knn_averagetime = knn_time/dataset.testsetdatanum
    # logger.info(vars())
    return Acc,knn_time,knn_averagetime