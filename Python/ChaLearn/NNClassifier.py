import numpy as np
import time
from align import *
from sklearn import metrics
import logging

def NNClassifier_opw(classnum,trainset,trainsetnum,
                    testsetdata,testsetdatanum,testsetlabel,options):

    logger = logging.getLogger('ChaLearnOPWLog')
    lambda1 = options.lambda1
    lambda2 = options.lambda2
    delta = options.delta

    trainsetdatanum = np.sum(trainsetnum)
    trainsetdata = [0]*trainsetdatanum
    trainsetlabel = np.zeros(trainsetdatanum)
    sample_count = 0
    for c in range(classnum):
        for per_sample_count in range(trainsetnum[c]):
            trainsetdata[sample_count] = trainset[c][per_sample_count]
            trainsetlabel[sample_count] = c+1
            sample_count = sample_count + 1

    testsetlabelori = testsetlabel
    testsetlabel = getLabel(testsetlabelori)
    trainsetlabelfull = getLabel(trainsetlabel)

    #save('./datamat/trainsetdata.mat','trainsetdata')
    #save('./datamat/trainsetlabel.mat','trainsetlabel')

    # k_pool = [1, 3, 5, 7, 9, 11, 15, 30]
    k_pool = [1]
    k_num = len(k_pool)
    Acc = np.zeros(k_num)
    #dtw_knn_map = zeros(vidsetnum,)
    #testsetlabelori = testsetlabel
    #testsetlabel = getLabel(testsetlabelori)

    #trainsetdatanum = sum(trainsetnum)
    tic = time.time()
    dis_totrain_scores = np.zeros((trainsetdatanum,testsetdatanum))
    ClassLabel = np.arange(classnum).T+1
    # dis_ap = np.zeros(testsetdatanum)
    Macro = np.zeros(testsetdatanum)
    # logger.info(Macro.shape)
    Micro = np.zeros(testsetdatanum)
    # logger.info(trainsetlabelfull)
    rightnum = np.zeros(k_num)
    for j in range(testsetdatanum):
        logger.info("j:{}".format(j))
        for m2 in range(trainsetdatanum):
            #[Dist,D,matchlength,w] = dtw2(trainsetdata[m2]',testsetdata[j]')
            #[Dist,T] = Sinkhorn_distance(trainsetdata[m2],testsetdata[j],lambda,0)
            Dist,T = OPW_w(trainsetdata[m2],testsetdata[j],[],[],lambda1,lambda2,delta,0)
            #[dist, T] = OPW_w(ct_barycenter[c].supp',ct_barycenter[c2].supp',ct_barycenter[c].w',ct_barycenter[c2].w',lambda1,lambda2,delta,0)
            dis_totrain_scores[m2,j] = Dist

            if np.isnan(Dist):
                logger.info('NaN distance!')


        distm = np.sort(dis_totrain_scores[:,j])
        index = np.argsort(dis_totrain_scores[:,j])

        for k_count in range(k_num):
            cnt = np.zeros(classnum)
            for temp_i in range(k_pool[k_count]):
                ind = np.nonzero(ClassLabel==trainsetlabel[index[temp_i]])
                cnt[ind] = cnt[ind]+ 1/(temp_i+1)

            distm2 = np.max(cnt)
            ind = np.argmax(cnt)
            predict = ClassLabel[ind]
            if predict==testsetlabelori[j]:
                rightnum[k_count] = rightnum[k_count] + 1

        temp_dis = -dis_totrain_scores[:,j]
        temp_dis[np.nonzero(np.isnan(temp_dis))] = 0
        # logger.info("{:.1f}%".format(j/testsetdatanum*100))
        Macro[j] = metrics.average_precision_score(trainsetlabelfull[:,testsetlabelori[j]-1],temp_dis, 'macro')
        Micro[j] = metrics.average_precision_score(trainsetlabelfull[:,testsetlabelori[j]-1],temp_dis, 'micro')

    Acc = rightnum/testsetdatanum
    macro = np.mean(Macro)
    micro = np.mean(Micro)

    knn_time = time.time()-tic
    knn_averagetime = knn_time/testsetdatanum
    # logger.info(vars())
    return macro,micro,Acc,knn_averagetime

def NNClassifier_dtw(classnum,trainset,trainsetnum,
                        testsetdata,testsetdatanum,testsetlabel,options):

    # lambda1 = options.lambda1
    # lambda2 = options.lambda2
    # delta = options.delta

    logger = logging.getLogger('ChaLearnDTWLog')
    trainsetdatanum = np.sum(trainsetnum)
    trainsetdata = [0]*trainsetdatanum
    trainsetlabel = np.zeros(trainsetdatanum)
    sample_count = 0
    for c in range(classnum):
        for per_sample_count in range(trainsetnum[c]):
            trainsetdata[sample_count] = trainset[c][per_sample_count]
            trainsetlabel[sample_count] = c+1
            sample_count = sample_count + 1

    testsetlabelori = testsetlabel
    # logger.info("testsetlabel:",testsetlabel)
    testsetlabel = getLabel(testsetlabel)
    # logger.info("trainsetlabel:",trainsetlabel)
    trainsetlabelfull = getLabel(trainsetlabel)

    #save('./datamat/trainsetdata.mat','trainsetdata')
    #save('./datamat/trainsetlabel.mat','trainsetlabel')

    # k_pool = [1, 3, 5, 7, 9, 11, 15, 30]
    k_pool = [1]
    k_num = len(k_pool)
    Acc = np.zeros(k_num)
    #dtw_knn_map = zeros(vidsetnum,)
    #testsetlabelori = testsetlabel
    #testsetlabel = getLabel(testsetlabelori)

    #trainsetdatanum = sum(trainsetnum)
    tic = time.time()
    dis_totrain_scores = np.zeros((trainsetdatanum,testsetdatanum))
    ClassLabel = np.arange(classnum).T+1
    # dis_ap = np.zeros(testsetdatanum)
    Macro = np.zeros(testsetdatanum)
    # logger.info(Macro.shape)
    Micro = np.zeros(testsetdatanum)
    # logger.info(trainsetlabelfull)
    rightnum = np.zeros(k_num)
    for j in range(testsetdatanum):
        logger.info("j:{}".format(j))
        for m2 in range(trainsetdatanum):
            # logger.info(trainsetdata[m2].shape,testsetdata[j].shape)
            Dist,T = dtw2(trainsetdata[m2], testsetdata[j])
            #[Dist,D,matchlength,w] = dtw2_fast(trainsetdata[m2]',testsetdata[j]')
            #[Dist,T] = Sinkhorn_distance(trainsetdata[m2],testsetdata[j],lambda,0)
            #[Dist,T] = OPW_w(trainsetdata[m2],testsetdata[j],[],[],lambda1,lambda2,delta,0)
            if np.isnan(Dist):
                logger.info('NaN distance!')

            #[dist, T] = OPW_w(ct_barycenter[c].supp',ct_barycenter[c2].supp',ct_barycenter[c].w',ct_barycenter[c2].w',lambda1,lambda2,delta,0)
            dis_totrain_scores[m2,j] = Dist

        distm = np.sort(dis_totrain_scores[:,j])
        index = np.argsort(dis_totrain_scores[:,j])

        for k_count in range(k_num):
            cnt = np.zeros(classnum)
            for temp_i in range(k_pool[k_count]):
                ind = np.nonzero(ClassLabel==trainsetlabel[index[temp_i]])
                cnt[ind] = cnt[ind]+1

            distm2 = np.max(cnt)
            ind = np.argmax(cnt)
            predict = ClassLabel[ind]
            # logger.info(predict)
            # predict = predict
            if predict==testsetlabelori[j]:
                rightnum[k_count] = rightnum[k_count] + 1

        temp_dis = -dis_totrain_scores[:,j]
        temp_dis[np.nonzero(np.isnan(temp_dis))] = 0
        # logger.info("[:.1f]#".format(j/testsetdatanum*100))
        Macro[j] = metrics.average_precision_score(trainsetlabelfull[:,testsetlabelori[j]-1],temp_dis, 'macro')
        Micro[j] = metrics.average_precision_score(trainsetlabelfull[:,testsetlabelori[j]-1],temp_dis, 'micro')
        # a,b,info = vl_pr(trainsetlabelfull[:,testsetlabelori[j]],temp_dis)
        # dis_ap[j] = info.ap


    Acc = rightnum/testsetdatanum
    macro = np.mean(Macro)
    micro = np.mean(Micro)

    knn_time = time.time()-tic
    knn_averagetime = knn_time/testsetdatanum
    # logger.info(vars())
    return macro,micro,Acc,knn_averagetime

def getLabel(classid):
    p = int(max(classid))
    # logger.info(p)
    X = np.zeros((np.size(classid),p))-1
    for i in range(p):
        indx = np.nonzero(classid == i+1)
        X[indx,i] = 1
    return X