import numpy as np
import time
from align import *
from sklearn import metrics

def NNClassifier(classnum,trainset,trainsetnum,testsetdata,testsetdatanum,testsetlabel,options,method='dtw'):
    if method=='opw':
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
            sample_count += 1

    testsetlabelori = testsetlabel
    testsetlabel = getLabel(testsetlabelori)
    trainsetlabelfull = getLabel(trainsetlabel)

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
    # print(Macro.shape)
    Micro = np.zeros(testsetdatanum)
    # print(trainsetlabelfull)
    rightnum = np.zeros(k_num)
    for j in range(testsetdatanum):
        # print("j:",j)
        # if j==10:
        #     print("stop")
        for m2 in range(trainsetdatanum):
            #[Dist,T] = Sinkhorn_distance(trainsetdata[m2],testsetdata[j],lambda,0)
            Dist,T = OPW_w(trainsetdata[m2],testsetdata[j],[],[],options,0)
            #[dist, T] = OPW_w(ct_barycenter[c].supp',ct_barycenter[c2].supp',ct_barycenter[c].w',ct_barycenter[c2].w',lambda1,lambda2,delta,0)
            dis_totrain_scores[m2,j] = Dist

            if np.isnan(Dist):
                print('NaN distance!')


        distm = np.sort(dis_totrain_scores[:,j])
        index = np.argsort(dis_totrain_scores[:,j])

        for k_count in range(k_num):
            cnt = np.zeros(classnum)
            for temp_i in range(k_pool[k_count]):
                ind = np.nonzero(ClassLabel==trainsetlabel[index[temp_i]])
                cnt[ind] += 1

            distm2 = np.max(cnt)
            ind = np.argmax(cnt)
            predict = ClassLabel[ind]
            if predict==testsetlabelori[j]:
                rightnum[k_count] += 1
            # else:
            #     print('truelabel:',testsetlabelori[j],'predict:',predict)
            #     print(*testsetdata[j].tolist(),sep='\n')
            #     print(dis_totrain_scores[:,j])
            #     print()

        temp_dis = -dis_totrain_scores[:,j]
        temp_dis[np.nonzero(np.isnan(temp_dis))] = 0
        # print("{:.1f}%".format(j/testsetdatanum*100))
        Macro[j] = metrics.average_precision_score(trainsetlabelfull[:,testsetlabelori[j]-1],temp_dis, 'macro')
        Micro[j] = metrics.average_precision_score(trainsetlabelfull[:,testsetlabelori[j]-1],temp_dis, 'micro')

    Acc = rightnum/testsetdatanum
    macro = np.mean(Macro)
    micro = np.mean(Micro)

    knn_time = time.time()-tic
    knn_averagetime = knn_time/testsetdatanum
    # print(vars())
    return macro,micro,Acc,knn_averagetime

def NNClassifier_dtw(classnum,trainset,trainsetnum,
                    testsetdata,testsetdatanum,testsetlabel,options):

    trainsetdatanum = np.sum(trainsetnum)
    trainsetdata = [0]*trainsetdatanum
    trainsetlabel = np.zeros(trainsetdatanum)
    sample_count = 0
    for c in range(classnum):
        for per_sample_count in range(trainsetnum[c]):
            trainsetdata[sample_count] = trainset[c][per_sample_count]
            trainsetlabel[sample_count] = c+1
            sample_count += 1

    testsetlabelori = testsetlabel
    # print("testsetlabel:",testsetlabel)
    testsetlabel = getLabel(testsetlabel)
    # print("trainsetlabel:",trainsetlabel)
    trainsetlabelfull = getLabel(trainsetlabel)

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
    # print(Macro.shape)
    Micro = np.zeros(testsetdatanum)
    # print(trainsetlabelfull)
    rightnum = np.zeros(k_num)
    for j in range(testsetdatanum):
        # print("j:", j)
        for m2 in range(trainsetdatanum):
            # print(trainsetdata[m2].shape,testsetdata[j].shape)
            Dist,T = dtw2(trainsetdata[m2], testsetdata[j])
            #[Dist,D,matchlength,w] = dtw2_fast(trainsetdata[m2]',testsetdata[j]')
            #[Dist,T] = Sinkhorn_distance(trainsetdata[m2],testsetdata[j],lambda,0)
            #[Dist,T] = OPW_w(trainsetdata[m2],testsetdata[j],[],[],lambda1,lambda2,delta,0)
            if np.isnan(Dist):
                print('NaN distance!')

            #[dist, T] = OPW_w(ct_barycenter[c].supp',ct_barycenter[c2].supp',ct_barycenter[c].w',ct_barycenter[c2].w',lambda1,lambda2,delta,0)
            dis_totrain_scores[m2,j] = Dist

        distm = np.sort(dis_totrain_scores[:,j])
        index = np.argsort(dis_totrain_scores[:,j])

        for k_count in range(k_num):
            cnt = np.zeros(classnum)
            for temp_i in range(k_pool[k_count]):
                ind = np.nonzero(ClassLabel==trainsetlabel[index[temp_i]])
                cnt[ind] += 1

            distm2 = np.max(cnt)
            ind = np.argmax(cnt)
            predict = ClassLabel[ind]
            # print(predict)
            # predict = predict
            if predict==testsetlabelori[j]:
                rightnum[k_count] += 1

        temp_dis = -dis_totrain_scores[:,j]
        temp_dis[np.nonzero(np.isnan(temp_dis))] = 0
        # print("[:.1f]#".format(j/testsetdatanum*100))
        Macro[j] = metrics.average_precision_score(trainsetlabelfull[:,testsetlabelori[j]-1],temp_dis, 'macro')
        Micro[j] = metrics.average_precision_score(trainsetlabelfull[:,testsetlabelori[j]-1],temp_dis, 'micro')
        # a,b,info = vl_pr(trainsetlabelfull[:,testsetlabelori[j]],temp_dis)
        # dis_ap[j] = info.ap


    Acc = rightnum/testsetdatanum
    macro = np.mean(Macro)
    micro = np.mean(Micro)

    knn_time = time.time()-tic
    knn_averagetime = knn_time/testsetdatanum
    # print(vars())
    return macro,micro,Acc,knn_averagetime

def getLabel(classid):
    p = int(max(classid))
    # print(p)
    X = np.zeros((np.size(classid),p))-1
    for i in range(p):
        indx = np.nonzero(classid == i+1)
        X[indx,i] = 1
    return X