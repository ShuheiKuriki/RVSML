"""並列化なしの分類コード"""
import time, logging, os, torch
from .align import get_alignment
# from sklearn import metrics

def Classifier(dataset, options):
    """分類する"""
    logger = logging.getLogger('{}Log'.format(dataset.dataname))
    tic = time.time()
    # Macro = torch.zeros(testsetdatanum)
    # Micro = torch.zeros(testsetdatanum)
    # rightnum = torch.zeros(k_num)
    # Macro = Value(ctypes.c_float * dataset.testsetdatanum)
    # Micro = Value(ctypes.c_float * dataset.testsetdatanum)

    if options.classify == 'knn':
        # k_pool = [1]
        # k_num = len(k_pool)
        # rightnum = torch.zeros(k_num)
        rightnum = 0
        for j in range(dataset.testsetdatanum):
            print("j:{}".format(j))
            dis_totrain_scores = torch.zeros(dataset.trainsetdatanum)
            for m2 in range(dataset.trainsetdatanum):
                Dist, T = get_alignment(dataset.traindownsetdata[m2], dataset.testdownsetdata[j], options)
                # logger.info(T)
                dis_totrain_scores[m2] = Dist
                if torch.isnan(Dist):
                    logger.info('NaN distance!')

            index = torch.argsort(dis_totrain_scores)
            # cnt = torch.zeros(dataset.classnum)
            # for k_count in range(len(k_pool)):
                # for temp_i in range(k_pool[k_count]):
            ind = torch.nonzero(dataset.ClassLabel == dataset.trainsetdatalabel[index[0]], as_tuple=True)
                    # cnt[ind[0]] += 1

            predict = ind[0]+1
            if predict == dataset.testsetdatalabel[j]:
                # rightnum[k_count] += 1
                rightnum += 1
    elif options.classify == 'virtual':
        rightnum = 0
        for j in range(dataset.testsetdatanum):
            print("j:{}".format(j))
            dis_to_virtual = torch.zeros(dataset.classnum)
            for c in range(dataset.classnum):
                Dist, T = get_alignment(dataset.virtual[c], dataset.testdownsetdata[j], options)
            #     logger.info(T)
                dis_to_virtual[c] = Dist
                if torch.isnan(Dist):
                    logger.info('NaN distance!')

            predict = torch.argmin(dis_to_virtual)+1
            if predict == dataset.testsetdatalabel[j]:
                rightnum += 1

    Acc = rightnum/dataset.testsetdatanum
    # Macro = np.ctypeslib.as_array(Macro.get_obj())
    # Micro = np.ctypeslib.as_array(Micro.get_obj())
    # macro = np.mean(Macro)
    # micro = np.mean(Micro)

    knn_time = time.time()-tic
    knn_averagetime = knn_time/dataset.testsetdatanum
    # logger.info(vars())
    return Acc, knn_time, knn_averagetime
