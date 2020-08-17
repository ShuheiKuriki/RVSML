import numpy as np
from scipy.io import loadmat
import time
from .Classifier import Classifier
from .RVSML_OT_Learning import RVSML_OT_Learning
import logging

def Evaluate_RVSML(dataset,options):
    logger = logging.getLogger('{}Log'.format(dataset.dataname))

    print("{} start".format(options.method))

    tic = time.time()
    virtual,L = RVSML_OT_Learning(dataset,options)
    dataset.virtual = virtual
    dataset.L = L
    RVSML_time = time.time() - tic
    print("{} lerning done".format(options.method))
    ## classification with the learned metric
    # print("Classification start")
    traindownset = [0]*dataset.classnum
    traindownsetdata = []
    testdownsetdata = [0]*dataset.testsetdatanum
    for j in range(dataset.classnum):
        traindownset[j] = [0]*dataset.trainsetnum[j]
        for m in range(dataset.trainsetnum[j]):
            downdata = np.dot(dataset.trainset[j][m],L)
            traindownset[j][m] = downdata
            traindownsetdata.append(downdata)

    for j in range(dataset.testsetdatanum):
        testdownsetdata[j] = np.dot(dataset.testsetdata[j], L)

    dataset.traindownset = traindownset
    dataset.traindownsetdata = traindownsetdata
    dataset.testdownsetdata = testdownsetdata

    # RVSML_macro,RVSML_micro
    accs,knn_time,knn_average_time = Classifier(dataset,options)
    # accs_1 = accs[0]

    print("{} Classification done".format(options.method))

    print("{} done".format(options.method))

    logger.info('Training time is {:.4f} \n'.format(RVSML_time))
    logger.info('Classification using 1 nearest neighbor classifier:\n')
    # logger.info('MAP macro is {:.4f}, MAP micro is {:.4f} \n'.format(RVSML_macro, RVSML_micro))
    # logger.info('Accuracy is .4f \n',accs_1)
    logger.info('knn_average_time is {:.4f} \n'.format(knn_average_time))
    logger.info('knn_total_time is {:.4f} \n'.format(knn_time))

    for acc in accs:
        logger.info('Accuracy is {:.4f} \n'.format(acc))

    # print("debug")