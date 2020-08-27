"""RVSMLを実行する"""
import torch, time, logging
from .Classifier2 import Classifier
from .RVSML_OT_Learning import RVSML_OT_Learning

def run_RVSML(dataset, options):
    """関数"""
    logger = logging.getLogger('{}Log'.format(dataset.dataname))

    print("{} start".format(options.method))

    tic = time.time()
    dataset = RVSML_OT_Learning(dataset, options)
    learning_time = time.time() - tic
    print("{} lerning done".format(options.method))
    ## classification with the learned metric
    # print("Classification start")
    traindownset = [0]*dataset.classnum
    traindownsetdata = []
    testdownsetdata = [0]*dataset.testsetdatanum
    for j in range(dataset.classnum):
        traindownset[j] = [0]*dataset.trainsetnum[j]
        for m in range(dataset.trainsetnum[j]):
            downdata = torch.mm(dataset.trainset[j][m], dataset.L)
            traindownset[j][m] = downdata
            traindownsetdata.append(downdata)

    for j in range(dataset.testsetdatanum):
        testdownsetdata[j] = torch.mm(dataset.testsetdata[j], dataset.L)

    dataset.traindownset = traindownset
    dataset.traindownsetdata = traindownsetdata
    dataset.testdownsetdata = testdownsetdata

    logger.info('Training time is {:.4f} \n'.format(learning_time))

    if 'Classify':
        knn_acc, knn_time, knn_average_time = 0, 0, 0
        # options.classify = 'knn'
        # knn_acc, knn_time, knn_average_time = Classifier(dataset, options)
        options.classify = 'virtual'
        virtual_acc, virtual_time, virtual_average_time = Classifier(dataset, options)

        print("{} Classification done".format(options.method))
        print("{} done".format(options.method))

        # logger.info('Classification using 1 nearest neighbor classifier:\n')
        # logger.info('MAP macro is {:.4f}, MAP micro is {:.4f} \n'.format(RVSML_macro, RVSML_micro))
        # logger.info('Accuracy is .4f \n',accs_1)
        logger.info('lambda0 is %.5f \n', options.lambda0)
        logger.info('knn_average_time is %.4f \n', knn_average_time)
        logger.info('knn_total_time is %.4f \n', knn_time)
        logger.info('virtual_average_time is %.4f \n', virtual_average_time)
        logger.info('virtual_total_time is %.4f \n', virtual_time)

        logger.info('knn accuracy is %.4f \n', knn_acc)
        logger.info('virtual accuracy is %.4f \n', virtual_acc)

    return dataset, knn_acc, virtual_acc
