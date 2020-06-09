import numpy as np
from align import *
import logging

logger = logging.getLogger('MSRAction3DLog')

def RVSML_OT_Learning(dataset,templatenum,options,method='dtw'):
    max_nIter = options.max_iters
    err_limit = options.err_limit

    classnum = dataset.classnum
    downdim = classnum*templatenum
    trainset = dataset.trainset
    dim = dataset.dim
    trainsetnum = dataset.trainsetnum

    virtual_sequence = [0]*classnum
    active_dim = -1
    for c in range(classnum):
        virtual_sequence[c] = np.zeros((templatenum,downdim))
        for a_d in range(templatenum):
            active_dim = active_dim + 1
            virtual_sequence[c][a_d,active_dim] = 1

    ## inilization
    R_A = np.zeros((dim,dim))
    R_B = np.zeros((dim,downdim))
    N = np.sum(trainsetnum)
    for c in range(classnum):
        for n in range(trainsetnum[c]):
            seqlen = np.shape(trainset[c][n])[0]
            # test = trainset[c][0][n]
            T_ini = np.ones((seqlen,templatenum))/(seqlen*templatenum)
            for i in range(seqlen):
                a = trainset[c][n][i,:]
                temp_ra = np.dot(a.reshape((len(a),1)), a.reshape((1,len(a))))
                for j in range(templatenum):
                    R_A += T_ini[i,j]*temp_ra
                    # logger.info(R_A.shape,R_B.shape,temp_ra.shape)
                    b = virtual_sequence[c][j,:]
                    R_B += T_ini[i,j]*np.dot(a.reshape((len(a),1)), b.reshape((1, len(b))))

    R_I = R_A + options.lambda0 * N * np.eye(dim)
    #L = inv(R_I) * R_B
    L = np.linalg.solve(R_I,R_B)

    logger.info("initialization done")
    logger.info("update start")
    ## update
    loss_old = 10**8
    for nIter in range(max_nIter):
        logger.info("iteration:{}".format(nIter))
        loss = 0
        R_A = np.zeros((dim,dim))
        R_B = np.zeros((dim,downdim))
        N = np.sum(trainsetnum)
        for c in range(classnum):
            for n in range(trainsetnum[c]):
                seqlen = np.shape(trainset[c][n])[0]
                if method == 'dtw':
                    dist, T = dtw2(np.dot(trainset[c][n],L), virtual_sequence[c])
                elif method == 'opw':
                    dist, T = OPW_w(np.dot(trainset[c][n],L), virtual_sequence[c],[],[],options,0)
                loss += dist
                for i in range(seqlen):
                    a = trainset[c][n][i,:]
                    temp_ra = np.dot(a.reshape((len(a),1)), a.reshape((1,len(a))))
                    for j in range(templatenum):
                        b = virtual_sequence[c][j,:]
                        R_A += T[i,j]*temp_ra
                        R_B += T[i,j]*np.dot(a.reshape((len(a),1)), b.reshape((1, len(b))))

        loss = loss/N + np.trace(np.dot(L.T,L))
        if np.abs(loss - loss_old) < err_limit:
            break
        else:
            loss_old = loss


        R_I = R_A + options.lambda0*N*np.eye(dim)
        #L = inv(R_I) * R_B
        L = np.linalg.solve(R_I,R_B)
    # logger.info(time.time()-tic)
    return L