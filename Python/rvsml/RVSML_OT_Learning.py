"""Learning with numpy"""
import numpy as np
from .align import get_alignment
import logging

def RVSML_OT_Learning(dataset, options):
    """Learning"""
    logger = logging.getLogger('{}Log'.format(dataset.dataname))

    classnum = dataset.classnum
    templatenums = dataset.templatenums
    downdim = sum(templatenums)
    trainset = dataset.trainset
    dim = dataset.dim
    trainsetnum = dataset.trainsetnum

    virtual_sequence = [0]*classnum
    active_dim = -1
    for c in range(classnum):
        virtual_sequence[c] = np.zeros((templatenums[c], downdim))
        for a_d in range(templatenums[c]):
            active_dim += 1
            virtual_sequence[c][a_d, active_dim] = 1

    ## inilization
    R_A = np.zeros((dim, dim))
    R_B = np.zeros((dim, downdim))
    N = np.sum(trainsetnum)
    for c in range(classnum):
        for n in range(trainsetnum[c]):
            seqlen = np.shape(trainset[c][n])[0]
            # test = trainset[c][0][n]
            if options.init == 'uniform':
                T_ini = np.ones((seqlen, templatenums[c]))/(seqlen*templatenums[c])
            elif options.init == 'random':
                T_ini = np.zeros((seqlen, templatenums[c]))
                for i in range(seqlen):
                    for j in range(templatenums[c]):
                        T_ini[i, j] = (1+np.random.randn()*options.init_delta)/(seqlen*templatenums[c])
            elif options.init == 'normal':
                T_ini = np.zeros((seqlen, templatenums[c]))
                mid_para = np.sqrt(1/seqlen**2 + 1/templatenums[c]**2)
                for i in range(seqlen):
                    for j in range(templatenums[c]):
                        d = np.abs(i/seqlen - j/templatenums[c])/mid_para
                        T_ini[i, j] = np.exp(-d**2/(2*options.init_delta**2))/(options.init_delta*np.sqrt(2*np.pi))
            # print(T_ini)
            for i in range(seqlen):
                a = trainset[c][n][i, :]
                temp_ra = np.dot(a.reshape((len(a), 1)), a.reshape((1, len(a))))
                for j in range(templatenums[c]):
                    R_A += T_ini[i, j]*temp_ra
                    # logger.info(R_A.shape,R_B.shape,temp_ra.shape)
                    b = virtual_sequence[c][j, :]
                    R_B += T_ini[i, j]*np.dot(a.reshape((len(a), 1)), b.reshape((1, len(b))))

    R_I = R_A + options.lambda0 * N * np.eye(dim)
    #L = inv(R_I) * R_B
    L = np.linalg.solve(R_I, R_B)

    print("initialization done")
    print("update start")
    ## update
    loss_old = 10**8
    for nIter in range(options.max_iters):
        loss = 0
        R_A = np.zeros((dim, dim))
        R_B = np.zeros((dim, downdim))
        N = np.sum(trainsetnum)
        # Ts = [[0]*trainsetnum[c] for _ in range(classnum)]
        for c in range(classnum):
            for n in range(trainsetnum[c]):
                # logger.info(trainset[c][n])
                seqlen = np.shape(trainset[c][n])[0]
                dist, T = get_alignment(np.dot(trainset[c][n], L), virtual_sequence[c], options)
                loss += dist
                # Ts[c][n] = T
                for i in range(seqlen):
                    a = trainset[c][n][i, :]
                    temp_ra = np.dot(a.reshape((len(a), 1)), a.reshape((1, len(a))))
                    for j in range(templatenums[c]):
                        b = virtual_sequence[c][j, :]
                        R_A += T[i, j] * temp_ra
                        R_B += T[i, j] * np.dot(a.reshape((len(a), 1)), b.reshape((1, len(b))))
        # logger.info(Ts)
        loss = loss/N + np.trace(np.dot(L.T, L))
        logger.info("iteration: %d, loss:%.5f", nIter, loss/N)
        if np.abs(loss - loss_old) < options.err_limit:
            logger.info(T)
            break
        loss_old = loss

        R_I = R_A + options.lambda0*N*np.eye(dim)
        #L = inv(R_I) * R_B
        L = np.linalg.solve(R_I, R_B)
    # logger.info(time.time()-tic)
    # for c in range(classnum):
        # for n in range(trainsetnum[c]):
            # logger.info(dataset.sentences[c])
            # logger.info(dataset.orders[c][n])
            # logger.info(Ts[c][n])
    dataset.L = L
    dataset.virtual = virtual_sequence
    return dataset
