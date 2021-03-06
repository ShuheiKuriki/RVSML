import numpy as np
from .align import OPW_w,dtw2
import logging,torch


def RVSML_OT_Learning(dataset,options,params):
    logger = logging.getLogger('{}Log'.format(dataset.dataname))

    classnum = dataset.classnum
    templatenum = options.templatenum
    downdim = classnum*templatenum
    trainset = dataset.trainset
    dim = dataset.dim
    trainsetnum = dataset.trainsetnum

    virtual_sequence = [0]*classnum
    active_dim = -1
    for c in range(classnum):
        virtual_sequence[c] = np.zeros((templatenum,downdim))
        for a_d in range(templatenum):
            active_dim += 1
            virtual_sequence[c][a_d,active_dim] = 1

    ## inilization
    if options.initialize==True:
        R_A = np.zeros((dim,dim))
        R_B = np.zeros((dim,downdim))
        N = np.sum(trainsetnum)
        for c in range(classnum):
            for n in range(trainsetnum[c]):
                seqlen = np.shape(trainset[c][n])[0]
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

    else:
        L = dataset.trans_mat.to('cpu').detach().numpy().copy()

    logger.info("initialization done")
    logger.info("update start")
    ## update
    loss_old = 10**8
    for nIter in range(options.max_nIter):
        loss = 0
        R_A = np.zeros((dim,dim))
        R_B = np.zeros((dim,downdim))
        N = np.sum(trainsetnum)
        for c in range(classnum):
            for n in range(trainsetnum[c]):
                seqlen = np.shape(trainset[c][n])[0]
                if options.method == 'dtw':
                    dist, T = dtw2(np.dot(trainset[c][n],L), virtual_sequence[c])
                elif options.method == 'opw':
                    dist, T = OPW_w(np.dot(trainset[c][n],L), virtual_sequence[c],[],[],options,0)
                loss += dist
                for i in range(seqlen):
                    a = trainset[c][n][i,:]
                    temp_ra = np.dot(a.reshape((len(a),1)), a.reshape((1,len(a))))
                    for j in range(templatenum):
                        b = virtual_sequence[c][j,:]
                        R_A += T[i,j]*temp_ra
                        R_B += T[i,j]*np.dot(a.reshape((len(a),1)), b.reshape((1, len(b))))
        logger.info("iteration:{}, loss:{}".format(nIter,loss/N))
        loss = loss/N + np.trace(np.dot(L.T,L))
        if np.abs(loss - loss_old) < options.err_limit:
            break
        else:
            loss_old = loss

        R_I = R_A + options.lambda0 * N * np.eye(dim)
        #L = inv(R_I) * R_B
        L = np.linalg.solve(R_I,R_B)
    device = 'cuda' if params.cuda else 'cpu'
    dataset.trans_mat = torch.from_numpy(L.astype(np.float32)).clone().to(device)
    dataset.virtual = virtual_sequence
    return dataset