import numpy as np
from align import *

def RVSML_OT_Learning_opw(trainset,templatenum,lambda0,options):
    # delta = 1
    # lambda1 = 50
    # lambda2 = 0.1
    # max_nIter = 200
    # err_limit = 10**(-6)

    delta = options.delta
    lambda1 = options.lambda1
    lambda2 = options.lambda2
    max_nIter = options.max_iters
    err_limit = options.err_limit

    classnum = len(trainset)
    downdim = classnum*templatenum
    dim = np.shape(trainset[0][0])[1]

    trainsetnum = np.zeros(classnum, dtype=int)
    virtual_sequence = [0]*classnum
    active_dim = 0
    for c in range(classnum):
        trainsetnum[c] = len(trainset[c])
        virtual_sequence[c] = np.zeros((templatenum,downdim))
        for a_d in range(templatenum):
            virtual_sequence[c][a_d,active_dim] = 1
            active_dim += 1

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
                    # print(R_A.shape,R_B.shape,temp_ra.shape)
                    b = virtual_sequence[c][j,:]
                    R_B += T_ini[i,j]*np.dot(a.reshape((len(a),1)), b.reshape((1, len(b))))

    R_I = R_A + lambda0 * N * np.eye(dim)
    #L = inv(R_I) * R_B
    L = np.linalg.solve(R_I,R_B)

    # print("initialization done")
    # print("update start")
    ## update
    loss_old = 10**8
    dists = [[0]*n for n in trainsetnum]
    Ts = [[0]*n for n in trainsetnum]
    for nIter in range(max_nIter):
        # if nIter == 1:
        #     print("stop")
        # print("iteration:", nIter)
        loss = 0
        R_A = np.zeros((dim,dim))
        R_B = np.zeros((dim,downdim))
        N = np.sum(trainsetnum)
        for c in range(classnum):
            for n in range(trainsetnum[c]):
                seqlen = np.shape(trainset[c][n])[0]
                dist, T = OPW_w(np.dot(trainset[c][n],L), virtual_sequence[c],[],[],lambda1,lambda2,delta,0)
                Ts[c][n] = T
                dists[c][n] = dist
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


        R_I = R_A + lambda0*N*np.eye(dim)
        #L = inv(R_I) * R_B
        L = np.linalg.solve(R_I,R_B)
    # print(time.time()-tic)
    return L, virtual_sequence,dists,Ts

def RVSML_OT_Learning_dtw(trainset,templatenum,lambda0,options):
    # delta = 1
    # lambda1 = 50
    # lambda2 = 0.1
    # max_nIter = 200
    # err_limit = 10**(-6)

    # delta = options.delta
    # lambda1 = options.lambda1
    # lambda2 = options.lambda2
    max_nIter = options.max_iters
    err_limit = options.err_limit

    classnum = len(trainset)
    downdim = classnum*templatenum
    dim = np.shape(trainset[0][0])[1]

    trainsetnum = np.zeros(classnum, dtype=int)
    virtual_sequence = [0]*classnum
    active_dim = 0
    for c in range(classnum):
        trainsetnum[c] = len(trainset[c])
        virtual_sequence[c] = np.zeros((templatenum,downdim))
        for a_d in range(templatenum):
            virtual_sequence[c][a_d,active_dim] = 1
            active_dim += 1

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
                    # print(R_A.shape,R_B.shape,temp_ra.shape)
                    b = virtual_sequence[c][j,:]
                    R_B += T_ini[i,j]*np.dot(a.reshape((len(a),1)), b.reshape((1, len(b))))

    R_I = R_A + lambda0 * N * np.eye(dim)
    #L = inv(R_I) * R_B
    L = np.linalg.solve(R_I,R_B)

    # print("initialization done")
    # print("update start")
    ## update
    # tic = time.time()
    loss_old = 10**8
    dists = [[0]*n for n in trainsetnum]
    Ts = [[0]*n for n in trainsetnum]
    for nIter in range(max_nIter):
        # print("iteration:", nIter)
        loss = 0
        R_A = np.zeros((dim,dim))
        R_B = np.zeros((dim,downdim))
        N = np.sum(trainsetnum)
        for c in range(classnum):
            for n in range(trainsetnum[c]):
                seqlen = np.shape(trainset[c][n])[0]
                dist, T = dtw2(np.dot(trainset[c][n],L), virtual_sequence[c])
                Ts[c][n] = T
                dists[c][n] = dist
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


        R_I = R_A + lambda0*N*np.eye(dim)
        #L = inv(R_I) * R_B
        L = np.linalg.solve(R_I,R_B)
    # print(time.time()-tic)
    return L, virtual_sequence,dists,Ts