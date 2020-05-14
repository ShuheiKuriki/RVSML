import numpy as np
from .pdist2 import pdist2

def dtw2(t,r):
    #Dynamic Time Warping Algorithm
    #Dist is unnormalized distance between t and r
    #D is the accumulated distance matrix
    #k is the normalizing factor
    #w is the optimal path
    #t is the vector you are testing against
    #r is the vector you are testing
    N = np.shape(t)[0]
    M = np.shape(r)[0]

    #for n=1:N
    #    for m=1:M
    #        d[n,m)=(t(n)-r(m))^2
    #
    #

    # d = zeros(N,M)
    # for i in range(N)
    #     for j in range(M):
    #          d[i,j) = sum((t(:,i)-r(:,j)).^2)
    #
    #

    d = pdist2(t, r, 'sqeuclidean')

    #d=(repmat(t(:),1,M)-repmat(r(:)',N,1)).^2 #this replaces the nested for loops from above Thanks Georg Schmitz

    D = np.zeros(np.shape(d))
    D[0,0]=d[0,0]

    for n in range(1,N):
        D[n,0]=d[n,0]+D[n-1,0]

    for m in range(1,M):
        D[0,m]=d[0,m]+D[0,m-1]

    for n in range(1,N):
        for m in range(1,M):
            # print(D[n-1,m],D[n-1,m-1],D[n,m-1])
            D[n,m]=d[n,m]+min(D[n-1,m],D[n-1,m-1],D[n,m-1])

    Dist=D[-1,-1]
    n=N-1
    m=M-1
    k=1
    w=[]
    # print(type(w))
    w.append([n,m])
    # print(w)
    # print(type(w))
    while (n+m)!=0:
        if n==0:
            m=m-1
        elif m==0:
            n=n-1
        else:
            values = min(D[n-1,m],D[n,m-1],D[n-1,m-1])
            if values==D[n-1,m]:
                n=n-1
            elif values==D[n,m-1]:
                m=m-1
            else:
                n=n-1
                m=m-1
        k=k+1
        # print(n,m)
        w.append([n,m])
    # print(n,m)
    # print(type(w))
    T = np.zeros((N,M))
    for temp_t in range(len(w)):
        # print(T.shape, w[temp_t])
        T[w[temp_t][0],w[temp_t][1]] = 1

    return Dist,T


def OPW_w(X,Y,a,b,options,VERBOSE=0):
    # Compute the Order-Preserving Wasserstein Distance (OPW) for two sequences
    # X and Y

    # -------------
    # INPUT:
    # -------------
    # X: a d * N matrix, representing the input sequence consists of of N
    # d-dimensional vectors, where N is the number of instances (vectors) in X,
    # and d is the dimensionality of instances
    # Y: a d * M matrix, representing the input sequence consists of of N
    # d-dimensional vectors, , where N is the number of instances (vectors) in
    # Y, and d is the dimensionality of instances
    # iterations = total number of iterations
    # lamda1: the weight of the IDM regularization, default value: 50
    # lamda2: the weight of the KL-divergence regularization, default value:
    # 0.1
    # delta: the parameter of the prior Gaussian distribution, default value: 1
    # VERBOSE: whether display the iteration status, default value: 0 (not display)

    # -------------
    # OUTPUT
    # -------------
    # dis: the OPW distance between X and Y
    # T: the learned transport between X and Y, which is a N*M matrix


    # -------------
    # c : barycenter according to weights
    # ADVICE: divide M by median(M) to have a natural scale
    # for lambda

    # -------------
    # Copyright (c) 2017 Bing Su, Gang Hua
    # -------------
    #
    # -------------
    # License
    # The code can be used for research purposes only.

    tolerance=.5e-3
    maxIter= 100
    # The maximum number of iterations with a default small value, the
    # tolerance and VERBOSE may not be used
    # Set it to a large value (e.g, 1000 or 10000) to obtain a more precise
    # transport

    N = np.shape(X)[0]
    M = np.shape(Y)[0]
    dim = np.shape(X)[1]

    if np.shape(Y)[1]!=dim:
        print('The dimensions of instances in the input sequences must be the same!')

    P = np.zeros((N,M))
    mid_para = np.sqrt((1/(N**2) + 1/(M**2)))
    for i in range(N):
        for j in range(M):
            d = np.abs(i/N - j/M)/mid_para
            P[i,j] = np.exp(-d**2/(2*options.delta**2))/(options.delta*np.sqrt(2*np.pi))

    #D = zeros(N,M)
    S = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            #D(i,j) = sum((X(i,:)-Y(j,:)).^2)
            S[i,j] = options.lambda1/((i/N-j/M)**2+1)

    D = pdist2(X,Y, 'sqeuclidean')
    # In cases the instances in sequences are not normalized and/or are very
    # high-dimensional, the matrix D can be normalized or scaled as follows:
    # D = D/max(max(D))  D = D/(10^2)
    #D = D/10

    K = P*np.exp((S - D)/options.lambda2)
    # With some parameters, some entries of K may exceed the machine-precision
    # limit in such cases, you may need to adjust the parameters, and/or
    # normalize the input features in sequences or the matrix D Please see the
    # paper for details.
    # In practical situations it might be a good idea to do the following:
    # K(K<1e-100)=1e-100

    if not a:
        a = np.ones((N,1))/N


    if not b:
        b = np.ones((M,1))/M


    ainvK = K/a

    compt = 0
    u = np.ones((N,1))/N

    # The Sinkhorn's fixed point iteration
    # This part of code is adopted from the code "sinkhornTransport.m" by Marco
    # Cuturi website: http://marcocuturi.net/SI.html
    # Relevant paper:
    # M. Cuturi,
    # Sinkhorn Distances : Lightspeed Computation of Optimal Transport,
    # Advances in Neural Information Processing Systems (NIPS) 26, 2013
    while compt < maxIter:
        u = 1/np.dot(ainvK, b/(np.dot(K.T,u)))
        compt = compt+1
        # check the stopping criterion every 20 fixed point iterations
        if compt%20 == 1 or compt == maxIter:
            # split computations to recover right and left scalings.
            v = b/(np.dot(K.T,u))
            u = 1/np.dot(ainvK,v)

            Criterion = np.sum(np.abs(v*(np.dot(K.T,u))-b))
            if Criterion < tolerance or np.isnan(Criterion): # norm of all || . ||_1 differences between the marginal of the current solution with the actual marginals.
                break

            compt += 1
            if VERBOSE > 0:
                print(['Iteration :',str(compt),' Criterion: ',str(Criterion)])

    U = K * D
    dis = np.sum(u * (np.dot(U, v)))
    T = v.T * u * K

    return dis,T