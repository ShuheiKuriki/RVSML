import numpy as np
from pdist2 import *

def dtw2(t,r):
    #Dynamic Time Warping Algorithm
    #Dist is unnormalized distance between t and r
    #D is the accumulated distance matrix
    #k is the normalizing factor
    #w is the optimal path
    #t is the vector you are testing against
    #r is the vector you are testing
    rows,N = np.shape(t)
    rows,M = np.shape(r)

    #for n=1:N
    #    for m=1:M
    #        d[n,m)=(t(n)-r(m))^2
    #
    #

    # d = zeros(N,M)
    # for i = 1:N
    #     for j = 1:M
    #          d[i,j) = sum((t(:,i)-r(:,j)).^2)
    #
    #

    d = pdist2(t.T,r.T, 'sqeuclidean')

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

