"""アライメント計算"""
from .pdist2 import pdist2
import ot, torch, math

def dtw2(t, r, options):
    """
    Dynamic Time Warping Algorithm
    Dist is unnormalized distance between t and r
    D is the accumulated distance matrix
    k is the normalizing factor
    w is the optimal path
    t is the vector you are testing against
    r is the vector you are testing
    """
    N = t.size()[0]
    M = r.size()[0]
    # print(N,M)
    d = pdist2(t, r, options)
    # print('a')
    #d=(repmat(t(:),1,M)-repmat(r(:)',N,1)).^2 #this replaces the nested for loops from above Thanks Georg Schmitz
    device = 'cuda' if options.cuda else 'cpu'
    D = torch.zeros(d.size(), dtype=torch.float64, device=device)
    D[0, 0] = d[0, 0]

    for n in range(1, N):
        D[n, 0] = d[n, 0]+D[n-1, 0]

    for m in range(1, M):
        D[0, m] = d[0, m]+D[0, m-1]

    for n in range(1, N):
        for m in range(1, M):
            D[n, m] = d[n, m]+min(D[n-1, m], D[n-1, m-1], D[n, m-1])

    Dist = D[-1, -1]
    n = N-1
    m = M-1
    k = 1
    w = []
    # print(type(w))
    w.append([n, m])
    # print(w)
    # print(type(w))
    while (n+m) != 0:
        if n == 0:
            m -= 1
        elif m == 0:
            n -= 1
        else:
            values = min(D[n-1, m], D[n, m-1], D[n-1, m-1])
            if values == D[n-1, m]:
                n -= 1
            elif values == D[n, m-1]:
                m -= 1
            else:
                n -= 1
                m -= 1
        k += 1
        # print(n,m)
        w.append([n, m])
    # print(n,m)
    # print(type(w))
    T = torch.zeros((N, M), dtype=torch.float64).to(device)
    for W in w:
        # print(T.shape, w[temp_t])
        T[W[0], W[1]] = 1

    return Dist, T

def OPW_w(X, Y, options, VERBOSE=0):
    """
    Compute the Order-Preserving Wasserstein Distance (OPW) for two sequences
    X and Y

    -------------
    INPUT:
    -------------
    X: a d * N matrix, representing the input sequence consists of of N
    d-dimensional vectors, where N is the number of instances (vectors) in X,
    and d is the dimensionality of instances
    Y: a d * M matrix, representing the input sequence consists of of N
    d-dimensional vectors, , where N is the number of instances (vectors) in
    Y, and d is the dimensionality of instances
    iterations = total number of iterations
    lamda1: the weight of the IDM regularization, default value: 50
    lamda2: the weight of the KL-divergence regularization, default value:
    0.1
    delta: the parameter of the prior Gaussian distribution, default value: 1
    VERBOSE: whether display the iteration status, default value: 0 (not display)

    -------------
    OUTPUT
    -------------
    dis: the OPW distance between X and Y
    T: the learned transport between X and Y, which is a N*M matrix


    -------------
    c : barycenter according to weights
    ADVICE: divide M by median(M) to have a natural scale
    for lambda

    -------------
    Copyright (c) 2017 Bing Su, Gang Hua
    -------------

    -------------
    License
    The code can be used for research purposes only.
    """
    tolerance = .5e-5
    maxIter = 100
    # The maximum number of iterations with a default small value, the
    # tolerance and VERBOSE may not be used
    # Set it to a large value (e.g, 1000 or 10000) to obtain a more precise
    # transport

    N = X.size()[0]
    M = Y.size()[0]
    dim = X.size()[1]

    if Y.size()[1] != dim:
        print('The dimensions of instances in the input sequences must be the same!')
    device = 'cuda' if options.cuda else 'cpu'
    P = torch.zeros((N, M), dtype=torch.float64).to(device)
    mid_para = math.sqrt(1/(N**2) + 1/(M**2))
    for i in range(N):
        for j in range(M):
            d = abs(i/N - j/M)/mid_para
            P[i, j] = math.exp(-d**2/(2*options.delta**2))/(options.delta*math.sqrt(2*math.pi))

    #D = zeros(N,M)
    S = torch.zeros((N, M), dtype=torch.float64).to(device)
    for i in range(N):
        for j in range(M):
            #D(i,j) = sum((X(i,:)-Y(j,:)).^2)
            S[i, j] = options.lambda1/((i/N-j/M)**2+1)

    D = pdist2(X, Y, options)
    # In cases the instances in sequences are not normalized and/or are very
    # high-dimensional, the matrix D can be normalized or scaled as follows:
    # D = D/max(max(D))  D = D/(10^2)
    #D = D/10

    k = (S - D)/options.lambda2
    k = torch.exp(k)
    K = P*k
    # K[K<1e-100]=1e-100
    # With some parameters, some entries of K may exceed the machine-precision
    # limit in such cases, you may need to adjust the parameters, and/or
    # normalize the input features in sequences or the matrix D Please see the
    # paper for details.
    # In practical situations it might be a good idea to do the following:
    # K(K<1e-100)=1e-100

    a = torch.ones((N, 1), dtype=torch.float64).to(device)/N
    b = torch.ones((M, 1), dtype=torch.float64).to(device)/M

    ainvK = K/a

    compt = 0
    u = torch.ones((N, 1), dtype=torch.float64).to(device)/N

    # The Sinkhorn's fixed point iteration
    # This part of code is adopted from the code "sinkhornTransport.m" by Marco
    # Cuturi website: http://marcocuturi.net/SI.html
    # Relevant paper:
    # M. Cuturi,
    # Sinkhorn Distances : Lightspeed Computation of Optimal Transport,
    # Advances in Neural Information Processing Systems (NIPS) 26, 2013
    while compt < maxIter:
        u = 1/torch.mm(ainvK, b/torch.mm(K.t(), u))
        compt = compt+1
        # check the stopping criterion every 20 fixed point iterations
        if compt%20 == 1 or compt == maxIter:
            # split computations to recover right and left scalings.
            v = b/torch.mm(K.t(), u)
            u = 1/torch.mm(ainvK, v)

            Criterion = torch.sum(torch.abs(v*(torch.mm(K.t(), u))-b))
            if Criterion < tolerance or torch.isnan(Criterion): # norm of all || . ||_1 differences between the marginal of the current solution with the actual marginals.
                break

            compt += 1
            if VERBOSE > 0:
                print('Iteration : {}, Criterion: {}'.format(str(compt), str(Criterion)))

    U = K * D
    dis = torch.sum(u * (torch.mm(U, v)))
    T = v.t() * u * K

    return dis, T

def greedy(X, Y, options):
    """貪欲法"""
    N = X.size()[0]
    M = Y.size()[0]
    device = 'cuda' if options.cuda else 'cpu'
    T = torch.zeros((N, M), dtype=torch.float64).to(device)
    total_dist = 0
    for i in range(N):
        min_dist = float('inf')
        for j in range(M):
            xy_dot = torch.mm(X[i], Y[j])
            if xy_dot == 0:
                d = 1
            else:
                d = 1 - xy_dot/(torch.norm(X[i])*torch.norm(Y[j]))
            if min_dist > d:
                min_ind = j
                min_dist = d
        T[i][min_ind] = 1
        total_dist += min_dist
    return total_dist, T

def OT(X, Y, options):
    """Optimal Transport"""
    D = pdist2(X, Y, options)
    device = 'cuda' if options.cuda else 'cpu'
    N, M = X.size()[0], Y.size()[0]
    a, b = torch.ones(N, dtype=torch.float64).to(device)/N, torch.ones(M, dtype=torch.float64).to(device)/M

    dist = ot.emd2(a, b, D)
    T = ot.emd(a, b, D)
    return dist, T

def sinkhorn(X, Y, options):
    """Sinkhorn Distance Regularized OT"""
    device = 'cuda' if options.cuda else 'cpu'
    D = pdist2(X, Y, options)
    N, M = X.size()[0], Y.size()[0]
    a, b = torch.ones(N, dtype=torch.float64).to(device)/N, torch.ones(M, dtype=torch.float64).to(device)/M

    dist = ot.sinkhorn2(a, b, D, options.regularize)[0]
    T = ot.sinkhorn(a, b, D, 0.1)
    return dist, T

def get_alignment(X, Y, options):
    """総まとめ"""
    if options.method == 'dtw':
        Dist, T = dtw2(X, Y, options)
    elif options.method == 'opw':
        Dist, T = OPW_w(X, Y, options, 0)
    elif options.method == 'greedy':
        Dist, T = greedy(X, Y, options)
    elif options.method == 'OT':
        Dist, T = OT(X, Y, options)
    elif options.method == 'sinkhorn':
        Dist, T = sinkhorn(X, Y, options)
    return Dist, T
    