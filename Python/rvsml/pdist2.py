#  This function belongs to Piotr Dollar's Toolbox
# http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html
# Please refer to the above web page for definitions and clarifications
#
# Calculates the distance between sets of vectors.
#
# Let X be an m-by-p matrix representing m points in p-dimensional space
# and Y be an n-by-p matrix representing another set of points in the same
# space. This function computes the m-by-n distance matrix D where D(i,j)
# is the distance between X(i,:) and Y(j,:).  This function has been
# optimized where possible, with most of the distance computations
# requiring few or no loops.
#
# The metric can be one of the following:
#
# 'euclidean' / 'sqeuclidean':
#   Euclidean / SQUARED Euclidean distance.  Note that 'sqeuclidean'
#   is significantly faster.
#
# 'chisq'
#   The chi-squared distance between two vectors is defined as:
#    d(x,y) = sum( (xi-yi)^2 / (xi+yi) ) / 2
#   The chi-squared distance is useful when comparing histograms.
#
# 'cosine'
#   Distance is defined as the cosine of the angle between two vectors.
#
# 'emd'
#   Earth Mover's Distance (EMD) between positive vectors (histograms).
#   Note for 1D, with all histograms having equal weight, there is a simple
#   closed form for the calculation of the EMD.  The EMD between histograms
#   x and y is given by the sum(abs(cdf(x)-cdf(y))), where cdf is the
#   cumulative distribution function (computed simply by cumsum).
#
# 'L1'
#   The L1 distance between two vectors is defined as:  sum(abs(x-y))
#
#
# USAGE
#  D = pdist2( X, Y, [metric] )
#
# INPUTS
#  X        - [m x p] matrix of m p-dimensional vectors
#  Y        - [n x p] matrix of n p-dimensional vectors
#  metric   - ['sqeuclidean'], 'chisq', 'cosine', 'emd', 'euclidean', 'L1'
#
# OUTPUTS
#  D        - [m x n] distance matrix
#
# EXAMPLE
#  [X,IDX] = demoGenData(100,0,5,4,10,2,0)
#  D = pdist2( X, X, 'sqeuclidean' )
#  distMatrixShow( D, IDX )
#
# See also PDIST, DISTMATRIXSHOW

# Piotr's Image&Video Toolbox      Version 2.0
# Copyright (C) 2007 Piotr Dollar.  [pdollar-at-caltech.edu]
# Please email me if you find bugs, or have suggestions or questions!
# Licensed under the Lesser GPL [see external/lgpl.txt]

import numpy as np

def pdist2(X, Y, metric='sqeuclidean'):
    x = np.shape(X)
    y = np.shape(Y)
    metrics = {'sqeuclidean': distEucSq,
               'euclidean'  : distEucSq,
               'L1'         : distL1,
               'cosine'     : distCosine,
               'emd'        : distEmd,
               'chisq'      : distChiSq,
               }
    # print(metric)
    try:
        D = metrics[metric]
        result = D(X,Y,x,y)
        if metric == 'euclidean':
            result = np.sqrt(result)
        return result
    except Exception as error:
        print("error occurred :", error)

def distL1(X,Y,x,y):
    # print(__name__)
    Z = np.zeros(x)
    D = np.zeros((x[0],y[0]))
    # print(vars())
    for i in range(y[0]):
        yi = Y[i,:]
        for j in range(x[0]):
            Z[j] = yi
        # print(Z)
        # print(X)
        D[:,i] = np.sum(np.abs(X-Z),axis=1)
        # print(np.abs(X-Z))
    return D

def distCosine(X,Y,x,y):
    # print(__name__)
    # print(X.dtype)
    # if( ~isa(X,'double') or ~isa(Y,'double')):
    #   error( 'Inputs must be of type double')
    X1 = np.zeros(x)
    for i in range(x[1]):
        X1[:,i] = np.sqrt(np.sum(X*X,axis=1))
    X = X/X1
    Y1 = np.zeros(y)
    for i in range(y[1]):
        Y1[:,i] = np.sqrt(np.sum(Y*Y,axis=1))
    # print(Y1)
    Y = Y/Y1
    D = 1 - np.dot(X,Y.T)
    return D

def distEmd(X,Y,x,y):
    Xcdf = np.cumsum(X,axis=1)
    Ycdf = np.cumsum(Y,axis=1)
    ycdfRep = np.zeros(x)
    D = np.zeros((x[0],y[0]))
    for i in range(y[0]):
      ycdf = Ycdf[i,:]
      # print(vars())
      for j in range(x[0]):
          ycdfRep[j] = ycdf
      D[:,i] = np.sum(np.abs(Xcdf - ycdfRep),axis=1)
    return D

def distChiSq(X,Y,x,y):
# supposedly it's possible to implement this without a loop!
    yiRep = np.zeros(x)
    D = np.zeros((x[0],y[0]))
    for i in range(y[0]):
        yi = Y[i,:]
        for j in range(x[0]):
            yiRep[j] = yi
        s = yiRep + X
        d = yiRep - X
        D[:,i] = np.sum( d**2 / (s+2**(-52)), axis=1)/2
    return D

def distEucSq(X,Y,x,y):
    # print(__name__)
    #if( ~isa(X,'double') or ~isa(Y,'double'))
     # error( 'Inputs must be of type double') end
    YYRep = np.zeros((x[0],y[0]))
    XXRep = np.zeros((x[0],y[0]))
    #Yt = Y'
    XX = np.sum(X*X,axis=1)
    YY = np.sum(Y*Y,axis=1).T
    # print(vars())
    for j in range(y[0]):
        XXRep[:,j] = XX
    for j in range(x[0]):
        YYRep[j] = YY
    D = XXRep + YYRep - 2*np.dot(X,Y.T)

    return D

# X = np.array([[1,0,4],[2,3,5],[6,4,2]])
# Y = np.array([[0,1,2],[1,0,2]])
# print("L1:",pdist2(X,Y,'L1'))
# print("Cosine:",pdist2(X,Y,'cosine'))
# print("Emd:",pdist2(X,Y,'emd'))
# print("ChiSq:",pdist2(X,Y,'chisq'))
# print("EucSq:",pdist2(X,Y,'sqeuclidean'))
# print("Euc:",pdist2(X,Y,'euclidean'))
#

# def distEucSq(X,Y,x,y):
#### code from Charles Elkan with variables renamed
# m = np.shape(X)[1] n = np.shape(Y)[1]
# D = sum(X.^2, 2) * ones(1,n) + ones(m,1) * sum(Y.^2, 2)' - 2.*X*Y'


### LOOP METHOD - SLOW
# [m p] = np.shape(X)
# [n p] = np.shape(Y)
#
# D = zeros(m,n)
# onesM = ones(m,1)
# for i=1:n
#   y = Y(i,:)
#   d = X - y(onesM,:)
#   D(:,i) = sum( d.*d, 2 )
# end


### PARALLEL METHOD THAT IS SUPER SLOW (slower then loop)!
# # From "MATLAB array manipulation tips and tricks" by Peter J. Acklam
# Xb = permute(X, [1 3 2])
# Yb = permute(Y, [3 1 2])
# D = sum( (Xb(:,ones(1,n),:) - Yb(ones(1,m),:,:)).^2, 3)


### USELESS FOR EVEN VERY LARGE ARRAYS X=16000x1000!! and Y=100x1000
# call recursively to save memory
# if( (m+n)*p > 10^5 && (m>1 or n>1))
#   if( m>n )
#     X1 = X(1:floor(end/2),:)
#     X2 = X((floor(end/2)+1):end,:)
#     D1 = distEucSq( X1, Y )
#     D2 = distEucSq( X2, Y )
#     D = cat( 1, D1, D2 )
#   else
#     Y1 = Y(1:floor(end/2),:)
#     Y2 = Y((floor(end/2)+1):end,:)
#     D1 = distEucSq( X, Y1 )
#     D2 = distEucSq( X, Y2 )
#     D = cat( 2, D1, D2 )
#   end
#   return
# end
