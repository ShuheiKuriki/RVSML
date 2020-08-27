"""This function belongs to Piotr Dollar's Toolbox
http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html
Please refer to the above web page for definitions and clarifications

Calculates the distance between sets of vectors.

Let X be an m-by-p matrix representing m points in p-dimensional space
and Y be an n-by-p matrix representing another set of points in the same
space. This function computes the m-by-n distance matrix D where D(i,j)
is the distance between X(i,:) and Y(j,:).  This function has been
optimized where possible, with most of the distance computations
requiring few or no loops.

The metric can be one of the following:

'euclidean' / 'sqeuclidean':
  Euclidean / SQUARED Euclidean distance.  Note that 'sqeuclidean'
  is significantly faster.

'chisq'
  The chi-squared distance between two vectors is defined as:
   d(x,y) = sum( (xi-yi)^2 / (xi+yi) ) / 2
  The chi-squared distance is useful when comparing histograms.

'cosine'
  Distance is defined as the cosine of the angle between two vectors.

'emd'
  Earth Mover's Distance (EMD) between positive vectors (histograms).
  Note for 1D, with all histograms having equal weight, there is a simple
  closed form for the calculation of the EMD.  The EMD between histograms
  x and y is given by the sum(abs(cdf(x)-cdf(y))), where cdf is the
  cumulative distribution function (computed simply by cumsum).

'L1'
  The L1 distance between two vectors is defined as:  sum(abs(x-y))


USAGE
 D = pdist2( X, Y, [metric] )

INPUTS
 X        - [m x p] matrix of m p-dimensional vectors
 Y        - [n x p] matrix of n p-dimensional vectors
 metric   - ['sqeuclidean'], 'chisq', 'cosine', 'emd', 'euclidean', 'L1'

OUTPUTS
 D        - [m x n] distance matrix

EXAMPLE
 [X,IDX] = demoGenData(100,0,5,4,10,2,0)
 D = pdist2( X, X, 'sqeuclidean' )
 distMatrixShow( D, IDX )

See also PDIST, DISTMATRIXSHOW

Piotr's Image&Video Toolbox      Version 2.0
Copyright (C) 2007 Piotr Dollar.  [pdollar-at-caltech.edu]
Please email me if you find bugs, or have suggestions or questions!
Licensed under the Lesser GPL [see external/lgpl.txt]
"""
import torch, math

def pdist2(X, Y, options):
    """距離を求める"""
    x = X.size()
    y = Y.size()
    metrics = {'sqeuclidean': distEucSq,
               'euclidean'  : distEucSq,
               'L1'         : distL1,
               'cosine'     : distCosine,
               'emd'        : distEmd,
               'chisq'      : distChiSq,
               }
    # print(metric)
    metric = options.metric
    device = 'cuda' if options.cuda else 'cpu'
    try:
        D = metrics[metric]
    except KeyError as error:
        print("error occurred :", error)
    result = D(X, Y, x, y, device)
    if metric == 'euclidean':
        result = torch.sqrt(result)
    return result

def distL1(X, Y, x, y, device):
    """L1距離"""
    Z = torch.zeros(x, dtype=torch.float64, device=device)
    D = torch.zeros((x[0], y[0]), dtype=torch.float64, device=device)
    # print(vars())
    for i in range(y[0]):
        yi = Y[i, :]
        for j in range(x[0]):
            Z[j] = yi
        # print(Z)
        # print(X)
        D[:, i] = torch.sum(torch.abs(X-Z), 1)
        # print(torch.abs(X-Z))
    return D

def distCosine(X, Y, x, y, device):
    """cosine距離"""
    # print(X.dtype)
    # if( ~isa(X,'double') or ~isa(Y,'double')):
    #   error( 'Inputs must be of type double')
    D = torch.ones((x[0], y[0]), dtype=torch.float64, device=device)
    for i in range(x[0]):
        for j in range(y[0]):
            xy_dot = torch.dot(X[i], Y[j])
            if xy_dot != 0:
                D[i, j] = 1 - xy_dot/(torch.norm(X[i])*torch.norm(Y[j]))
    return D

def distEmd(X, Y, x, y, device):
    """Earth Mover's Distance"""
    Xcdf = torch.cumsum(X, 1)
    Ycdf = torch.cumsum(Y, 1)
    ycdfRep = torch.zeros(x, dtype=torch.float64, device=device)
    D = torch.zeros(x[0], y[0])
    for i in range(y[0]):
        ycdf = Ycdf[i, :]
        # print(vars())
        for j in range(x[0]):
            ycdfRep[j] = ycdf
        D[:, i] = torch.sum(torch.abs(Xcdf - ycdfRep), 1)
    return D

def distChiSq(X, Y, x, y, device):
    """Chi Square"""
# supposedly it's possible to implement this without a loop!
    yiRep = torch.zeros(x)
    D = torch.zeros((x[0], y[0]), dtype=torch.float64, device=device)
    for i in range(y[0]):
        yi = Y[i, :]
        for j in range(x[0]):
            yiRep[j] = yi
        s = yiRep + X
        d = yiRep - X
        D[:, i] = torch.sum(d**2 / (s+2**(-52)), 1)/2
    return D

def distEucSq(X, Y, x, y, device):
    """ユークリッド距離の２乗"""
    # print(__name__)
    #if( ~isa(X,'double') or ~isa(Y,'double'))
     # error( 'Inputs must be of type double') end
    YYRep = torch.zeros((x[0], y[0]), dtype=torch.float64, device=device)
    XXRep = torch.zeros((x[0], y[0]), dtype=torch.float64, device=device)
    #Yt = Y'
    XX = torch.sum(X*X, 1)
    YY = torch.sum(Y*Y, 1).t()
    # print(vars())
    for j in range(y[0]):
        XXRep[:, j] = XX
    for j in range(x[0]):
        YYRep[j] = YY
    D = XXRep + YYRep - 2*torch.mm(X, Y.t())
    return D

# X = torch.array([[1,0,4],[2,3,5],[6,4,2]])
# Y = torch.array([[0,1,2],[1,0,2]])
# print("L1:",pdist2(X,Y,'L1'))
# print("Cosine:",pdist2(X,Y,'cosine'))
# print("Emd:",pdist2(X,Y,'emd'))
# print("ChiSq:",pdist2(X,Y,'chisq'))
# print("EucSq:",pdist2(X,Y,'sqeuclidean'))
# print("Euc:",pdist2(X,Y,'euclidean'))
#

# def distEucSq(X, Y, x, y, device):
#### code from Charles Elkan with variables renamed
# m = X.size()[1] n = Y.size()[1]
# D = sum(X.^2, 2) * ones(1,n) + ones(m,1) * sum(Y.^2, 2)' - 2.*X*Y'


### LOOP METHOD - SLOW
# [m p] = X.size()
# [n p] = Y.size()
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
