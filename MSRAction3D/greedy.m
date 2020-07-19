function [total_dist,T] = greedy(X,Y)
    N = size(X,1);
    M = size(Y,1);
    T = zeros(N,M);
    d = pdist2(X,Y, 'sqeuclidean');
%     disp(d)
    [dist,I] = min(d,[],2,'linear');
%         disp(dist)
%         disp(ind)
    T(I) = 1;
    total_dist = sum(dist);
%         disp(total_dist)
end