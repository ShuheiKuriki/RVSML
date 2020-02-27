A = [1 2 10; 3 4 20; 9 6 15]
C = bsxfun(@minus, A, [4,3,4,5])
D = bsxfun(@rdivide, C, std(A))