
charnum = 20;
classnum = charnum;
dim = 60;
CVAL = 1;

 % add path
addpath('/usr/local/Cellar/vlfeat-0.9.21/toolbox');
vl_setup();
% addpath('libsvm-3.20/matlab');

lambda = 0.0001;
delta = 3;
init_delta = 1;
lambda1 = 0.001;
lambda2 = 0.01;
options.max_iters = 100;
options.err_limit = 10^(-3);
options.lambda1 = lambda1;
options.lambda2 = lambda2;
options.delta = delta;
options.init_delta = init_delta;
options.init = "normal";
options.method = "opw";

load MSR_Python_ori.mat;
trainset_m = trainset;
testsetdata_m = testsetdata;
testsetlabel = testsetdatalabel;

for temp = 1:5
    templatenum = temp*2;
    tic;
    if options.method=="opw"
        L = RVSML_OT_Learning(trainset,templatenum,lambda,options);
    elseif options.method=="dtw"
        L = RVSML_OT_Learning_dtw(trainset,templatenum,lambda,options);
    elseif options.method=="greedy"
        L = RVSML_OT_Learning_greedy(trainset,templatenum,lambda,options);
    end
    RVSML_time = toc;
    % classification with the learned metric
    traindownset = cell(1,classnum);
    testdownsetdata = cell(1,testsetdatanum);
    for j = 1:classnum
        traindownset{j} = cell(trainsetnum(j),1);
        for m = 1:trainsetnum(j)
            traindownset{j}{m} = trainset{j}{m} * L;
        end
    end
    for j = 1:testsetdatanum
        testdownsetdata{j} = testsetdata{j} * L;
    end
    RVSML_acc_virtual = VirtualClassifier(classnum,templatenum,testdownsetdata,testsetdatanum,testsetlabel,options);
    
    if options.method=="opw"
        [RVSML_map,RVSML_acc,knn_average_time] = NNClassifier(classnum,traindownset,trainsetnum,testdownsetdata,testsetdatanum,testsetlabel,options);
    elseif options.method=="dtw"
        [RVSML_map,RVSML_acc,knn_average_time] = NNClassifier_dtw(classnum,traindownset,trainsetnum,testdownsetdata,testsetdatanum,testsetlabel,options);
    elseif options.method=="greedy"
        [RVSML_map,RVSML_acc,knn_average_time] = NNClassifier_greedy(classnum,traindownset,trainsetnum,testdownsetdata,testsetdatanum,testsetlabel,options);
    end
    RVSML_acc_nn = RVSML_acc(1);
    
    fprintf('method is %s, templatnum is %d \n',options.method,templatenum);
    fprintf('Training time of RVSML is %.4f \n',RVSML_time);
%  fprintf('Classification using 1 nearest neighbor classifier with OPW distance:\n');
%     fprintf('MAP is %.4f \n',RVSML_map);
    fprintf('Accuracy by nn is %.4f \n',RVSML_acc_nn);
    fprintf('Accuracy by virtual is %.4f \n',RVSML_acc_virtual);
%  fprintf('opw_knn_average_time is %.4f \n',knn_average_time);
end