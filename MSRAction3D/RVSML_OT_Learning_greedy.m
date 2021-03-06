function L = RVML_OT_Learning_greedy(trainset,templatenum,lambda,options)

% delta = 1;
% lambda1 = 50;
% lambda2 = 0.1;
% max_nIter = 200;
% err_limit = 10^(-6);

max_nIter = options.max_iters;
err_limit = options.err_limit;

classnum = length(trainset);
downdim = classnum*templatenum;
dim = size(trainset{1}{1},2);

trainsetnum = zeros(1,classnum);
virtual_sequence = cell(1,classnum);
active_dim = 0;
for c = 1:classnum
    trainsetnum(c) = length(trainset{c});
    virtual_sequence{c} = zeros(templatenum,downdim);
    for a_d = 1:templatenum
        active_dim = active_dim + 1;
        virtual_sequence{c}(a_d,active_dim) = 1;
    end
end

%% inilization
R_A = zeros(dim,dim);
R_B = zeros(dim,downdim);
N = sum(trainsetnum);
for c = 1:classnum
    for n = 1:trainsetnum(c)
        seqlen = size(trainset{c}{n},1);
        if options.init == "uniform"
            T_ini = ones(seqlen,templatenum)./(seqlen*templatenum);
        elseif options.init == "normal"
            T_ini = zeros(seqlen,templatenum);
            mid_para = sqrt(1/seqlen^2 + 1/templatenum^2);
            for i = 1:seqlen
                for j = 1:templatenum
                    d = abs(i/seqlen - j/templatenum)/mid_para;
                    T_ini(i,j) = exp(-d^2/(2*options.init_delta^2))/(options.init_delta*sqrt(2*pi));
                end
            end
        elseif options.init == "random"
            T_ini = zeros(seqlen,templatenum);
            for i = 1:seqlen
                for j = 1:templatenum
                    T_ini(i,j) = 1+randn*options.init_delta;
                end
            end
        end
        for i = 1:seqlen
            temp_ra = trainset{c}{n}(i,:)'*trainset{c}{n}(i,:);
            for j = 1:templatenum
                R_A = R_A + T_ini(i,j)*temp_ra;
                R_B = R_B + T_ini(i,j)*trainset{c}{n}(i,:)'*virtual_sequence{c}(j,:);
            end
        end
    end
end
R_I = R_A + lambda*N*eye(dim); 
%L = inv(R_I) * R_B;
L = R_I\R_B;

%% update
loss_old = 10^8;
for nIter = 1:max_nIter
%     disp(nIter)
%     if nIter == 2
%         disp("stop");
%     end
    loss = 0;
    R_A = zeros(dim,dim);
    R_B = zeros(dim,downdim);
    N = sum(trainsetnum);
    for c = 1:classnum
%         if c == 2
%             disp("stop");
%         end
        for n = 1:trainsetnum(c)
            seqlen = size(trainset{c}{n},1);
            [dist, T] = greedy(trainset{c}{n}*L,virtual_sequence{c});
%             disp(d)
%             disp(T)
            loss = loss + dist;
            for i = 1:seqlen
                temp_ra = trainset{c}{n}(i,:)'*trainset{c}{n}(i,:);
                for j = 1:templatenum
                    R_A = R_A + T(i,j)*temp_ra;
                    R_B = R_B + T(i,j)*trainset{c}{n}(i,:)'*virtual_sequence{c}(j,:);
                end
            end
        end
    end
    loss = loss/N + trace(L'*L);
    if mod(nIter,10)==0
        disp(loss);
    end
    if abs(loss - loss_old) < err_limit
        disp(T);
        disp(nIter);
        break;
    else
        loss_old = loss;
    end
    
    R_I = R_A + lambda*N*eye(dim); 
    %L = inv(R_I) * R_B;
    L = R_I\R_B;   
end