function Acc = VirtualClassifier(classnum,templatenum,testsetdata,testsetdatanum,testsetlabel,options)
    lambda1 = options.lambda1;
    lambda2 = options.lambda2;
    delta = options.delta;
    rightnum = 0;
    downdim = classnum*templatenum;
    virtual_sequence = cell(1,classnum);
    active_dim = 0;
    for c = 1:classnum
        virtual_sequence{c} = zeros(templatenum,downdim);
        for a_d = 1:templatenum
            active_dim = active_dim + 1;
            virtual_sequence{c}(a_d,active_dim) = 1;
        end
    end
    
    for j = 1:testsetdatanum
        dis_to_virtual = zeros(1,classnum);
        for c = 1:classnum
%             disp(virtual_sequence{c}')
%             disp(testsetdata{j}')
            if options.method == "dtw"
                [Dist,T] = dtw2(virtual_sequence{c}',testsetdata{j}');
            elseif options.method == "opw"
                [Dist,T] = OPW_w(virtual_sequence{c},testsetdata{j},[],[],lambda1,lambda2,delta,0);
            elseif options.method == "greedy"
                [Dist,T] = greedy(virtual_sequence{c},testsetdata{j});
            end
            dis_to_virtual(c) = Dist;
            if isnan(Dist)
                disp('NaN distance!')
                break;
            end
        end
        [~,ind]= min(dis_to_virtual);
%         disp(dis_to_virtual)
%         disp(testsetlabel(j))
        if ind==testsetlabel(j)
            rightnum = rightnum+1;
        end
    end
    Acc = rightnum/testsetdatanum;
end