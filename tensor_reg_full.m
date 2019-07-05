function [AUC_list,C,mat_coef_list] = tensor_reg_full(n,n_train,p,q,rank)
% n is the sample_size
% n_train is the training test size

% load true beta
%all_true_beta = csvread('true_beta.csv',1,1);
%true_beta = all_true_beta(6,:);
    % load data
% filename
%formatSpec = '"data_%d_p=%d_q=%d_new_eigen=0.1.csv"';

%size_AUC = (max_p0-min_p0)*(max_q0-min_q0);
AUC_list = [];
mat_coef_list = [];
%MSE_list = [];
    filename = "data_525_p=4_q=4_no_structure_cor.csv";
    disp(filename)
    % load csv
    data = csvread(filename,1,1);
    
    % binary outcome y
    outcome = data(:,1); 
    y_train = outcome(1:n_train);
    y_test = outcome((n_train+1):n);
    %disp(filename);
    %disp(y_test);
    
    % regular covariates
    %X = data(:,2); 
    X_train = zeros(n_train,1);
    X_test = zeros(n-n_train,1);


    % matrix covariate
    M = tensor();
    for i = 1:n
        raw_vector = data(i, 2:end);
        new_matrix = reshape(raw_vector,[p,q]);
        M(:,:,i) = new_matrix;
    end
    M_train = M(:,:,1:n_train);
    M_test = M(:,:,(n_train+1):n);
    
    % direct
    for r = 1:rank
        [beta0,beta,~,~] = kruskal_reg(X_train,M_train,y_train,r,'binomial');
        
        %estimated_beta = double(beta);
        %est_beta = reshape(estimated_beta,[1,p*p]);
        %MSE = mean((est_beta-true_beta).^2);

        pred_mu = X_test*beta0 + double(ttt(tensor(beta), M_test, 1:2));
        pred_prob = 1./(1+exp(-pred_mu));
        [a,b,~,AUC] = perfcurve(y_test,pred_prob,1);
        AUC_list =[AUC_list,AUC];
        %plot(a,b);
        %xlabel('False positive rate');
        %ylabel('True positive rate');
        %title('ROC for Classification by Logistic Regression');
        predicted_y = [];
        for i = 1:length(pred_prob)
            if pred_prob(i) >= 0.5
                predicted_y(i) = 1;
            else
                predicted_y(i) = 0;
            end
        end
        %disp(y_test);
        %disp(predicted_y);
        mat_coef = matthewscorr(y_test,transpose(predicted_y));
        mat_coef_list =[mat_coef_list,mat_coef];
        %MSE_list(r,num_of_data) = MSE;
        C = confusionmat(y_test,transpose(predicted_y));
        
    end  
    end

% find largest AUC test
%AUC_list_mean = mean(AUC_list,2);
%[Max_mean_AUC, max_rank] = max(AUC_list_mean);
%max_AUC_list = AUC_list(max_rank,:);
% find largest MSE 
%avg_MSE = mean(MSE_list,2);
% csvwrite("AUC_output_p0_q0.csv",AUC_list);

