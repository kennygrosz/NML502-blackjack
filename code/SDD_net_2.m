function SDD_net_2

close all hidden

%set parameters
hid_n = 10; %number of hiden PEs in layer 2
mu = .01; %learning rate
N = 50000; %maximum number of iterations
K=1; %epoch size
tol = .000001;
check_freq = 1000; %check results every check_freq # of iterations

%momentum parameters
alpha = 0;


%------------load data-------------
A = load('SDD_Data.mat');

train_inp = A.Training_Data(:,1:25000);
train_des = A.Training_Desired(:,1:25000);
test_inp = A.Testing_Data(:,1:10000);
test_des = A.Testing_Desired(:,1:10000);
                                

    
%initialize weights variables
W1 = cell(N,1); W2 = cell(N,1);
[inp_n, ~] = size(train_inp);
[out_n, ~] = size(train_des);

W1{1} = -.1 + (.1-(-.1)).*rand(hid_n, inp_n+1); %10x2
W2{1} = -.1 + (.1-(-.1)).*rand(out_n, hid_n +1); %1x11


%train the network
for i = 1:N
    % run through the net for one epoch and update weights
    [W1{i+1},W2{i+1}] = BP_learn(train_inp, train_des, mu,K,W1{i},W2{i});
    
    if alpha > 0 %if momentum exists, recalculate weights
        [W1{i+1},W2{i+1}] = momentum_calc(W1{i},W1{i+1},W2{i},W2{i+1},alpha);
    end
    
    % check errors and tolerance conditions every 100 iterations
    if mod(i+1,check_freq)==0
        % run through current network for training and test data
        train_out = run_net(W1{i+1},W2{i+1},train_inp);
        test1_out = run_net(W1{i+1},W2{i+1},test_inp);
        
        
        % calculate errors
        E_train(:,(i+1)/check_freq) = MSE_calc(train_des,train_out);
        E_test1(:,(i+1)/check_freq) = MSE_calc(test_des,test1_out);
        
%         E_train
%         E_test1
        
        % check tolerance conditions
        if E_train((i+1)/check_freq) < tol
            disp(strcat('Training Terminated, Tolerance of',num2str(tol),' Reached'));
            disp('Number of learn steps ='), disp((i+1)*K)
            break
        end
        
        if i == N
            disp('Training Terminated, Max Learning Steps Reached')
            disp('Number of Learning Steps ='), disp((i+1)*K)
        end
    end
end

        % find minimum error and recall network at this stage
%         min_index = check_freq*find(E_train == min(E_train))-1;
        min_index = i+1;
        
        
        % recover weights
        W1_min = W1{min_index};
        W2_min = W2{min_index};
        
        % run net at this step
        train_out = run_net(W1_min,W2_min,train_inp);
        test1_out = run_net(W1_min,W2_min,test_inp);
        
        % calculate errors
        E_train_min = MSE_calc(train_des,train_out);
        E_test1_min = MSE_calc(test_des,test1_out);
%         
%         disp('Minimum training MSE achieved at learning step'),disp(min_index*K)
%         disp('Minimum MSE for training data is: '),disp(E_train_min(p))
%         disp('Which yields')
%         disp('MSE for s_1(n):'),disp(E_test1_min)
%         disp('MSE for s_2(n):'),disp(E_test2_min)
                
    
    
    
%--------------------PLOTTING ----------------
    
% learning history for training data
figure
semilogy((1:length(E_train))*check_freq*K,E_train),hold on
semilogy((1:length(E_test1))*check_freq*K, E_test1), hold on
title('Learning Histories') 
xlabel('Learn Steps'), ylabel('Unscaled MSE')
legend('Training Data','Testing Data')
hold off


end

function E = MSE_calc(desired, real)
if length(desired)~= length(real), error('real and desired must match in dimension'), end

n = length(desired);

E = sum((desired-real).^2)./n; %calculate mean squared error;

end

function [W1,W2] = BP_learn(training_mat, desired_mat, mu,K,W1,W2)
[~, P] = size(desired_mat);

RN = randperm(P);
INP = training_mat(:,[RN]);
DES = desired_mat(:,[RN]);

cum_dW1 = zeros(size(W1));
cum_dW2 = zeros(size(W2));


for i = 1:K %in each epoch
    hid_net = W1*[1;INP(:,i)];  %10x1
    %hid_net
    hid_out = tanh(hid_net); %10x1
    % hid_out
    y_net = W2*[1;hid_out]; %1x1
    % y_net
    y_out = tanh(y_net); %1x1
    %  y_out
    
    yp_out = 1-(y_out.^2); %1x1
    % algorithm for changing weights
    
    %step 1, most outer layer
    outer_delta= (DES(:,i) - y_out).*yp_out; %1x1
    yp_hid = 1-(hid_out.^2);  %10x1
    
%     size(W2(:,2:end))
%     size(outer_delta)
%     size(yp_hid)
    inner_delta = yp_hid .* (W2(:,2:end)'*outer_delta); %10x1
    
    cum_dW2 = cum_dW2 + mu*outer_delta*[1;hid_out]';
    cum_dW1 = cum_dW1 + mu*inner_delta*[1;INP(:,i)]' ;
    
end

%update weights, batch style!
W2 = W2 + cum_dW2;
W1 = W1 + cum_dW1;

end


function y = run_net(W1,W2, INP)
[~,n]=size(INP); %number of samples

for i = 1 : n %for each sample
    input_vec = INP(:,i);
    hid_net = W1*[1;input_vec];  %10x1
    hid_out = tanh(hid_net); %10x1
    y_net = W2*[1;hid_out]; %1x1
    y_out = tanh(y_net); %1x1
    y(:,i)=y_out; 
end


end


function [y,m,b]=scale(x,fmin,fmax)
%take a vector x and linearly scale it to be between [fmin, fmax]
%return the new vector and (m,b), the constants needed to return it to its
%original value

xmin=min(x); xmax = max(x);
m = (fmax-fmin)/(xmax-xmin); %slope formula
b = fmin-(fmax-fmin)/(xmax-xmin)*xmin; %intercept

y = m*x+b;
%to get back to previous scale, use x=(y-b)/m
end


function [W1_new, W2_new] = momentum_calc(prev_W1,W1,prev_W2,W2,alpha)
    W2_new = W2 + + alpha*(W2 - prev_W2);
    W1_new = W1  + alpha*(W1 - prev_W1);
end
