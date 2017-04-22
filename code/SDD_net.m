function SDD_net
%train a backprop neural network to figure out whether to stay, double
%down, or split given a hand of blackjack
close all hidden

%network has 3 inputs and 3 outputs

%----------set parameter-----------------------
hid_n = 15; %number of hiden PEs in layer 2
mu = .1; %learning rate
N = 30000; %maximum number of training iterations (learn steps * epoch size
K=5; %epoch size
tol = .0;     
check_freq = 1000; %check results every check_freq # of iterations
alpha = 0; %momentum

%------------load data-------------
A = load('SDD_Data.mat');

train_inp = A.Training_Data(:,1:25000);
train_des = A.Training_Desired(:,1:25000);
test_inp = A.Testing_Data(:,1:10000);
test_des = A.Testing_Desired(:,1:10000);

unique_inp = unique(train_inp','rows');
unique_inp = unique_inp';
[~,wid]=size(unique_inp);
if wid ~= 1000, error('regenrate data-- not all inputs represented'), end

train_des = 2*train_des-1;
test_des = 2*test_des-1;


min(min(train_inp))
min(min(test_inp))
min(min(train_des))
min(min(test_des))
pause
% ------- run network
[W1,W2] = BP_ANN(train_inp, train_des, mu,tol,alpha,N,K,check_freq,hid_n, test_inp,test_des);

% --- run network with best weights
% [OUT1,~,~] = run_net(W1,W2,train_inp,train_des);

% OUT2 = run_net(W1,W2,test_inp,test_des);

%plot original, reconstructed, and residual images
% plot1(OUT1,ocelot,'Ocelot')
% plot1(OUT2,fruitstill,'Fruitstill')

[OUT]=run_net2(W1,W2,unique_inp);
sum(OUT,2)/1000
sum(train_des,2)/25000
sum(test_des,2)/10000

end


function [W1_min,W2_min] = BP_ANN(INP, DES, mu, tol, alpha, N, K, check_freq, hid_n, TEST,TEST_DES)

%initialize weights variables
W1 = cell(N,1); W2 = cell(N,1);
[inp_n, ~] = size(INP);
[out_n, ~] = size(DES);

W1{1} = -.1 + (.1-(-.1)).*rand(hid_n, inp_n+1); %10x2
W2{1} = -.1 + (.1-(-.1)).*rand(out_n, hid_n +1); %1x11


%train the network
for i = 1:N
    % run through the net for one epoch and update weights
    [W1{i+1},W2{i+1}] = BP_learn(INP, DES, mu,K,W1{i},W2{i});
    
    if alpha > 0 %if momentum exists, recalculate weights
        [W1{i+1},W2{i+1}] = momentum_calc(W1{i},W1{i+1},W2{i},W2{i+1},alpha);
    end
    
    % check errors and tolerance conditions every 100 iterations
    if mod(i+1,check_freq)==0
        
        % run through current network for training and test data
        Y_train= run_net2(W1{i+1},W2{i+1},INP);
        T_test= run_net2(W1{i+1},W2{i+1},TEST);
        [E_train((i+1)/check_freq)] = MSE_calc(DES,Y_train)
        [E_test((i+1)/check_freq)]= MSE_calc(TEST_DES,T_test)
        %         % unscale
%         train_out_unscale = (train_out-INP_b)./INP_m;
%         test_out_unscale = (test_out-INP_b)./INP_m;
        
        % calculate errors
%         E_train(:,(i+1)/check_freq) = MSE_calc(INP,DES);
%         E_test(:,(i+1)/check_freq) = MSE_calc(TEST,TEST_DES);
        
        % check tolerance conditions
        if E_train((i+1)/check_freq) < tol
            disp(strcat('Training Terminated, Tolerance of',num2str(tol),' Reached'));
            disp('Number of learn steps ='), disp((i+1)*K)
            break
        end
        
        if (i+1) == N
            disp('Training Terminated, Max Learning Steps Reached')
            disp('Number of Learning Steps ='), disp((i+1)*K)
        end
        
    end
end

% recover network at its minimum error
    %find minimum error and recall network at this stage
    min_index = check_freq*find(E_train == min(E_train));


    %recover weights
    W1_min = W1{min_index};
    W2_min = W2{min_index};
   
% figure
% histogram(W1_min)
% hold on
% histogram(W2_min)

figure
plot((1:length(E_train))*check_freq*K, E_train)
hold on
plot((1:length(E_train))*check_freq*K,E_test)
title('Learning History')
legend('Training Data','Test Data')
xlabel('Learn Steps'); ylabel('Percentage of Samples Misclassified')

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
    
    
    inner_delta = yp_hid .* (W2(:,2:end)'*outer_delta); %10x1
    
    cum_dW2 = cum_dW2 + mu*outer_delta*[1;hid_out]';
    cum_dW1 = cum_dW1 + mu*inner_delta*[1;INP(:,i)]' ;
    
end

%update weights, batch style!
W2 = W2 + cum_dW2;
W1 = W1 + cum_dW1;
end

function [W1_new, W2_new] = momentum_calc(prev_W1,W1,prev_W2,W2,alpha)
    W2_new = W2 + + alpha*(W2 - prev_W2);
    W1_new = W1  + alpha*(W1 - prev_W1);
end


function [err] = run_net(W1,W2, INP, DES)
[~,n]=size(INP);
misclass = 0;

for i = 1 : n
    input_vec = INP(:,i);
    hid_net = W1*[1;input_vec];  %10x1
    hid_out = tanh(hid_net); %10x1
    y_net = W2*[1;hid_out]; %1x1
    y_out = tanh(y_net); %1x1
    
    y_out = round(y_out);
    
%     y_out
%     DES(:,i)
%     y_out == DES(:,i)
    if sum(y_out == DES(:,i)) < 3
        misclass = misclass + 1;
    end
%     misclass
%     pause   
    
end
err = misclass/n*100 %classification percentage
end

function Y = run_net2(W1,W2, INP)
[~,n]=size(INP);
Y = zeros(3,n);

for i = 1 : n
    input_vec = INP(:,i);
    hid_net = W1*[1;input_vec];  %10x1
    hid_out = tanh(hid_net); %10x1
    y_net = W2*[1;hid_out]; %1x1
    y_out = tanh(y_net); %1x1
    
    y_out = round(y_out);
    
    Y(:,i) = y_out;
end


end

function E = MSE_calc(desired, real)
if length(desired)~= length(real), error('real and desired must match in dimension'), end

n = length(desired);

E = sum((desired-real).^2)./n; %calculate mean squared error;

E=mean(E);

end
