function SDD_net_3
close all hidden

A = load('SDD_Data2.mat');

Training_Data = A.Training_Data(:,1:25000);
Training_Desired = A.Training_Desired(:,1:25000);
Testing_Data = A.Testing_Data(:,1:25000);
Testing_Desired = A.Testing_Desired(:,1:25000);


% scale data
[Training_Data(2,:),m,b] = scale(Training_Data(2,:),-.9,.9);

Testing_Data(2,:)= Testing_Data(2,:)*m+b;
    



mu = @(ls) (ls < 5000)*0.1 + (ls > 4999 && ls < 10000) *0.05 + (ls > 9999 && ls < 20000)*.025 + (ls > 19999)*0.005;
% alpha = @(ls) (ls < 10000)*.5 + (ls > 9999 && ls < 20000) *.7 + (ls > 19999)*.7;
alpha = @(ls) 0 ;
hid_n = 20; %number of hiden PEs in layer 2
max_iter = 5000; %maximum number of learn steps
K= 10; %epoch size
tol = 1;

[W1,W2,~] = gen_net(Training_Data, Training_Desired,Testing_Data, Testing_Desired,  hid_n, mu, max_iter, tol,K,alpha);

save('SDD_Weights.mat','W1','W2');
% recall_matrix(W1,W2)

split_matrix(W1,W2,m,b)

return

function split_matrix(W1,W2,m,b)
map = [-0.9 linspace(-0.5,0.9,9) 0.9 0.9 0.9];

p_sums = [12, 4:2:20]; %possible player sums that are splittable
soft = [1, zeros(size(4:20))];
dealer = map(1:13);
% split = ones(size(p_sums));
S = zeros(length(p_sums),length(dealer)); %split matrix

for i = 1:length(p_sums)
    for j = 1:length(dealer)
        Inp = [dealer(j); m*p_sums(i)+b ; soft(i); 1 ];
        Out = test_all_net(W1,W2,Inp);
        [~,split]=max(Out);
        if split == 3
            S(i,j)=1;
        end
    end
end
S = [1:13;S];
S= [[0;p_sums'],S];
S

S2 = zeros(length(p_sums),length(dealer)); %split matrix
for i = 1:length(p_sums)
    for j = 1:length(dealer)
        Inp = [dealer(j); m*p_sums(i)+b ; soft(i); 0 ];
        Out = test_all_net(W1,W2,Inp);
        [~,split]=max(Out);
        if split == 3
            S2(i,j)=1;
        end
    end
end
S2

return

function recall_matrix(W1,W2)

poss_scores = [2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21];
soft = [-.9,.9];
map = [-0.9 linspace(-0.5,0.9,9) 0.9 0.9 0.9];
INP = zeros(4,520);
count = 1;

for j = 1:1
    for k = 1:length(map)
        for i = 1:length(poss_scores)
            INP(:,count) = [map(k);poss_scores(i);soft(j); 0];
            count = count + 1;
        end
    end
end

for j = 2:2
    for k = 1:length(map)
        for i = 1:length(poss_scores)
            INP(:,count) = [map(k);poss_scores(i);soft(j); 0];
            count = count + 1;
        end
    end
end

[INP(2,:),~,~] = scale(INP(2,:),-0.9,0.9);

Outold = test_all_net(W1,W2,INP);
Out = Outold(1,:);

for i = 1:length(Out)
    [~,ind] = max(Outold(:,i));
    Out(:,i) = ind-1;
end

Out = reshape(Out,20,26);

Outhard = Out(:,1:13);
Outhard(:,14) = 2:21
Outsoft = Out(:,14:26);
Outsoft(:,14) = 2:21

return


function [W1,W2,E] = gen_net(training_mat, desired_mat,test_mat,...
    desired_testmat,  hid_n, mu, n, tol,K,alpha)
[inp_n, ~] = size(training_mat);
[out_n, P] = size(desired_mat);

%initialize weight matrices
W1 = -.1 + (.1-(-.1)).*rand(hid_n, inp_n+1); %10x2
W2 = -.1 + (.1-(-.1)).*rand(out_n, hid_n +1); %1x11
prev_W1 = W1;
prev_W2 = W2;
E=[];
ind = 1;
indvec = [];
for j = 1:n
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
%         W2_new = W2 + mu*outer_delta*[1;hid_out]'; %1x11
        yp_hid = 1-(hid_out.^2);  %10x1
        inner_delta = yp_hid .* (W2(:,2:end)'*outer_delta); %10x1
%         W1 = W1 + mu*inner_delta*[1;INP(:,i)]';%10x2 

        cum_dW2 = cum_dW2 + mu(j)*outer_delta*[1;hid_out]';
        cum_dW1 = cum_dW1 + mu(j)*inner_delta*[1;INP(:,i)]' ;
        
    end

    %update weights, batch style!
    holder = W2;
    W2 = W2 + cum_dW2 + alpha(j)*(W2 - prev_W2);
    prev_W2 = holder;
    holder = W1;
    W1 = W1 + cum_dW1 + alpha(j)*(W1 - prev_W1);
    prev_W1 = holder;
    
%    test if errors are within tolerance for given W1,W2:
    if mod(j,400) == 0
        train_out=test_all_net(W1,W2,INP); 
        test_out = test_all_net(W1,W2,test_mat);
        E(ind) = 1 - get_accuracy(train_out,DES);
        E_test(ind) = 1 - get_accuracy(test_out,desired_testmat);
        indvec(ind) = j;
        ind = ind + 1;
    end
%     if E(j) <= tol
%         disp(strcat('Training Terminated, Tolerance of', tol,' Reached'));
%         disp('Number of Learn Steps ='), disp(K*j+i)
%         break
%     end
    if j == n
        disp('Training Terminated, Max Learning Steps Reached')
        disp('Number of Learning Steps ='), disp(n*K)
    end
end
figure
plot(indvec,E)
xlabel('Learn Steps')
ylabel('Percent of Set Misclassified')
hold on
plot(indvec,E_test)
legend('training data', 'testing data')
ylim([0 1])
hold off
return

function y = test_all_net(W1,W2, input_vec)

[~,n]=size(input_vec);

for i = 1 : n
    input = input_vec(:,i);
    hid_net = W1*[1;input];  %10x1
    hid_out = tanh(hid_net); %10x1
    y_net = W2*[1;hid_out]; %1x1
    y_out = tanh(y_net); %1x1
    y(:,i)=y_out;
end

return


function correctness = get_accuracy(test_out,Y_test)
[~,compare_output] = max(test_out);
[~,compare_desired] = max(Y_test);

count = 0;
for i = 1:length(compare_desired)
    if compare_output(i) == compare_desired(i)
        count = count + 1;
    end
end


correctness = count/length(compare_desired);
return

function [y,m,b]=scale(x,fmin,fmax)
%take a vector x and linearly scale it to be between [fmin, fmax]
%return the new vector and (m,b), the constants needed to return it to its
%original value

xmin=min(min(x)); xmax = max(max(x));
m = (fmax-fmin)/(xmax-xmin); %slope formula
b = fmin-(fmax-fmin)/(xmax-xmin)*xmin; %intercept

y = m*x+b;
%to get back to previous scale, use x=(y-b)/m
return

