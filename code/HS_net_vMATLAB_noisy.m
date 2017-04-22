function HS_net_vMATLAB_noisy
close all hidden

A = load('BJ_data.mat');
% A = load('INPDES.mat');

% Training_Data = A.INP;
% Training_Desired = A.DES;
Training_Data = A.Training_Data;
Training_Desired = A.Training_Desired;
% Testing_Data = A.Testing_Data(:,1:25000);
% Testing_Desired = A.Testing_Desired(:,1:25000);
% Training_Data(:,1:50)


% e1 = (Training_Data(1,:)>=15) & (Training_Data(1,:) <=19); %for error weights
% e2 = ~e1; %lower importance weights
% EW = .8 * e1 + .2 * e2;

size(Training_Data)

% scale data
[Training_Data(2,:),m1,b1] = scale(Training_Data(2,:),-.9,.9);
[Training_Data(1,:),m2,b2] = scale(Training_Data(1,:),-.9,.9);

% [Testing_Data(2,:),~,~] = scale(Testing_Data(2,:),-.9,.9);


%generate net using MATLAB toolbox
net = feedforwardnet([20,10,5]);
net.performFcn='msereg';
net.trainParam.goal=1e-4;
% net.trainFcn = 'trainb';
% net.trainParam.epochs=13500;

net = configure(net,Training_Data,Training_Desired); %configure network
net = init(net); %initialize weights
[net, tr] = train(net, Training_Data, Training_Desired,[],[]);
view(net)


%run network after training
Y_Train = round(sim(net, Training_Data));

Y_Train(:,1:30)
Training_Desired(:,1:30)

correctness = get_accuracy(Y_Train, Training_Desired)

strat_matrix(net,m2,b2,Training_Data,Training_Desired)

% strat_matrix(net,m,b)
save('HS2_net_noisy.mat','net')
return

function strat_matrix(net,m1,b1,INP,DES)
map = [-0.9 linspace(-0.5,0.9,9) 0.9 0.9 0.9];


p_sums1 = 4:21; %possible player sums for unsplittable cards
p_sums = scale(p_sums1,-.9,.9);



dealer = map(1:13); %possible dealer up cards
% split = ones(size(p_sums));
S = zeros(length(p_sums),length(dealer)); %split matrix

%for hard sums, not splittable
for i = 1:length(p_sums)
    for j = 1:length(dealer)
        Inp = [p_sums(i) ; -.9 ; dealer(j)];

%         ix = find(ismember(INP',Inp','rows'),1)
%         ismember(INP',Inp')
%         [~,indx]=ismember(Inp',INP','rows')
%         pause
        Out = round(sim(net, Inp));
        [~,strat]=max(Out);
        
        S(i,j) = strat;
    end
end
S = [1:13;S];
S= [[0;p_sums1'],S];
S

%for soft sums, not splittable
S3 = zeros(length(p_sums),length(dealer)); %split matrix
for i = 1:length(p_sums)
    for j = 1:length(dealer)
        Inp = [p_sums(i); .9 ; dealer(j) ];
        Out = sim(net, Inp);
        [~,strat]=max(Out);
        S3(i,j) = strat;
    end
end
S3 = [1:13;S3];
S3= [[0;p_sums1'],S3];
S3


return

function split_matrix(net,m,b)
map = [-0.9 linspace(-0.5,0.9,9) 0.9 0.9 0.9];

p_sums = [12, 4:2:20]; %possible player sums that are splittable
soft = [1, zeros(size(4:20))];
dealer = map(1:13);
% split = ones(size(p_sums));
S = zeros(length(p_sums),length(dealer)); %split matrix

for i = 1:length(p_sums)
    for j = 1:length(dealer)
        Inp = [dealer(j); m*p_sums(i)+b ; soft(i); 1 ];
        Out = sim(net, Inp);
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
        Out = sim(net, Inp);
        [~,split]=max(Out);
        if split == 3
            S2(i,j)=1;
        end
    end
end
S2

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

