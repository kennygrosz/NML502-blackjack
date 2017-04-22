function SDD_net_4
close all hidden

A = load('SDD_Data2.mat');

Training_Data = A.Training_Data(:,1:25000);
Training_Desired = A.Training_Desired(:,1:25000);
% Testing_Data = A.Testing_Data(:,1:25000);
% Testing_Desired = A.Testing_Desired(:,1:25000);


% scale data
[Training_Data(2,:),m,b] = scale(Training_Data(2,:),-.9,.9);
% [Testing_Data(2,:),~,~] = scale(Testing_Data(2,:),-.9,.9);


%generate net using MATLAB toolbox
net = feedforwardnet([20]);
net = configure(net,Training_Data,Training_Desired); %configure network
net = init(net); %initialize weights
view(net)
[net, tr] = train(net, Training_Data, Training_Desired);


%run network after training
Y_Train = round(sim(net, Training_Data));

correctness = get_accuracy(Y_Train, Training_Desired)

% split_matrix(net,m,b)

strat_matrix(net,m,b)
% save('SDD_net.mat','net')
return

function strat_matrix(net,m,b)
map = [-0.9 linspace(-0.5,0.9,9) 0.9 0.9 0.9];

p_sums = 2:20; %possible player sums for unsplittable cards
p_sums_split = [12, 4:2:20]; %possible player sums that are splittable

dealer = map(1:13); %possible dealer up cards
% split = ones(size(p_sums));
S = zeros(length(p_sums),length(dealer)); %split matrix

%for hard sums, not splittable
for i = 1:length(p_sums)
    for j = 1:length(dealer)
        Inp = [dealer(j); m*p_sums(i)+b ; 0 ; 0 ];
        Out = sim(net, Inp);
        [~,strat]=max(Out);
        S(i,j) = strat;
    end
end
S = [1:13;S];
S= [[0;p_sums'],S];
S

%for soft sums, not splittable
S3 = zeros(length(p_sums),length(dealer)); %split matrix
for i = 1:length(p_sums)
    for j = 1:length(dealer)
        Inp = [dealer(j); m*p_sums(i)+b ; 1 ; 0 ];
        Out = sim(net, Inp);
        [~,strat]=max(Out);
        S3(i,j) = strat;
    end
end
S3 = [1:13;S3];
S3= [[0;p_sums'],S3];
S3



%for splittable sums
soft =  [1, zeros(size(4:20))];

S2 = zeros(length(p_sums_split),length(dealer)); %split matrix
for i = 1:length(p_sums_split)
    for j = 1:length(dealer)
        Inp = [dealer(j); m*p_sums_split(i)+b ; soft(i) ; 1 ];
        Out = sim(net, Inp);
        [~,strat]=max(Out);
        S2(i,j) = strat;
    end
end
S2 = [1:13;S2];
S2= [[0;p_sums_split'],S2];
S2
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

