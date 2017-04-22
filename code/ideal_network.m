function ideal_network
[INP_SDD,OUT_SDD] = all_inputs_SDD(); %scaled data
[INP_HS,OUT_HS] = all_inputs_HS;

%create and train SDD net
net = feedforwardnet([5]);
net = configure(net,INP_SDD,OUT_SDD); %configure network
net = init(net); %initialize weights
[net_SDD, tr] = train(net, INP_SDD, OUT_SDD);

%verify SDD net
%run network after training
INP_SDD(:,1:20)
OUT_SDD(:,1:20)
Y_Train = (sim(net, INP_SDD));
Y_Train(:,1:20)

correctness = get_accuracy(Y_Train, OUT_SDD)
end


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
end

function [INP,OUT] = all_inputs_SDD()
[~,~,SDD_soft,SDD_hard]  = makeidealdata();

O = eye(3); %output options;

dealer = 1:10;
p_sum = 4:21;



count = 1;

%hard sums
for i = 1:length(dealer)
   for j = 1:length(p_sum)
        INP(:,count) = [dealer(i); p_sum(j); -.9; .9];
        if SDD_hard(j,i)==3
            OUT(:,count) = O(:,1); %don't split if you can't
        else
            OUT(:,count) = O(:,SDD_hard(j,i)); %corresponding strategy
        end
        
        count=count+1;
        
        %add a possibility if it is splittable
        if mod(p_sum,2) == 0
            INP(:,count) = [dealer(i); p_sum(j); -.9 ; .9];
            OUT(:,count) = O(:,SDD_hard(j,i)); %corresponding strategy
            count=count+1;
        end   
   end
end


%soft sums
for i = 1:length(dealer)
   for j = 1:length(p_sum)
        INP(:,count) = [dealer(i); p_sum(j); .9; -.9];
        
        if SDD_soft(j,i)==3
            OUT(:,count) = O(:,1); %don't split if you can't
        else
            OUT(:,count) = O(:,SDD_soft(j,i)); %corresponding strategy
        end
        
        count=count+1;
        
        %add a possibility if it is splittable
        if mod(p_sum,2) == 0
            INP(:,count) = [dealer(i); p_sum(j); .9 ; .9];
            OUT(:,count) = O(:,SDD_soft(j,i)); %corresponding strategy
            count=count+1;
        end   
   end
end

%scaling
map = [-0.9 linspace(-0.5,0.9,9) 0.9 0.9 0.9]'; %scaling, for cards

INP(1,:) = map(INP(1,:));
INP(2,:) = scale(INP(2,:),-.9,.9);

end

function [INP,OUT] = all_inputs_HS()
[A_soft,A_hard,~,~]  = makeidealdata();

O = eye(2); %output options;

dealer = 1:10;
p_sum = 4:21;


count = 1;

%hard sums
for i = 1:length(dealer)
   for j = 1:length(p_sum)
        INP(:,count) = [p_sum(j); -.9 ; dealer(i)];
        OUT(:,count) = O(:,A_hard(j,i)+1); %corresponding strategy        
        count=count+1;

   end
end


%soft sums
for i = 1:length(dealer)
   for j = 1:length(p_sum)
        INP(:,count) = [p_sum(j); .9 ; dealer(i)];
        OUT(:,count) = O(:,A_soft(j,i)+1); %corresponding strategy
        count=count+1;

   end
end

%scaling
map = [-0.9 linspace(-0.5,0.9,9) 0.9 0.9 0.9]'; %scaling, for cards

[INP(1,:),~,~] = scale(INP(1,:),-.9,.9); %player sum
INP(3,:) = map(INP(3,:)); %dealer card

end

function [A_soft,A_hard,SDD_soft,SDD_hard]  = makeidealdata()

%hit = 1, stand = 0
A_hard = [...
    1 1 1 1 1 1 1 1 1 1;... %4
    1 1 1 1 1 1 1 1 1 1;... %5 
    1 1 1 1 1 1 1 1 1 1;.... %6
    1 1 1 1 1 1 1 1 1 1;.... %7
    1 1 1 1 1 1 1 1 1 1;... %8
    1 1 1 1 1 1 1 1 1 1;... %9
    1 1 1 1 1 1 1 1 1 1;... %10
    1 1 1 1 1 1 1 1 1 1;... %11
    1 1 0 0 0 1 1 1 1 1; ...%12
    0 0 0 0 0 1 1 1 1 1; ...%13
    0 0 0 0 0 1 1 1 1 1;...%14
    0 0 0 0 0 1 1 1 1 1;...%15
    0 0 0 0 0 1 1 1 1 1;...%16
    0 0 0 0 0 0 0 0 0 0; ...%17
    0 0 0 0 0 0 0 0 0 0; ...%18
    0 0 0 0 0 0 0 0 0 0; ...%19
    0 0 0 0 0 0 0 0 0 0; ...%20
    0 0 0 0 0 0 0 0 0 0; ...%21
    ];

A_soft = [ ...
    1 1 1 1 1 1 1 1 1 1;... %4
    1 1 1 1 1 1 1 1 1 1;... %5 
    1 1 1 1 1 1 1 1 1 1;.... %6
    1 1 1 1 1 1 1 1 1 1;.... %7
    1 1 1 1 1 1 1 1 1 1;... %8
    1 1 1 1 1 1 1 1 1 1;... %9
    1 1 1 1 1 1 1 1 1 1;... %10
    1 1 1 1 1 1 1 1 1 1;... %11
    1 1 1 1 1 1 1 1 1 1;... %12
    1 1 1 1 1 1 1 1 1 1;... %13
    1 1 1 1 1 1 1 1 1 1;... %14
    1 1 1 1 1 1 1 1 1 1;... %15
    1 1 1 1 1 1 1 1 1 1;... %16
    1 1 1 1 1 1 1 1 1 1;... %17
    0 0 0 0 0 0 0 1 1 1; ...%18
    0 0 0 0 0 0 0 0 0 0; ...%19
    0 0 0 0 0 0 0 0 0 0; ...%20
    0 0 0 0 0 0 0 0 0 0; ...%21
    ];

SDD_hard = [ ... %1 = stand, 2 = DD, 3=split
    3 3 3 3 3 3 1 1 1 1;... %4
    1 1 1 1 1 1 1 1 1 1;... %5 
    3 3 3 3 3 3 1 1 1 1;.... %6
    1 1 1 1 1 1 1 1 1 1;.... %7
    1 1 1 3 3 1 1 1 1 1;... %8
    1 2 2 2 2 1 1 1 1 1;... %9
    2 2 2 2 2 2 2 2 1 1;... %10
    2 2 2 2 2 2 2 2 2 2;... %11
    3 3 3 3 3 1 1 1 1 1;... %12
    1 1 1 1 1 1 1 1 1 1;... %13
    3 3 3 3 3 3 1 1 1 1;... %14
    1 1 1 1 1 1 1 1 1 1;... %15
    3 3 3 3 3 3 3 3 3 3;... %16
    1 1 1 1 1 1 1 1 1 1;... %17
    3 3 3 3 3 3 3 3 3 3; ...%18
    1 1 1 1 1 1 1 1 1 1; ...%19
    1 1 1 1 1 1 1 1 1 1; ...%20
    1 1 1 1 1 1 1 1 1 1; ...%21
    ];
    
SDD_soft = [ ... %1 = stand, 2 = DD, 3=split
    1 1 1 1 1 1 1 1 1 1;... %4
    1 1 1 1 1 1 1 1 1 1;... %5 
    1 1 1 1 1 1 1 1 1 1;.... %6
    1 1 1 1 1 1 1 1 1 1;.... %7
    1 1 1 1 1 1 1 1 1 1;... %8
    1 1 1 1 1 1 1 1 1 1;... %9
    1 1 1 1 1 1 1 1 1 1;... %10
    1 1 1 1 1 1 1 1 1 1;... %11
    3 3 3 3 3 3 3 3 3 3;... %12
    1 1 1 2 2 1 1 1 1 1;... %13
    1 1 1 2 2 1 1 1 1 1;... %14
    1 1 2 2 2 1 1 1 1 1;... %15
    1 1 2 2 2 1 1 1 1 1;... %16
    1 2 2 2 2 1 1 1 1 1;... %17
    2 2 2 2 2 1 1 1 1 1; ...%18
    1 1 1 1 2 1 1 1 1 1; ...%19
    1 1 1 1 1 1 1 1 1 1; ...%20
    1 1 1 1 1 1 1 1 1 1; ...%21
    ];




end

function [y,m,b]=scale(x,fmin,fmax)
%take a vector x and linearly scale it to be between [fmin, fmax]
%return the new vector and (m,b), the constants needed to return it to its
%original value

xmin=min(min(x)); xmax = max(max(x));
m = (fmax-fmin)/(xmax-xmin); %slope formula
b = fmin-(fmax-fmin)/(xmax-xmin)*xmin; %intercept

y = m*x+b;
%to get back to previous scale, use x=(y-b)/m
end
