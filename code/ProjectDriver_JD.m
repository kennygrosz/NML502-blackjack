function ProjectDriver_JD
% gen training data
n = 40000;
D  = zeros(2,n);
I = zeros(3,n);
for i = 1:n
    [D_Out,Inp] = blackjackdriver;
    D(:,i) = D_Out;
    I(:,i) = Inp;
end

Training_Data = I;
Training_Desired = D; 

% gen testing data
n = 10000;
D  = zeros(2,n);
I = zeros(3,n);
for i = 1:n
    [D_Out,Inp] = blackjackdriver;
    D(:,i) = D_Out;
    I(:,i) = Inp;
end

Testing_Data = I;
Testing_Desired = D; 

save('BJ_Data.mat','Training_Data', 'Training_Desired', 'Testing_Data', 'Testing_Desired')

% hit_0 = sum(D(1,:))/length(D(1,:))
% hit_1 = sum(D(2,:))/length(D(1,:))
% hit_2 = sum(D(3,:))/length(D(1,:))
% hit_3 = sum(D(4,:))/length(D(1,:))
% hid_n = 5; %number of hiden PEs in layer 2
% mu = .10; %learning rate
% n = 1; %maximum number of learn steps
% K= 1; %epoch size
% alpha = .50;
% tol = .0000001;
% % 
% [W1p,W2p,~] = gen_net(I, D,I, D,  hid_n, mu, n, tol,K,alpha);
% win1 = 0;
% lose1 = 0;
% tie1 = 0;
% 
% 
% [W1,W2,~] = gen_net(I, D,I, D,  hid_n, mu, 10000, tol,K,alpha);
% win = 0;
% lose = 0;
% tie = 0;
% done_training = 1
% for i = 1:10000
%     [win_tie_lose,win_tie_lose1]  = simround_test(0, 6, W1,W2,W1p,W2p);
%     if win_tie_lose == 2
%         win = win + 1;
%     elseif win_tie_lose == 1
%         tie = tie + 1;
%     else
%         lose = lose + 1;
%     end
%     if win_tie_lose1 == 2
%         win1 = win1 + 1;
%     elseif win_tie_lose1 == 1
%         tie1 = tie1 + 1;
%     else
%         lose1 = lose1 + 1;
%     end
% end
% vec1 = [win1 lose1 tie1]
% bar(vec1)
% xlabel('win                                  lose                                       tie')
% title('untrained')
% figure
% vec2 = [win lose tie]
% bar(vec2)
% xlabel('win                                  lose                                       tie')
% title('trained')
return


function [win_tie_lose,win_tie_lose1] = simround_test(n_previous, n_decks, W1,W2,W1p,W2p)
[dealer, player, ~,Deck] = gen_cards(n_previous, n_decks);
Inp = [player dealer(1)]';
map = [-0.9 linspace(-0.5,0.9,9) 0.9 0.9 0.9];
Inp = map(Inp)';
Out_Choose = eye(4);
deck_loc = 1;
Dealer_val = sumcards(dealer);
deal_bust = 0;
if isblackjack(dealer) && ~isblackjack(player)
    win_tie_lose = 0;
    win_tie_lose1 = 0;
    return
elseif isblackjack(player) && ~isblackjack(dealer)
    win_tie_lose = 2;
    win_tie_lose1 = 2;
    return
elseif isblackjack(player) && isblackjack(dealer)
    win_tie_lose = 1;
    win_tie_lose1 = 1;
    return
end

%simualte player
Decision = test_all_net(W1,W2,Inp);
Decision1 = test_all_net(W1p,W2p,Inp);
for i = 1:4
    if Decision(i) == 1
        break
    end
    player(end+1) = Deck(deck_loc);
    deck_loc = deck_loc + 1;
end

[~,indun] = max(Decision1);
[~,indtr] = max(Decision);
if indun > indtr
    playerun = player;
    for i = 1:indun-indtr
        playerun(end+1) = Deck(deck_loc);
        deck_loc = deck_loc + 1;
    end
elseif indtr == indun
    playerun = player;
else
    playerun = player(1:2+(indun-1));
end
  
Player_val = sumcards(player);
Player_val1 = sumcards(playerun);

%%%simulate dealer 
while Dealer_val < 17 
    dealer(end + 1) = Deck(deck_loc);
    deck_loc = deck_loc + 1;
    Dealer_val = sumcards(dealer);
end

win_tie_lose = checkwin(Player_val,Dealer_val);
win_tie_lose1 = checkwin(Player_val1,Dealer_val);


return

function win_tie_lose = checkwin(Player_val, Dealer_val)

if Player_val > 21
    win_tie_lose = 0;
    return
end

if Dealer_val > 21 
    win_tie_lose = 2;
    return
end



if Dealer_val > Player_val
    win_tie_lose = 0;
elseif Player_val > Dealer_val
    win_tie_lose = 2;
else 
    win_tie_lose = 1;
end
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
E=zeros(n,1);
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

        cum_dW2 = cum_dW2 + mu*outer_delta*[1;hid_out]';
        cum_dW1 = cum_dW1 + mu*inner_delta*[1;INP(:,i)]' ;
        
    end

    %update weights, batch style!
    holder = W2;
    W2 = W2 + cum_dW2 + alpha*(W2 - prev_W2);
    prev_W2 = holder;
    holder = W1;
    W1 = W1 + cum_dW1 + alpha*(W1 - prev_W1);
    prev_W1 = holder;
    
    %test if errors are within tolerance for given W1,W2:
%     train_out=test_all_net(W1,W2,INP); 
%     test_out = test_all_net(W1,W2,test_mat);
%     E(j) = 1 - get_accuracy(train_out,DES,0);
%     E_test(j) = 1 - get_accuracy(test_out,desired_testmat,0);
%     if E(j) <= tol
%         disp(strcat('Training Terminated, Tolerance of', tol,' Reached'));
%         disp('Number of Learn Steps ='), disp(K*j+i)
%         break
%     end
%     if j == n
%         disp('Training Terminated, Max Learning Steps Reached')
%         disp('Number of Learning Steps ='), disp(K*j+i)
%     end
end
% figure
% plot(E(1:j))
% xlabel('Epochs')
% ylabel('Percent of Set Misclassified')
% hold on
% plot(E_test(1:j))
% legend('training data', 'testing data')
% 
% hold off
return

function y = test_all_net(W1,W2, input_vec)

hid_net = W1*[1;input_vec];  
hid_out = tanh(hid_net); 
y_net = W2*[1;hid_out]; 
y_out = tanh(y_net); 
[val,ind] = max(y_out);
y = zeros(size(y_out));
y(ind) = 1;




return

function [D_Out,Inp] = blackjackdriver
n_decks = 6;
n_previous = 0;
[D_Out, Inp] = sim_round(n_previous,n_decks);

return

function [dealer,player,previous,Deck] = gen_cards(n_previous, n_decks)
Deck2 = 1:13;
Deck1 = [Deck2 Deck2 Deck2 Deck2];
Deck = Deck1;
for i = 1:n_decks
    Deck = [Deck Deck1];
end

Deck = Deck(randperm(n_decks*52));
if n_previous > 0
    previous = Deck(1:n_previous);
else 
    previous = 0;
end
dealer = Deck(n_previous + 1:n_previous+2);
player = Deck(n_previous + 3:n_previous+4);
Deck = Deck(n_previous+5:end);
return

function [score, soft] = sumcards(cards)
values = [11 2 3 4 5 6 7 8 9 10 10 10 10];
score = 0;
soft = 0;
aces = 0;
for i = 1:length(cards)
    
    if cards(i) == 1
        aces = aces + 1;
    else
        score = score + values(cards(i)); %% add making it 1 if it would bust u
    end
end

for i = 1:aces
    if score <= 10 %if sum is small enough, add 11. This means the sum is "soft"
        score = score+11;
        soft = 1;
    else
        score = score+1;
    end
end
    
return

function [D_Out, Inp] = sim_round(n_previous,n_decks)
[dealer, player, ~,Deck] = gen_cards(n_previous, n_decks);
Inp = [player dealer(1)]';
map = [-0.9 linspace(-0.5,0.9,9) 0.9 0.9 0.9]; %scaling, sorta

Out_Choose = eye(2); %one in n encoding for outputs
deck_loc = 1;
[Dealer_val, Dealer_soft] = sumcards(dealer);
deal_bust = 0;
% deal_blackjack = isblackjack(dealer);

%%%simulate dealer 
while Dealer_val <= 16 || (Dealer_val <= 17 && Dealer_soft==1)
    dealer(end + 1) = Deck(deck_loc);
    deck_loc = deck_loc + 1;
    [Dealer_val, Dealer_soft] = sumcards(dealer);
end
if Dealer_val > 21 
    deal_bust = 1;
end

%simualte player
[Player_val,Player_soft] = sumcards(player);
Inp = [Player_val, Player_soft, map(dealer(1))]';

for i = 1:2
    D_Out = Out_Choose(:,i);
    if (Player_val > Dealer_val) || (deal_bust && Player_val > 11)
        break
    else 
        player(end+1) = Deck(deck_loc);
        deck_loc = deck_loc + 1;
        [Player_val,~] = sumcards(player);
        if Player_val > 21
            break
        end
    end
end

return

function bool = isblackjack(cards)
if sumcards(cards) == 21
    bool = 1;
else
    bool = 0;
end
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

