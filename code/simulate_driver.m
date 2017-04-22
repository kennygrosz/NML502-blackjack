function simulate_driver
%walk through neural net's playing of several hands of blackjack
N = 10000; %length of simulation

%set bet
bet = 1;

%get neural net weights
    %net 1
    SDD = load('SDD_Weights.mat');
    W1_1 = SDD.W1;
    W2_1 = SDD.W2;
    net = load('SDD_net.mat');
    net=net.net;

    %net 2
    HS = load('HS2_net_noisy.mat');
%     W1_2 = HS.W1;
%     W2_2= HS.W2;
    HS_net= HS.net;

[A_soft,A_hard,SDD_soft,SDD_hard]  = makeidealdata();

for i = 1:N
%generate hand
[dealer, player,~,Deck] = gen_cards(0, 6); %(n_previous, n_decks)
deck_loc=1;

%-----begin playing hand as neural net----------

%----1. generate inputs to first neural net (split/double/etc) ------
%inputs are: [map(dealer card), scale(player_sum), binary(player soft),
%binary(player split?)]

map = [-0.9 linspace(-0.5,0.9,9) 0.9 0.9 0.9]'; %scaling, for cards
[p_sum,soft]=sumcards(player);
split = cansplit(player(1), player(2));
m = .1059;
b = -1.3235;
Inp = [map(dealer(1)); m*p_sum+b ; soft; split];


%run net 1
% y = run_net(W1_1,W2_1,Inp);
y = sim(net,Inp);
[~,y] = max(y); %generate optimal strategy for S/DD
s(i)=y;

%----now, split into multiple options

if y == 1
    pay(1,i) = playhand(dealer,player,bet,HS_net,Deck,deck_loc);
    
elseif y==2
    bet2 = 2*bet;
    pay(1,i) = playhand(dealer,player,bet2,HS_net,Deck,deck_loc);
elseif y == 3
    bet3 = bet;
    
    player1 = [player(1), Deck(deck_loc)];
    player2 = [player(2), Deck(deck_loc + 1)];
    deck_loc = deck_loc +2;
    
    %hand 1
    [pay1,deck_loc] = playhand(dealer,player1,bet3,HS_net,Deck,deck_loc);

    %hand 2
    [pay2,~] = playhand(dealer,player2,bet3,HS_net,Deck,deck_loc);
    
    pay(1,i) = pay1+pay2; 
end


%------------------------------------------------------------------
%play hand with random decisions
deck_loc =1;
split = cansplit(player(1), player(2));

if split == 0
    y=randi(2,1);
else
    y = randi(3,1);
end

if y == 1
    pay(2,i) = playhand_rand(dealer,player,bet,Deck,deck_loc);
    
elseif y==2
    bet2 = 2*bet;
    pay(2,i) = playhand_rand(dealer,player,bet2,Deck,deck_loc);
elseif y == 3
    bet3 = bet;
    
    player1 = [player(1), Deck(deck_loc)];
    player2 = [player(2), Deck(deck_loc + 1)];
    deck_loc = deck_loc +2;
    
    %hand 1
    [pay1,deck_loc] = playhand_rand(dealer,player1,bet3,Deck,deck_loc);

    %hand 2
    [pay2,~] = playhand_rand(dealer,player2,bet3,Deck,deck_loc);
    
    pay(2,i) = pay1+pay2; 
end


%-------------------------------------------------------------
% play hand with optimal strategy

deck_loc=1;
d_map = [1:10, 10, 10, 10];
dealer_new = d_map(dealer); 
[p_sum,soft]=sumcards(player);


if soft == 1
    y = SDD_soft(p_sum,dealer_new(1));
else
    y = SDD_hard(p_sum,dealer_new(1));
end


if y == 1
    pay(3,i) = playhand_opt(dealer_new,player,bet,Deck,deck_loc,A_hard,A_soft);
    
elseif y==2
    bet2 = 2*bet;
    pay(3,i) = playhand_opt(dealer_new,player,bet2,Deck,deck_loc,A_hard,A_soft);
elseif y == 3
    bet3 = bet;
    
    player1 = [player(1), Deck(deck_loc)];
    player2 = [player(2), Deck(deck_loc + 1)];
    deck_loc = deck_loc +2;
    
    %hand 1 
    [pay1,deck_loc] = playhand_opt(dealer_new,player1,bet3,Deck,deck_loc,A_hard,A_soft);

    %hand 2
    [pay2,~] = playhand_opt(dealer_new,player2,bet3,Deck,deck_loc,A_hard,A_soft);
    
    pay(3,i) = pay1+pay2; 
end


end





figure
plot(cumsum(pay(1,:))); hold on
plot(cumsum(pay(2,:)));
plot(cumsum(pay(3,:)));
ylabel('Cumulative Money Made')
xlabel('Rounds Played')
legend('Neural Net','Random Strategy','Optimal Strategy','Location','SouthWest')
title(strcat('Average Winnings Per Hand on a $1 bet =',num2str(sum(pay,2)/length(pay))))
hold off


end

function [pay,deck_loc] = playhand_opt(dealer,player,bet,Deck,deck_loc,A_hard,A_soft)
%given all of the parts of a blackjack hand, simulate how the hand goes
%based on neural net


%generate inputs to neural net #2
%in net 2: Inp = [scale(Player_val), Player_soft, map(dealer(1))]';

[p_sum,soft]=sumcards(player);


if soft == 1
   hit = A_soft(p_sum,dealer(1));
else
    hit = A_hard(p_sum,dealer(1));
end

%run net 2
% y = run_net(W1,W2,Inp);
% [~,hit]=max(y); %get optimal strategy. hit = 1 if stay, 2 if hit



while hit == 2
    %if hit
    
    %hit card
    player(end+1) = Deck(deck_loc);
    deck_loc = deck_loc+1;
    
    %regenerate inputs
    [p_sum,soft]=sumcards(player);
    
    %run net
    %     y = run_net(W1,W2,Inp);
    %     [~,hit]=max(y); %get optimal strategy. hit = 0 if stay
    if  p_sum <=21
        if soft == 1
            hit = A_soft(p_sum,dealer(1));
        else
            hit = A_hard(p_sum,dealer(1));
        end
    else
        hit = 1;
    end
       
end


%%%simulate dealer hand
[Dealer_val, Dealer_soft] = sumcards(dealer);

while Dealer_val <= 16 || (Dealer_val <= 17 && Dealer_soft==1)
    dealer(end + 1) = Deck(deck_loc);
    deck_loc = deck_loc + 1;
    [Dealer_val, Dealer_soft] = sumcards(dealer);
end

%figure out who wins
pay = payout(player,dealer);

pay = bet*pay;
end

function [pay,deck_loc] = playhand_rand(dealer,player,bet,Deck,deck_loc)
%given all of the parts of a blackjack hand, simulate how the hand goes
%based on neural net


%generate inputs to neural net #2
%in net 2: Inp = [scale(Player_val), Player_soft, map(dealer(1))]';


%run net 2
% y = run_net(W1,W2,Inp);
% [~,hit]=max(y); %get optimal strategy. hit = 1 if stay, 2 if hit
hit = randi(2,1); %pick a random integer between 1 and 2;


while hit == 2
    %if hit
    
    %hit card
    player(end+1) = Deck(deck_loc);
    deck_loc = deck_loc+1;
        
    %run net
%     y = run_net(W1,W2,Inp);
%     [~,hit]=max(y); %get optimal strategy. hit = 0 if stay
      hit = randi(2,1);
    
end


%%%simulate dealer hand
[Dealer_val, Dealer_soft] = sumcards(dealer);

while Dealer_val <= 16 || (Dealer_val <= 17 && Dealer_soft==1)
    dealer(end + 1) = Deck(deck_loc);
    deck_loc = deck_loc + 1;
    [Dealer_val, Dealer_soft] = sumcards(dealer);
end

%figure out who wins
pay = payout(player,dealer);

pay = bet*pay;
end


function [pay,deck_loc] = playhand(dealer,player,bet,net,Deck,deck_loc)
%given all of the parts of a blackjack hand, simulate how the hand goes
%based on neural net


%generate inputs to neural net #2
%in net 2: Inp = [scale(Player_val), Player_soft, map(dealer(1))]';

map = [-0.9 linspace(-0.5,0.9,9) 0.9 0.9 0.9]'; %scaling, for cards
[p_sum,soft]=sumcards(player);
m = .1059;
b = -1.3235;

Inp = [ m*p_sum+b ; map(dealer(1)); soft];

%run net 2
y = sim(net,Inp);
[~,hit]=max(y); %get optimal strategy. hit = 1 if stay, 2 if hit

while hit == 2
    %if hit
    
    %hit card
    player(end+1) = Deck(deck_loc);
    deck_loc = deck_loc+1;
    
    %regenerate inputs
    [p_sum,soft]=sumcards(player);
    Inp = [ m*p_sum+b ; map(dealer(1)); soft];
    
    %run net
    y = sim(net,Inp);
    [~,hit]=max(y); %get optimal strategy. hit = 0 if stay
    
end


%%%simulate dealer hand
[Dealer_val, Dealer_soft] = sumcards(dealer);

while Dealer_val <= 16 || (Dealer_val <= 17 && Dealer_soft==1)
    dealer(end + 1) = Deck(deck_loc);
    deck_loc = deck_loc + 1;
    [Dealer_val, Dealer_soft] = sumcards(dealer);
end

%figure out who wins
pay = payout(player,dealer);

pay = bet*pay;
end

function y = run_net(W1,W2, input_vec)

[~,n]=size(input_vec);

for i = 1 : n
    input = input_vec(:,i);
    hid_net = W1*[1;input];  %10x1
    hid_out = tanh(hid_net); %10x1
    y_net = W2*[1;hid_out]; %1x1
    y_out = tanh(y_net); %1x1
    y(:,i)=y_out;
end

end

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
end

function b = cansplit(c1,c2)
%b is 1 if c1 and c2 have same value
b = 0;
if c1 == c2 %if c1 and c2 are the same, you're good
    b = 1;
elseif c1>=10 && c2 >= 10 %also if both are 10 or a face card
    b = 1;
end

end

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

function [A_soft,A_hard,SDD_soft,SDD_hard]  = makeidealdata()

%hit = 1, stand = 0
A_hard = [...
    0 0 0 0 0 0 0 0 0 0; ...
    0 0 0 0 0 0 0 0 0 0; ...
    0 0 0 0 0 0 0 0 0 0; ...
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
    ] +1;

A_soft = [ ...
    0 0 0 0 0 0 0 0 0 0; ...
    0 0 0 0 0 0 0 0 0 0; ...
    0 0 0 0 0 0 0 0 0 0; ...
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
    ] +1;

SDD_hard = [ ... %1 = stand, 2 = DD, 3=split
    0 0 0 0 0 0 0 0 0 0; ...
    0 0 0 0 0 0 0 0 0 0; ...
    0 0 0 0 0 0 0 0 0 0; ...
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
    0 0 0 0 0 0 0 0 0 0; ...
    0 0 0 0 0 0 0 0 0 0; ...
    0 0 0 0 0 0 0 0 0 0; ...
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