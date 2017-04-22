function split_dd_get_data_driver

% gen training data
n = 50000;
data  = zeros(4,n); %3 cards (two player cards, dealer card)
desired = zeros(3,n); %one in n representation of split, DD, stay decision

for i = 1:n
    [D_Out,Inp] = SDD_decision_sim;
    desired(:,i) = D_Out;
    data(:,i) = Inp;
end

Training_Data = data;
Training_Desired = desired; 


% gen testing data
n = 50000;
data  = zeros(4,n); %3 cards (two player cards, dealer card)
desired = zeros(3,n); %one in n representation of split, DD, stay decision

for i = 1:n
    [D_Out,Inp] = SDD_decision_sim;
    desired(:,i) = D_Out;
    data(:,i) = Inp;
end

% sum(desired,2)

Testing_Data = data;
Testing_Desired = desired; 

save('SDD_Data2.mat','Training_Data', 'Training_Desired', 'Testing_Data', 'Testing_Desired')

end

function [D_out, Inp] = SDD_decision_sim
%generate desired output (D_out) and Input (Inp) to a neural net that
%decides whether to stay, split, or double down. do this by simulating all
%3 of these alternatives and choosing that one that gives the most
%profit
n_decks = 6;
n_previous = 0;
[D_out, Inp] = sim_round(n_previous,n_decks);
end

function [D_Out, Inp] = sim_round(n_previous,n_decks)
%simulate a round of doubling down, staying, or splitting.

%first, make dealer cards, player cards, and deck stack.
[dealer, player, ~,Deck] = gen_cards(n_previous, n_decks);

%make input vector (cards)
Inp = [player dealer(1)]';
map = [-0.9 linspace(-0.5,0.9,9) 0.9 0.9 0.9]'; %scaling, sorta
Inp = map(Inp); %scale inputs

deck_loc = 1; %initialize deck index counter

[Dealer_val, Dealer_soft] = sumcards(dealer);
% deal_blackjack = isblackjack(dealer);

%%%simulate dealer hand
while Dealer_val <= 16 || (Dealer_val <= 17 && Dealer_soft==1)
    dealer(end + 1) = Deck(deck_loc);
    deck_loc = deck_loc + 1;
    [Dealer_val, Dealer_soft] = sumcards(dealer);
end

%simualte player hand
    % simulate player staying
    [player,~] = simplayer(player, Deck, deck_loc);
    pay(1) = payout(player,dealer); %simualte playout for hand
    
    
    % simulate doubling down 
        %if you don't get blackjackm simulate payout if n_hits == 1
        if length(player) >=3 %only double down if you have at least 3 cards. otherwise, doubling down would make you bust and thus is useless
            pay(2) = 2*payout(player(1:3), dealer); %compare your first 3 cards to dealer
        else 
            pay(2) = -Inf; %if 2 cards is your best option, doubling down is stupid
        end
    
   % simulate splitting
        %first, determine if splitting is allowed. you can split cards of
        %identical value
        b=cansplit(player(1), player(2));
        if b == 1 %if you can split
            player1 = [player(1), Deck(deck_loc)];
            player2 = [player(2), Deck(deck_loc + 1)];
            deck_loc = deck_loc +2;
            
            %simulate hand 1
            [player1,n_hits] = simplayer(player1, Deck, deck_loc);
            deck_loc = deck_loc + n_hits; %update deck index
            
            %simulate hand 2
            [player2,~] = simplayer(player2, Deck, deck_loc);

            pay(3) = payout(player1,dealer) + payout(player2,dealer); %simualte playout for hand
        else
            pay(3) = -Inf;
        end
        
        
%now, determine which strategy was best
win_strat = find(pay == max(pay), 1, 'first'); %return the lowest risk strategy that wins (stay is lowest risk, then double down, then split)
D_Out = eye(3);
D_Out =D_Out(:,win_strat);

% %play one hand
% disp('if staying or doubling down')
% disp(dealer)
% disp(player)
% 
% if b == 1
% disp('if split')
% disp(dealer)
% disp(player1)
% disp(player2)
% end
% 
% disp('pay and winning strategy')
% disp(pay)
% disp(win_strat)

[S1,S2]=sumcards(player(1:2));
Inp = [map(dealer(1));S1;S2;b];

% Inp
% pause 

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

function [player,n_hits] = simplayer(player, Deck, deck_loc)
%agnostic to the dealer, hit as many times as possible without busting


[Player_val,~] = sumcards(player);
n_hits = -1;

while Player_val <=21
    n_hits = n_hits + 1; %if going through once makes you bust, # hits = 0

    player(end+1) = Deck(deck_loc); %hit   
    deck_loc = deck_loc + 1; %update deck
    [Player_val,~] = sumcards(player); %update player value
end

player = player(1:2+n_hits);

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

function p = payout(player, dealer)
%given the dealer and player's hands, calculate the payout

p_sum = sumcards(player);
d_sum = sumcards(dealer);

p_bj = isblackjack(player);
d_bj = isblackjack(dealer);


%do payouts if wither player has blackjack

if d_bj ==1 %dealer has blackjack
    if p_bj == 1 %player also has blackjack
        p=0; %push
    else
        p=-1; %dealer wins
        
    end
elseif p_bj == 1 %player has blackjack and dealer doesn't
    p = 1.5; %yay!

else %no one has blackjack
    
    if p_sum > 21 %if player busts, he loses money regardless
        p = -1;
    elseif d_sum > 21 %if dealer busts, everyone wins money unless they bust
        p = 1;

    else %if no one busts, compare sums
        
        if p_sum > d_sum
            p = 1;
        elseif p_sum == d_sum
            p = 0;
        else
            p = -1;
        end

    end
end

end

function flag = isblackjack(hand)
%this only applies if a hand has two cards, return 1 if blackjack, 0
%otherwise

flag = 0;

if length(hand) == 2
    if sum(hand==1)==1
        if sum(hand==10)==1 || sum(hand==11)==1 || sum(hand==12)==1 || sum(hand==13)==1
            flag = 1;
        end
    end
end
    
end
