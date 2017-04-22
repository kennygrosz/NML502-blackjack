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