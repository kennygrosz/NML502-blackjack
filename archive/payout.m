function [p, p_sum] = payout(player, dealer)
%given the dealer and player's hands, calculate the payout

p_sum = bj_sum(player);
d_sum = bj_sum(dealer);

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