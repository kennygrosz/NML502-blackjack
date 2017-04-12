function [strat,max_pay] = opt_strat(pay,p_sum)
%find the optimal strategy given a set of rules:
    %if you win, the optimal strategy is the lowest risk strategy that wins
    %if you lose, the optimal strategy is the strategy that maximizes your
    %score
    
max_pay = max(pay);

if max_pay >= 0 %if you win or tie the hand
    strat = find(pay == max_pay,1); %find the lowest risk strategy that wins
elseif max_pay < 0 %if yoiu lose the hand
    no_bust = find(p_sum <= 21);
    strat = find(p_sum==max(p_sum(no_bust)),1);
end
    
    
end