function [cards, dealer_sum, opt_strat, opt_pay] = blackjacksim(p,decks)
%play blackjack for p players and one dealer with the specified number of
%decks

%first, generate cards for p players and one dealer. Each player needs two
%cards
C = deckdraw(2*(p+1),[],decks);
dealer = C(1:2);
player = C(3:4);
if p > 1
    others=C(5:end);
end

%-------play as dealer------------------
[d_sum, d_soft] = bj_sum(dealer);

%dealer hits if they have a 16 or less, OR if they have a soft 17
while (d_sum <=16) || (d_sum <=17 && d_soft ==1)
    
    %HIT
    hit = deckdraw(1,C,decks); %draw one card
    dealer = [dealer, hit]; %add hit to dealer's cards
    [d_sum, d_soft] = bj_sum(dealer); %update dealer sum
    C = [C, hit]; %add the hit card to the cards in play
end


%-------play as player------------------

%---stand---%
if bj_sum(player) <= 11
    pay(1) = -inf;
else
    pay(1) = payout(player, dealer);
end


%---hit once---
hit1 = deckdraw(1,C,decks); %draw one card
playerh1 = [player, hit1]; %add hit to player's cards
C = [C, hit1]; %add the hit card to the cards in play

if bj_sum(playerh1) <= 11
    pay(2) = -inf;
else
    pay(2) = payout(playerh1, dealer);
end


%---hit a second time---
hit2 = deckdraw(1,C,decks); %draw one card
playerh2 = [playerh1, hit2]; %add hit to player's cards
C = [C, hit2]; %add the hit card to the cards in play

if bj_sum(playerh2) <= 11
    pay(3) = -inf;
else
    pay(3) = payout(playerh2, dealer);
end 


%---hit a third time---
hit3 = deckdraw(1,C,decks); %draw one card
playerh3 = [playerh2, hit3]; %add hit to player's cards
C = [C, hit3]; %add the hit card to the cards in play

pay(4) = payout(playerh3, dealer);

% %---split---
% if player(1) ~= player(2)
%     pay(5) =  -inf; %in situations where you can't split, make the payout infinitely negative
% else
%     %you can only split if it's the same card twice
% 
%     player1 = [player(1), hit1]; %new hand one
%     player2 = [player(2), hit2]; %new hand two
% 
%     pay(5) = payout(player1, dealer) + payout(player2,dealer); %payout of both hands
% end
% 
% %--- double down ---
%     %take the player who hit once and double his payout
%     pay(6) = 2* payout(playerh1, dealer);
%     %**********************************************************
%     %NOTE: THIS STRATEGY WILL ALWAYS BE PREFERRED TO HITTING ONCE, SO THIS
%     %IS WRONG
%     %**********************************************************

    
% generate outputs to function

dealer_sum = d_sum;
cards = [dealer(1), dealer(2), player(1), player(2),hit1,hit2,hit3];

opt_strat = find(pay==max(pay),1,'first'); %***** NEED TO INCORPORATE HOW TO DEAL WITH TIES. Right now, just takes the first answer (in theory, the least risky??)
opt_pay = max(pay);


end


