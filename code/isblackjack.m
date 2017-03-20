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