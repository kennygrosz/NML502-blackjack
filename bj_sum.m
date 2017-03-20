
function [s,soft] = bj_sum(X)
%return the sum of a hand of blackjack

s=0;
soft = 0;

aces = X(X==1); %find aces
n_aces = X(X~=1); %find non-aces

%find sum of non-aces
for i = 1:length(n_aces)
    if n_aces(i)<=10
        s = s + n_aces(i);
    else
        s = s + 10; %add 10 for face cards
    end
end

%now, add aces
for i = 1:length(aces)
    if s <= 10 %if sum is small enough, add 11. This means the sum is "soft"
        s = s+11;
        soft = 1;
    else
        s = s+1;
    end
end

end