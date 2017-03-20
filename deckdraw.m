function C = deckdraw(n,C_old,decks)
%given a number of decks to play with, draw n random cards without
%replacement

%error checking
if n > ((52*decks)-length(C_old)), error('not enough decks to pull that many cards'),end
if decks - floor(decks)~= 0, error('number of decks must be an integer'),end
if n-floor(n) ~= 0, error('number of cards drawn must be an integer'),end

%create a single deck, where 1 = Ace, 2-10 = numbers, 11-13 = {Jack, Q, K}
D = 1:13;
D1 = [D D D D]; %a single deck

D=D1;

%add more decks as necessary
if decks > 1
   for i = 1:decks-1 
    D = [D D1]; %add one deck to the pile
   end
end

%remove cards that are already in play from the deck
for j = 1:length(C_old)
    temp = find(D==C_old(j),1);
    
    if isempty(temp), error('cannot remove card from deck as card is not in deck already'), end
    
    D = D([1:temp-1,temp+1:end]); %remove the old card from the deck 
end


%scramble up the deck
deck_len = length(D);
scramble = randperm(deck_len); 

D = D(scramble); %scrambled deck


C = D(1:n); %draw the first n cards from the scrambled deck
end