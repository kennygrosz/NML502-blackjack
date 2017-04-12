function KG_driver
n=10000;
A = zeros(n,3);
B=zeros(n,1);
for i = 1:n
    [A(i,:),~,B(i,:)] = blackjacksim(1,6);
end
end
    