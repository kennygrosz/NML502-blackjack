function output_distribution

for i = 1:10000
    [~,~,x(i),~]=blackjacksim(1,4);
end

p(1)=sum(x==1);
p(2)=sum(x==2);
p(3)=sum(x==3);
p(4)=sum(x==4);

bar(p/10000*100); hold on
ylabel('% of Hands'); xlabel('Optimal Strategy');
set(gca,'XTick',1:4,'XTickLabel',{'Stand', 'Hit Once', 'Hit Twice','Hit Thrice'})

end