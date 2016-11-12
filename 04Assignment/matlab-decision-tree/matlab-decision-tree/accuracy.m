function [ acc] = accuracy( pred,Label )
count=0;
for i=1:size(pred,1)
    if pred(i)==Label(i)
        count=count+1;
    end
end
acc=count/length(pred)*100;


end

