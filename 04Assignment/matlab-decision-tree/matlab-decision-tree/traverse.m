function [ pred ] = traverse( t,S,Att )

for i=1:size(S,1)
    index=1;
    SL=S(i,:);
while t.isleaf(index)~=1  %get to the leaf node 
    ind_ch=  t.getchildren(index);
    if length(ind_ch)~=1
  fl= find(ismember( Att, t.get(index) ));
  flag=0;
    for k=ind_ch
        
       if SL(fl)==t.get(k)
           index=k;
           flag=1;
       end
       
    end
    if flag==0
         index= k;
       end
    else
        index=ind_ch; %index stores index of leaf node 
    end
        
end
pred(i,1)=t.get(index);% traverse from root to leaf based on attribute values and predict the output
end

end

