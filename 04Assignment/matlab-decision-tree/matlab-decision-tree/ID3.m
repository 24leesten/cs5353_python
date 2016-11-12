function [ t ] = ID3( S,Attributes,Label)
C = unique(Label);
if length(C)==1  %base case for recursion, The tree will not split any further if all labels are unique
t=tree         % a new tree is created and grafted to previous tree on recursion
t=t.addnode(0,C)  %node woth unique label is added
else
    t1=tree    %a new tree is created 
    Hs=0
    for i=1:length(C)
       hs(i)= length(find(C(i)==Label))/length(Label);
       Hs=Hs+(-hs(i)*log2(hs(i))); % finds H(s) of the given dataset ie. entropy of the dataset S
    end
    for i=1:size(Attributes,2)
C1{i} = unique(S(:,i))
Hse=zeros(length((C1{i})),1);
EE=0;
for j=1:length(C1{i})
    clear hse
   h=find(C1{i}(j)==S(:,i));
   ht= length(h);
   for k=1:length(C)
       hse(k)= length(find(C(k)==Label([h])))/ht;
       Hse(j)=Hse(j)+(-hse(k)*log2(hse(k)));
      Hse(isnan(Hse)) = 0 ;
    end
   EE=EE+ht*Hse(j)/length(Label);  %expected entopy of all attributes in the dataset S
    
end
    IG(i)=Hs-EE;    %information gain
    end
    [Amax,A]=max(IG);   %attribute with highest I.G is selected, A is the index of the attribute
[t1,ind]=t1.addnode(0,Attributes{A}) %that attribute is added as a node to the tree
     Attributes(:,A)=[];   % attribute used is removed so it is not selected again along the same node
    for j=1:length(C1{A})   %iterate through all labels of that attribute
        clear Sv
[t1,ind]=t1.addnode(1,C1{A}(j));
    vi=find(C1{A}(j)==S(:,A))
    Sv1=S(vi,:); %create a new dataset for each of the label in attribute selected for the tree to keep splitting along that node
    Sv1(:,A)=[] %remove those feature values for which a node is created
    Sv=Sv1
    Labelv=Label(vi,:);
    if isempty(Sv)==1
        t1=t1.addnode(ind,mode(Label));
    else
     t=ID3( Sv,Attributes,Labelv )% call ID3 to recursively create a new tree for the subset of the data belonging to the attributte selected 
     t1=t1.graft(ind,t) %tree t is grafted with the original tree t1.
    end
    end
    t=t1;
end

end

    




