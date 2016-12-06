function [S,Label,Attributes  ] = scan( fid )
A = textscan(fid,'%c %c %c %c %c %c %c %c %c %c %c %c %c %c %c %c %c %c %c %c %c %c %c  ','delimiter',',/r/n')
%A1 = zeros(size(A,2),size(A{1},1));
for j=1:size(A,2)
 for i=1:size(A{1},1)  
   A1(i,j)=  A{1,j}(i);
 end
end
S=A1(:,1:end-1)
Label=A1(:,end)
for i=1:size(S,2)
Attributes{i}= strcat('f', num2str(i));
end
Attributes

end

