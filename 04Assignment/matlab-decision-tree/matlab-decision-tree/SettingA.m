clear all
close all
clc
f = fullfile('ML- project 1\datasets\SettingA')
f1=fullfile('ML- project 1\tinevez-matlab-tree-3d13d15\@tree') %add the matlab tree class to your directory
addpath(f,f1)
old=cd('ML- project 1\datasets\SettingA')

fileID = fopen('training.txt','r');% train data
[S,Label,Attributes] = scan( fileID );%The data has to be in this format, This scan function will not work for new data which is not in the same format. Write your own scan function to bring it this format.


fileID2 = fopen('test.txt','r');
[S_test,Label_test,Attributes_test] = scan( fileID2 );%test data
cd(old)
[ t ] = ID3( S,Attributes,Label );% builds the tress using train data
[ pred ] = traverse( t,S_test,Attributes_test );%test examples and attributes are used to traverse the tree to get labels using the tree t

figure('Position', [100 100 300 300])
t.plot;
[ acc] = accuracy( pred,Label_test )% finds the accuracy between predicted label and label given
 [ t ] = ID3_prune( S,Attributes,Label,2 ); %2 is the depth of the tree after pruning, can be changed to any arbitrary depth
 figure('Position', [100 100 300 300])
 t.plot;
[ pred ] = traverse( t,S_test,Attributes_test );%test_data 
[ acc1] = accuracy( pred,Label_test )
