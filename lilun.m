function [ label_data,unlabel_data,labels,test_data,tast_labels ] = lilun( trace )
%LILUN 理论数据的生成
%%
% trace2=trace;
% trace=trace1;
sizeM = 2000;
label_sample_num = 100;
index_label_data_1 = randperm(sizeM,label_sample_num);
label_data_1=trace(:,index_label_data_1);
index_label_data_2 = 2000+randperm(3000,label_sample_num);
label_data_2=trace(:,index_label_data_2);
label_data=[label_data_1,label_data_2];
% label_data=label_data(81:160,:);
% label_data=tr_dat;%人脸识别数据
% label_data=training_feats(:,101:600);%is a matrix with d*N， d is the dimension， N is the number of samples
% 
% labels1=H_train(:,101:600);% labels of labels_data, 1*N
% 
% labels1(2,:) = labels1(2,:)*2;

labels = zeros(1,label_sample_num*2);
labels(:,1:label_sample_num)=2;
labels(:,label_sample_num+1:label_sample_num*2)=1;
%% 抽取未标签数据
sizeM = size(trace);
unlabel_sample_num = 300;
index_unlabel_data = randperm(sizeM(2),unlabel_sample_num);
unlabel_data=trace(:,index_unlabel_data);% d*M
% unlabel_data=unlabel_data(81:160,:);
% index_unlabel_data=randperm(size(tt_dat,2),unlabel_sample_num);%人脸识别数据
% unlabel_data=tt_dat(:,index_unlabel_data);%人脸识别数据
% for i=1:200
% if index_unlabel_data(i)>2000
% index_u(i)=1;
% else
% index_u(i)=2;
% end
% end
% label_unlabel_data=zeros(1,size(unlabel_data,2));
%%
test_data=trace;
% test_data=trace(81:160,:);%测试数据
% test_data=tt_dat;%人脸识别数据
% tast_labels=H_test(1,:);%测试数据的标签
tast_labels=zeros(1,5000);
 tast_labels(1,1:2000)=2;
 tast_labels(1,2001:5000)=1;
%  tast_labels=ttls;%人脸识别数据
% labels=trls;%人脸识别数据
end

