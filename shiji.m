function [ label_data,unlabel_data,labels,test_data,tast_labels ] = shiji( training_feats,testing_feats )
%SHIJI �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
%%
sizeM1 = 442;
sizeM2 = 984;
sizeM3 = 1234;
% trace=training_feats;
% label_sample_num = 50;
% index_label_data_1 = randperm(sizeM1,label_sample_num);
% label_data_1=trace(:,index_label_data_1);
% index_label_data_2 = sizeM1+randperm(sizeM2,label_sample_num);
% label_data_2=trace(:,index_label_data_2);
% index_label_data_3 = sizeM1+sizeM2+randperm(sizeM3,label_sample_num);
% label_data_3=trace(:,index_label_data_1);
% label_data=[label_data_1,label_data_2,label_data_3];
label_data=training_feats;
% label_data=label_data(61:130,:);
% label_data=training_feats(:,101:600);%is a matrix with d*N�� d is the dimension�� N is the number of samples
% 
% labels1=H_train(:,101:600);% labels of labels_data, 1*N
% 
% labels1(2,:) = labels1(2,:)*2;

labels = zeros(1,sizeM3);
labels(:,1:sizeM1)=1;
labels(:,sizeM1+1:sizeM2)=2;
labels(:,sizeM2+1:sizeM3)=3;
%% ��ȡδ��ǩ����
sizeM = size(testing_feats);
unlabel_sample_num = 150;
index_unlabel_data = randperm(sizeM(2),unlabel_sample_num);
unlabel_data=testing_feats(:,index_unlabel_data);% d*M
% unlabel_data=unlabel_data(61:130,:);
% label_unlabel_data=zeros(1,size(unlabel_data,2));
%%
test_data=testing_feats;%��������

% tast_labels=H_test(1,:);%�������ݵı�ǩ
tast_labels=zeros(1,5000);
%  tast_labels(1,1:2000)=2;
%  tast_labels(1,2001:5000)=1;

end

