function [ class,class_rate ] = Classification_function_lilun( test_data,tmp_A,tmp_D1,tmp_D2,maxL,LABELS,per_D_size,sparsitythres,tast_labels,X,test_A )
%对测试数据进行分类，并计算分辨率
%% 修改字典大小
% for j=1:maxL
% for i=1:per_D_size
%     if tmp_D1(:,i,j)==zeros(size(tmp_D1,1),1)
%         tmp_D1(1,i,j)=1;
%     end
% end
% end
% tmp_D1(:,per_D_size,:)=[];
% per_D_size=per_D_size-1;

index_tmp_A=ones(1,size(tmp_D2,2)) ;
D_size=maxL*per_D_size;
tmp_D22=tmp_D2;
j=0;
for i=1:D_size
     if tmp_D22(:,i+j)==zeros(size(tmp_D22,1),1)
         tmp_D22(:,i+j)=[];
         index_tmp_A(1,i)=0;
         j=j-1;
     end
 end
% tmp_D2=reshape(tmp_D1,size(X,1),[]);
% tmp_A(per_D_size+1,:)=[];
% tmp_A(end,:)=[];
%% 分类
for j=1:maxL
   cccc=tmp_A(:,LABELS==j);        
   m{j}=mean(cccc,2);%各个类别系数的平均值
   clear cccc;
end
% a=m{1};
% m{1}=m{2};
% m{2}=a;
%%
% G = tmp_D22'*tmp_D22;
% test_A1 = omp(tmp_D22'* test_data , G ,1);%sparsitythres);%稀疏系数
% test_A1=full(test_A1);
% loc = zeros(1,size(test_data,2));
% test_A=zeros(size(tmp_D2,2),size(test_data,2));
% for i=1:maxL*per_D_size
%      if index_tmp_A(1,i)==1
%          test_A(i,:)=test_A1(sum(index_tmp_A(1,1:i)),:);
%      end
% end
%%
class=zeros(1,size(test_data,2));
e=zeros(size(test_data,2),maxL);
for i=1:size(test_data,2)
%      [loc(i),~] = find(test_A(:,i)~=0);%%%%%%%%%%%%%%%%%%%%%理论数据使用
    for j=1:maxL
        e(i,j)=norm((test_data(:,i)-(tmp_D1(:,:,j)*test_A(per_D_size*(j-1)+1:per_D_size*j,i))),2)^2+3.5*(norm((test_A(:,i)-m{j}),2))^2;      
    end
%     [~,class(i)]=min(e,[],2);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ee=zeros(size(test_data,2),maxL);
% for j=1:maxL
%     ee(1,j)=e(1,j)+1*e(2,j);
% end
% for j=1:maxL
%     ee(size(test_data,2),j)=1*e(size(test_data,2)-1,j)+e(size(test_data,2),j);
% end
% for i=2:size(test_data,2)-1
%     for j=1:maxL
%         ee(i,j)=1*e(i-1,j)+e(i,j)+1*e(i+1,j);
%     end
% end
% [~,class] = min(ee,[],2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ee=zeros(size(test_data,2),maxL);
for j=1:maxL
    ee(1,j)=e(1,j)+1*e(2,j)+1*e(3,j);
end
for j=1:maxL
    ee(2,j)=e(1,j)+1*e(2,j)+1*e(3,j)+1*e(4,j);
end
for j=1:maxL
    ee(size(test_data,2),j)=1*e(size(test_data,2)-2,j)+1*e(size(test_data,2)-1,j)+e(size(test_data,2),j);
end
for j=1:maxL
    ee(size(test_data,2)-1,j)=1*e(size(test_data,2)-3,j)+1*e(size(test_data,2)-2,j)+1*e(size(test_data,2)-1,j)+e(size(test_data,2),j);
end
for i=3:size(test_data,2)-2
    for j=1:maxL
        ee(i,j)=1*e(i-2,j)+1*e(i-1,j)+e(i,j)+1*e(i+1,j)+1*e(i+2,j);
    end
end
[~,class] = min(ee,[],2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [~,class] = min(e,[],2);
%% 实际数据使用
% crossNum = 601;
% inNum = 501;
% faciesData_2 = reshape(class, inNum-2,crossNum-2);
% facies9 = flipud(faciesData_2);
% figure;
% imagesc(facies9);
% colormap(jet(maxL));
% class_rate=0;
%%
figure;%%%%%%%%%%%%%%%%%%%%%理论数据使用
imagesc(class');%%%%%%%%%%%%%%%%%%%%%理论数据使用
% colormap(jet(2));%%%%%%%%%%%%%%%%%%%%%理论数据使用
% [~,class(i)]=min(e,[],2);
 class_rate=size(find(tast_labels==class'),2)/size(tast_labels,2);%%%%%%%%%%%%%%%%%%%%%理论数据使用
% class_rate = 0;