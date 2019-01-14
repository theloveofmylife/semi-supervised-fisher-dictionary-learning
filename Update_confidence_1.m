function [ tmp_P ] = Update_confidence_1( tmp_A,tmp_D1,X,maxL,per_D_size,label_data ,threshold,o,tmp_P)
%Update_confidence 置信度矩阵更新
%ini_D1 字典矩阵
%threshold 阈值
e=zeros(size(X,2),maxL);
ex=zeros(size(X,2),maxL);
for i=size(label_data,2)+1:size(X,2)
    for j=1:maxL
    e(i,j)=(norm(X(:,i)-tmp_D1(:,:,j)*tmp_A(per_D_size*(j-1)+1:per_D_size*j,i),2))^2;  %计算重构误差
    ex(i,j)=exp(-e(i,j)/o^2);               %ex为该公式的结果矩阵，o为未知数
    end    
end
for j=1:maxL
for i=size(label_data,2)+1:size(X,2)
        if ex(i,j)/sum(ex(i,1:maxL))>threshold  
           tmp_P(i,j)=ex(i,j)/sum(ex(i,1:maxL));
        else
           tmp_P(i,j)=0;
        end
end
end
end