function  [JJ]  = objective_function( ii,tmp_A,tmp_D1,tmp_D2,tmp_P,X,maxL,LABELS,lambda1,lambda2 ,label_data,per_D_size,M,m,F,mmm) 
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
tmp_A_1=tmp_A(:,1:size(label_data,2));%tmp_A_1是有标签数据的系数矩阵
M=mean(tmp_A_1,2);%有标签数据的系数的平均值
% X_size=zeros(1,maxL);
for j=1:maxL
   c=tmp_A_1(:,LABELS==j);        
   m{j}=mean(c,2);%各个类别系数的平均值
%    X_size(j)=size(c,2);%各类别中样本的个数
%    ccc{j}=X_size(j)*(m{j}-M)*(m{j}-M)'; 
   clear c;
end
% for i=1:size(label_data,2)
%     for j=1:maxL
%         if LABELS(i)==j
%            cc{i}=(tmp_A_1(:,i)-m{j})*(tmp_A_1(:,i)-m{j})';               
%         end
%     end    
% end
% SW=sum(cat(3,cc{:}),3);                    %计算类内散射矩阵
% SB=sum(cat(3,ccc{:}),3);                   %计算类间散射矩阵
% F=lambda1*(trace(SW-SB)+(norm(tmp_A_1,'fro')));                     %鉴别项F
for iii=1:maxL
    mm(iii)=norm((m{iii}-M),'fro');
end
mmm=sum(mm);
F=lambda1*(norm((tmp_A(:,ii)-m{LABELS(1,ii)}),'fro')^2-mmm+(norm(tmp_A(:,ii),'fro')^2));
J1=norm((X(:,ii)-tmp_D2*tmp_A(:,ii)),2)^2;
    for j=1:maxL
        J22(1,j)=norm((X(:,ii)-tmp_D1(:,:,j)*tmp_A(per_D_size*(j-1)+1:per_D_size*j,ii))*tmp_P(ii,j),2)^2;
        J33(1,j)=norm((tmp_D1(:,:,j)*tmp_A(per_D_size*(j-1)+1:per_D_size*j,ii))*(1-tmp_P(ii,j)),2)^2;
    end
    J2=sum(J22(1,:));
    J3=sum(J33(1,:));
J4=lambda2*sum(abs(tmp_A(:,ii)));
JJ=J1+J2+J3+J4+F;                %判别保真度成本+稀疏约束+鉴别项F

end

