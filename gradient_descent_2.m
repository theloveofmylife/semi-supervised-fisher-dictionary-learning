function  grad  = gradient_descent_2( i,tmp_A,tmp_D1,tmp_D2,tmp_P,X,maxL,lambda1,lambda2,DD,per_D_size,lambda3,X_size,LABELS,label_data,M,m)
%gradient_descent 计算该样本的目标函数的梯度
%tmp_A_1=tmp_A(:,1:size(label_data,2));%tmp_A_1是有标签数据的系数矩阵
% M=mean(tmp_A_1,2);%有标签数据的系数的平均值
% for j=1:maxL
%     c=tmp_A_1(:,LABELS==j);        
%     m{j}=mean(c,2);%各个类别系数的平均值
%     clear c;
% end
% jj=LABELS(i);%该样本的类别
% MM=(size(label_data,2)*M-tmp_A_1(:,i))/(size(label_data,2)-1);%除去Ai的总样本平均值
% mm{i}=(X_size(jj)*m{jj}-tmp_A_1(:,i))/(X_size(jj)-1);%除去Ai的第j类样本平均值
grad1 = 2*DD*tmp_A(:,i)-2*tmp_D2'*X(:,i);
for j=1:maxL
tmp_D0=zeros(size(tmp_D2));
tmp_D0(:,per_D_size*(j-1)+1:per_D_size*j)=tmp_D1(:,:,j);
ccc{j} = tmp_P(i,j)*(2*(tmp_D0'*tmp_D0)*tmp_A(:,i)-2*tmp_D0'*X(:,i));
cc{j} = (1-tmp_P(i,j))*(2*(tmp_D0'*tmp_D0)*tmp_A(:,i));
end
grad2 = sum(cat(3,ccc{:}),3);
grad3 = sum(cat(3,cc{:}),3);
%grad4=lambda2*sign(tmp_A(:,i));
% grad5=lambda1*2*(((X_size(jj)-1)/X_size(jj))^2)*(tmp_A_1(:,i)-mm{i});
% s=(size(label_data,2)*(X_size(jj)-1)*mm{i}-(size(label_data,2)-1)*X_size(jj)*MM)/(size(label_data,2)-X_size(jj));
% grad66{jj}=2*(((size(label_data,2)-X_size(jj))/(X_size(jj)*size(label_data,2)))^2)*(tmp_A_1(:,i)+s);
% for j=1:jj
%     grad66{j}=2*(1/size(label_data,2))^2*(tmp_A_1(:,i)+(size(label_data,2)-1)*MM-size(label_data,2)*m{j});
% end
% for j=(jj+1):maxL
%     grad66{j}=2*(1/size(label_data,2))^2*(tmp_A_1(:,i)+(size(label_data,2)-1)*MM-size(label_data,2)*m{j});
% end
% grad6=lambda1*sum(cat(3,grad66{:}),3);
%grad7=lambda1*lambda3*2*tmp_A(:,i);
grad=grad1+grad2+grad3;
end
