function  grad  = gradient_descent( i,tmp_A,tmp_D1,tmp_D2,tmp_P,X,maxL,lambda1,lambda2,DD,per_D_size,lambda3,X_size,LABELS,label_data,M,m)
%gradient_descent �����������Ŀ�꺯�����ݶ�
tmp_A_1=tmp_A(:,1:size(label_data,2));%tmp_A_1���б�ǩ���ݵ�ϵ������
% M=mean(tmp_A_1,2);%�б�ǩ���ݵ�ϵ����ƽ��ֵ
% for j=1:maxL
%     c=tmp_A_1(:,LABELS==j);        
%     m{j}=mean(c,2);%�������ϵ����ƽ��ֵ
%     clear c;
% end
jj=LABELS(i);%�����������
tmp_A_2=tmp_A_1;
tmp_A_2(:,i)=[];
LABELS(:,i)=[];
MM=mean(tmp_A_2,2);
cccc=tmp_A_2(:,LABELS==jj);        
mm=mean(cccc,2);%�������ϵ����ƽ��ֵ
% MM=(size(label_data,2)*M-tmp_A_1(:,i))/(size(label_data,2)-1);%��ȥAi��������ƽ��ֵ
% mm{i}=(X_size(jj)*m{jj}-tmp_A_1(:,i))/(X_size(jj)-1);%��ȥAi�ĵ�j������ƽ��ֵ
grad1 = 2*DD*tmp_A_1(:,i)-2*tmp_D2'*X(:,i);
for j=1:maxL
tmp_D0=zeros(size(tmp_D2));
tmp_D0(:,per_D_size*(j-1)+1:per_D_size*j)=tmp_D1(:,:,j);
ccc{j} = tmp_P(i,j)*(2*(tmp_D0'*tmp_D0)*tmp_A_1(:,i)-2*tmp_D0'*X(:,i));
cc{j} = (1-tmp_P(i,j))*(2*(tmp_D0'*tmp_D0)*tmp_A_1(:,i));
end
grad2 = sum(cat(3,ccc{:}),3);
grad3 = sum(cat(3,cc{:}),3);
%grad4=lambda2*sign(tmp_A_1(:,i));
grad5=lambda1*2*(((X_size(jj)-1)/X_size(jj))^2)*(tmp_A_1(:,i)-mm);
s=(size(label_data,2)*(X_size(jj)-1)*mm-(size(label_data,2)-1)*X_size(jj)*MM)/(size(label_data,2)-X_size(jj));
grad66{jj}=2*(((size(label_data,2)-X_size(jj))/(X_size(jj)*size(label_data,2)))^2)*(tmp_A_1(:,i)+s);
% for j=1:jj
%     grad66{j}=2*(1/size(label_data,2))^2*(tmp_A_1(:,i)+(size(label_data,2)-1)*MM-size(label_data,2)*m{j});
% end
% for j=(jj+1):maxL
%     grad66{j}=2*(1/size(label_data,2))^2*(tmp_A_1(:,i)+(size(label_data,2)-1)*MM-size(label_data,2)*m{j});
% end
grad6=lambda1*sum(cat(3,grad66{:}),3);
 grad7=lambda1*lambda3*2*tmp_A(:,i);
grad=grad1+grad2+grad3+grad5+grad6+grad7;
end
