function [ tmp_D1,tmp_D2 ] = Update_dictionary( tmp_A,tmp_D1,tmp_D2,tmp_P,X,maxL,per_D_size )
%Update_dictionary ���¾���
%   �˴���ʾ��ϸ˵��
 tmp_D11=tmp_D1;
for j=1:maxL
    AA=tmp_A;%(:,LABELS==j); %���ڵ�j���ϵ������AA
    XX=X;%(:,LABELS==j);     %���ڵ�j�����������XX
    PP=tmp_P(:,j); %���ڵ�j���������Ӧ��j������ŶȾ���PP
    aa=AA(per_D_size*(j-1)+1:per_D_size*j,:); %���ڵ�j���������Ӧ��j���ϵ������aa
    c=zeros(size(XX));
    for f=1:per_D_size
%            fprintf('i=%f',f);
        for i=1:size(tmp_D2,2)
            if i~=per_D_size*(j-1)+f
            c=c+tmp_D2(:,i)*AA(i,:);
            end
        end
        Y_1=XX-c;
        YB_1=Y_1*(AA(per_D_size*(j-1)+f,:)');
        c=zeros(size(XX));
        c1=zeros(size(XX));
        k=repmat(PP',size(X,1),1);
        b_1=k.*XX;
        k=repmat(PP',size(aa,1),1);
        b_2=k.*aa;
        k=repmat((ones(size(PP))-PP)',size(aa,1),1);
        b_22=k.*aa;
        for i=1:per_D_size
            if i~=f
               c=c+tmp_D1(:,i,j)*b_2(i,:);
               c1=c1+tmp_D1(:,i,j)*b_22(i,:);
            end
        end
        Y_2=b_1-c;
        YB_2=Y_2*(b_2(f,:)');
        Y_3=c1;
        YB_3=Y_3*(b_22(f,:)');
        YB=YB_1+YB_2+YB_3;
        
         if norm(YB,2)<1e-6
            tmp_D11(:,f,j) = zeros(size(YB));
         else
           tmp_D11(:,f,j) = YB./norm(YB,2);   
         end
    end
end
tmp_D1=tmp_D11;
tmp_D2=reshape(tmp_D1,size(X,1),[]);
end

