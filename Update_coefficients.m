function [ tmp_A,JJJJJ] = Update_coefficients( tmp_A,tmp_D1,tmp_D2,tmp_P,X,maxL,LABELS,lambda1,lambda2,lambda3,label_data,per_D_size ) 
%Update_coefficients 系数矩阵更新
%   此处显示详细说明
if size(tmp_D2,1)>size(tmp_D2,2)
   c = 0.02*eigs(tmp_D2'*tmp_D2,1);
else
   c = 0.02*eigs(tmp_D2*tmp_D2',1);
end       %梯度的参数
% c=0.07;
cw=c;
tau=0.005;%软阈值的参数
DD=tmp_D2'*tmp_D2;%tmp_D2转置*tmp_D2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tmp_A_1=tmp_A(:,1:size(label_data,2));%tmp_A_1是有标签数据的系数矩阵
tmp_A_2=tmp_A(:,size(label_data,2)+1:size(tmp_A,2));%tmp_A_2是无标签数据的系数矩阵
M=mean(tmp_A_1,2);%有标签数据的系数的平均值
X_size=zeros(1,maxL);
for j=1:maxL
   cccc=tmp_A_1(:,LABELS==j);        
   m{j}=mean(cccc,2);%各个类别系数的平均值
   X_size(j)=size(cccc,2);%各类别中样本的个数
   ccc{j}=X_size(j)*(m{j}-M)*(m{j}-M)'; 
   clear cccc;
end
for i=1:size(label_data,2)
    for j=1:maxL
        if LABELS(i)==j
           cc{i}=(tmp_A_1(:,i)-m{j})*(tmp_A_1(:,i)-m{j})';               
        end
    end    
end

SW=sum(cat(3,cc{:}),3);                    %计算类内散射矩阵
SB=sum(cat(3,ccc{:}),3);                   %计算类间散射矩阵
for iii=1:maxL
    mm(iii)=norm((m{iii}-M),'fro')^2;
end
mmm=sum(mm);
FF=zeros(1,size(tmp_A_1,2));
for i=1:size(tmp_A_1,2)
FF(i)=lambda1*(norm((tmp_A(:,i)-m{LABELS(1,i)}),'fro')^2-mmm+(norm(tmp_A(:,i),'fro')^2));
end
F=lambda1*(trace(SW-SB)+(norm(tmp_A_1,'fro'))^2);                    %鉴别项F
for i=1:size(X,2)
    J1(i)=norm((X(:,i)-tmp_D2*tmp_A(:,i)),2)^2;
    for j=1:maxL
        J22(i,j)=norm((X(:,i)-tmp_D1(:,:,j)*tmp_A(per_D_size*(j-1)+1:per_D_size*j,i))*tmp_P(i,j),2)^2;
        J33(i,j)=norm((tmp_D1(:,:,j)*tmp_A(per_D_size*(j-1)+1:per_D_size*j,i))*(1-tmp_P(i,j)),2)^2;
    end
    J2(i)=sum(J22(1,:));
    J3(i)=sum(J33(1,:));
J4(i)=lambda2*sum(abs(tmp_A(:,i)));
J(i)=J1(i)+J2(i)+J3(i)+J4(i);               %判别保真度成本+稀疏约束+鉴别项F
end
JJJJJ=sum(J)+F;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%TWIST parameter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%temp_AA=tmp_A;
tmp_AAA=tmp_A;
for_ever           =         1;
IST_iters          =         0;
TwIST_iters        =         0;
sparse             =         1;
verbose            =         1;
enforceMonotone    =         1;
lam1               =         1e-4;   %default minimal eigenvalues
lamN               =         1;      %default maximal eigenvalues
rho0               =         (1-lam1/lamN)/(1+lam1/lamN); 
alpha              =         2/(1+sqrt(1-rho0^2));        %default,user can set
beta               =         alpha*2/(lam1+lamN);         %default,user can set
nIter              =         300;                          %迭代次数                        
sigma              =         c;
citeT    =     1e-6;  % stop criterion停止准则
cT       =     1e+10; % stop criterion
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:size(label_data,2)
    c=cw;
    sigma              =         c;
    tmp_A=tmp_AAA;
    fprintf('i=%f',i);
    Xt_now = tmp_A(:,i);
    xm2   = tmp_A(:,i);%A(:,trls==index);
    xm1   = tmp_A(:,i);%A(:,trls==index); % now
    Xi    =  xm1;
%     [gap] = J(i);
    [gap] = objective_function( i,tmp_A,tmp_D1,tmp_D2,tmp_P,X,maxL,LABELS,lambda1,lambda2 ,label_data,per_D_size,M,m,F,mmm);
    prev_f = gap;
    ert(1) = gap;
for n_it = 2 : nIter;
    tmp_A(:,i) = Xi;
    while for_ever
        % IPM estimate         
        grad=gradient_descent( i,tmp_A,tmp_D1,tmp_D2,tmp_P,X,maxL,lambda1,lambda2,DD,per_D_size,lambda3,X_size,LABELS,label_data,M,m);
        v = xm1-grad./(2*sigma);
        tem = soft(v,tau/sigma);
        x_temp   =   tem;
        
        if (IST_iters >= 2) || ( TwIST_iters ~= 0)
            % set to zero the past when the present is zero
            % suitable for sparse inducing priors
            if sparse
                mask    =   (x_temp ~= 0);
                xm1     =   xm1.* mask;
                xm2     =   xm2.* mask;
            end
            % two-step iteration
            xm2    =   (alpha-beta)*xm1 + (1-alpha)*xm2 + beta*x_temp;
            % compute residual
            tmp_A(:,i) = xm2;%%%%%%%%%%%%%%%%%%%%%%%%%
            [gap] = objective_function( i,tmp_A,tmp_D1,tmp_D2,tmp_P,X,maxL,LABELS,lambda1,lambda2 ,label_data,per_D_size,M,m,F,mmm);
            f   =   gap;          
            if (f > prev_f) && (enforceMonotone)
                TwIST_iters   =  0;  % do a IST iteration if monotonocity fails
            else
                TwIST_iters =   TwIST_iters+1; % TwIST iterations
                IST_iters   =    0;
                x_temp      =   xm2;
                if mod(TwIST_iters,10000) ==0
                   c = 0.9*c; 
                   sigma = c;
                end
                break;  % break loop while
            end
        else
        tmp_A(:,i) = x_temp;%%%%%%%%%%%%%%%%%%%%%%%%
        [gap] = objective_function( i,tmp_A,tmp_D1,tmp_D2,tmp_P,X,maxL,LABELS,lambda1,lambda2 ,label_data,per_D_size,M,m,F,mmm);
    
        f   =   gap;
         
         if f > prev_f
                % if monotonicity  fails here  is  because
                % max eig (A'A) > 1. Thus, we increase our guess
                % of max_svs
                c         =    2*c;  
                sigma     =    c;
                if verbose
%                     fprintf('Incrementing c=%2.2e\n',c);
                end
                if  c > cT
                    break;  % break loop while    
                end
                IST_iters = 0;
                TwIST_iters = 0;
           else
                TwIST_iters = TwIST_iters + 1;
                break;  % break loop while
          end
        end
        
    end

    citerion      =   abs(f-prev_f)/abs(prev_f);
    if citerion < citeT || c > cT
%        fprintf('Stop!\n c=%2.2e\n citerion=%2.2e\n',c,citerion);
       break;
    end
    
    xm2           =   xm1;
    xm1           =   x_temp;
    Xt_now        =   x_temp;
    Xi            =   Xt_now; 
    prev_f        =   f;
    ert(n_it)     =   f;
%     fprintf('Iteration:%f  Total gap:%f\n',n_it,ert(n_it-1));
 end  


tmp_AA(:,i)  =      Xt_now;
opts.ert   =       ert;%%%%%暂未使用
end
for i=size(label_data,2)+1:size(X,2)
   tmp_A=tmp_AAA;
   c=cw;
   sigma   =    c;
   fprintf('i=%f',i);
    Xt_now = tmp_A(:,i);
    xm2   = tmp_A(:,i);%A(:,trls==index);
    xm1   = tmp_A(:,i);%A(:,trls==index); % now
    Xi    =  xm1;
    [gap] = objective_function_2( i,tmp_A,tmp_D1,tmp_D2,tmp_P,X,maxL,LABELS,lambda1,lambda2 ,label_data,per_D_size,M,m);
    prev_f = gap;
    ert(1) = gap;
for n_it = 2 : nIter;
    tmp_A(:,i) = Xi;
    while for_ever
        % IPM estimate         
        grad=gradient_descent_2( i,tmp_A,tmp_D1,tmp_D2,tmp_P,X,maxL,lambda1,lambda2,DD,per_D_size,lambda3,X_size,LABELS,label_data,M,m);
        v = xm1-grad./(2*sigma);
        tem = soft(v,tau/sigma);
        x_temp   =   tem;
        
        if (IST_iters >= 2) || ( TwIST_iters ~= 0)
            % set to zero the past when the present is zero
            % suitable for sparse inducing priors
            if sparse
                mask    =   (x_temp ~= 0);
                xm1     =   xm1.* mask;
                xm2     =   xm2.* mask;
            end
            % two-step iteration
            xm2    =   (alpha-beta)*xm1 + (1-alpha)*xm2 + beta*x_temp;
            % compute residual
            tmp_A(:,i) = xm2;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [gap] = objective_function_2( i,tmp_A,tmp_D1,tmp_D2,tmp_P,X,maxL,LABELS,lambda1,lambda2 ,label_data,per_D_size,M,m);
           f   =   gap;          
            if (f > prev_f) && (enforceMonotone)
                TwIST_iters   =  0;  % do a IST iteration if monotonocity fails
            else
                TwIST_iters =   TwIST_iters+1; % TwIST iterations
                IST_iters   =    0;
                x_temp      =   xm2;
                if mod(TwIST_iters,10000) ==0
                   c = 0.9*c; 
                   sigma = c;
                end
                break;  % break loop while
            end
        else
        tmp_A(:,i) = x_temp;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [gap] = objective_function_2( i,tmp_A,tmp_D1,tmp_D2,tmp_P,X,maxL,LABELS,lambda1,lambda2 ,label_data,per_D_size,M,m);
    
        f   =   gap;
         
         if f > prev_f
                % if monotonicity  fails here  is  because
                % max eig (A'A) > 1. Thus, we increase our guess
                % of max_svs
                c         =    2*c;  
                sigma     =    c;
                if verbose
%                     fprintf('Incrementing c=%2.2e\n',c);
                end
                if  c > cT
                    break;  % break loop while    
                end
                IST_iters = 0;
                TwIST_iters = 0;
           else
                TwIST_iters = TwIST_iters + 1;
                break;  % break loop while
          end
        end
        
    end

    citerion      =   abs(f-prev_f)/abs(prev_f);
    if citerion < citeT || c > cT
%        fprintf('Stop!\n c=%2.2e\n citerion=%2.2e\n',c,citerion);
       break;
    end
    
    xm2           =   xm1;
    xm1           =   x_temp;
    Xt_now        =   x_temp;
    Xi            =   Xt_now; 
    prev_f        =   f;
    ert(n_it)     =   f;
%     fprintf('Iteration:%f  Total gap:%f\n',n_it,ert(n_it-1));
 end  


tmp_AA(:,i)  =      Xt_now;
opts.ert   =       ert;%%%%%暂未使用
end
tmp_A=tmp_AA;
end
