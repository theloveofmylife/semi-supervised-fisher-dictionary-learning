tic
addpath(genpath('.\ksvdbox'));  % add K-SVD box
addpath(genpath('.\OMPbox'));   % add sparse coding algorithem OMP
load('.\zao2_zidiandaxiao_25_yangben_50');
%%   参数设定   
sparsitythres     =   1;          %OMP方法的稀疏度

max_iteration     =   40;         %更新字典与系数的迭代次数

per_D_size        =   25;        %每一类字典的大小

lambda1           =   0.001;      %F的尺度参数

lambda2           =   0.005;%1.2/sqrt(size(X,1));     %稀疏度的尺度参数

lambda3           =   1;          %||A_i||F范数的参数

threshold         =   0.7;        %更新P时的阈值

o                 =   0.3;        %更新P中的sigema

Dataway           =   'lilun';    %数据的来源，目前有三种：lilun为尖灭模型的理论地震数据，shiji为中国某工区实际地震数据，zidingyi为用户根据需要可自行编辑

Iniway            =   'pca';    %初始化方法，目前有四种：suiji为随机初始化，pca为主成分分析初始化，ksvd为KSVD字典初始化，zidingyi为用户根据需要可自行编辑

Classificationway =   'homemade'; %测试数据稀疏系数的求解方法，目前有两种：homemade为自己编写的程序，fisher为借鉴FDDL的方法

class_rate_way    =   'everytime';%是否每一次迭代都求解分类准确率，everytime为每一次都求解，noeverytime为仅迭代最后一次求解
%%   数据获取
% [ label_data,unlabel_data,labels,test_data,tast_labels ] = data_resource( trace,0,Dataway );%data_resource( 标签数据,未标签数据,数据的来源)，使用理论数据时所有数据均为标签数据
% X=[label_data,unlabel_data];% data, d*(N+M)
% X = X*diag(1./t(sum(X.*X)));
maxL = max(labels);
%%   初始化
size_Udata = size(unlabel_data);
ini_P = init_latent_P(labels,size_Udata);%初始化 P
fprintf(' ini_P done! \n')
switch lower(Iniway)
    case {'suiji'}
        [ini_D1,ini_D2,~] = init_dic_suiji(per_D_size,label_data,labels,sparsitythres,X);
        ini_A = rand(per_D_size*maxL,size(X,2));
    case {'pca'}
        [ini_D1,ini_D2,ini_A] = init_dic_fisher(per_D_size,label_data,labels,sparsitythres,X,lambda1,lambda2);
    case {'ksvd'}
        [ini_D1,ini_D2,ini_A] = init_dic_ksvd(per_D_size,label_data,labels,sparsitythres,X,lambda1,lambda2);
    case {'zidingyi'}
        %可根据需要自己编辑
    otherwise 
        fprintf('Unkonw method of initialize.\n');
end
fprintf(' ini_D  ini_A done! \n')
ini_A=full(ini_A);
%ini_P = Update_confidence( ini_A,ini_D1,X,maxL,per_D_size ,label_data,threshold,o,ini_P);
%%   第一次更新
[tmp_A,JJ ] = Update_coefficients( ini_A,ini_D1,ini_D2,ini_P,X,maxL,labels,lambda1,lambda2,lambda3,label_data,per_D_size ) ;
% [tmp_A,JJ] = Update_coefficients( tmp_A,ini_D1,ini_D2,ini_P,X,maxL,labels,lambda1,lambda2,lambda3,label_data,per_D_size ) ;
J(1)=JJ;
 [tmp_D1,tmp_D2]=Update_dictionary( tmp_A,ini_D1,ini_D2,ini_P,X,maxL,per_D_size );
%  [tmp_D1,tmp_D2]=Update_dictionary( tmp_A,ini_D1,ini_D2,ini_P,X,maxL,per_D_size );
n=1;
tmp_P = Update_confidence( tmp_A,tmp_D1,X,maxL,per_D_size ,label_data,threshold,o,ini_P,n); 
class_ratee=zeros(1,max_iteration);
class_rateee=zeros(max_iteration,size(test_data,2));
     G = tmp_D2'*tmp_D2;
     tmp_A1 = omp(tmp_D2'* label_data , G ,sparsitythres);%稀疏系数
     tmp_A1=full(tmp_A1);
if strcmp(class_rate_way,'everytime')==1
switch lower(Classificationway)
    case {'fisher'}
%%
        nClass   =   maxL;
        weight   =   0.005;
        td1_ipts.D    =   tmp_D2;
        td1_ipts.tau1 =   lambda2;
        if size(td1_ipts.D,1)>=size(td1_ipts.D,2)
            td1_par.eigenv = 0.02*eigs(td1_ipts.D'*td1_ipts.D,1);
        else
            td1_par.eigenv = 0.02*eigs(td1_ipts.D*td1_ipts.D',1);
        end
        
        for indTest = 1:size(test_data,2)
            %     fprintf(['Totalnum:' num2str(size(test_data,2)) 'Nowprocess:' num2str(indTest) '\n']);
            td1_ipts.y          =      test_data(:,indTest);   
            [opts]              =      IPM_SC(td1_ipts,td1_par);
            s                   =      opts.x;
            test_A(:,indTest)   =      s;
            tmp_A1              =     tmp_A;
        end
%%        
    case {'homemade'}
        test_A = omp(tmp_D2'* test_data , G ,sparsitythres); 
end        
switch lower(Dataway)
    case {'lilun'}
        [ class,class_rate ] = Classification_function_lilun( test_data,tmp_A1,tmp_D1,tmp_D2,maxL,labels,per_D_size,sparsitythres,tast_labels ,X,test_A);
    case {'shiji'}
        [ class,class_rate ] = Classification_function_shiji( test_data,tmp_A1,tmp_D1,tmp_D2,maxL,labels,per_D_size,sparsitythres,tast_labels ,X,test_A);
    case {'zidingyi'}
        %可根据需要自己编辑
    otherwise 
        fprintf('Unkonw method of data.\n');
end
 class_ratee(1,1)=class_rate;
 class_rateee(1,:)=class;
 n=1; 
 fprintf('n=%f\n',n);
 fprintf('class_rate=%f\n',class_rate);
end
%%   迭代更新
for n =2:(max_iteration-1) 
    
    fprintf('n=%f\n',n);
    
    [tmp_A,JJ] = Update_coefficients( tmp_A,tmp_D1,tmp_D2,tmp_P,X,maxL,labels,lambda1,lambda2,lambda3,label_data,per_D_size ) ;
%     [tmp_A,JJ] = Update_coefficients( tmp_A,tmp_D1,tmp_D2,tmp_P,X,maxL,labels,lambda1,lambda2,lambda3,label_data,per_D_size ) ;
    J(n)=JJ;
    fprintf('A=%f\n',n);
    
    [tmp_D1,tmp_D2]=Update_dictionary( tmp_A,tmp_D1,tmp_D2,tmp_P,X,maxL,per_D_size );
%     [tmp_D1,tmp_D2]=Update_dictionary( tmp_A,tmp_D1,tmp_D2,tmp_P,X,maxL,per_D_size );
    fprintf('D=%f\n',n);
    
    tmp_P = Update_confidence( tmp_A,tmp_D1,X,maxL,per_D_size,label_data ,threshold,o,tmp_P,n);
    
    fprintf('P=%f\n',n);
     G = tmp_D2'*tmp_D2;
     tmp_A1 = omp(tmp_D2'* label_data , G ,sparsitythres);%稀疏系数
     tmp_A1=full(tmp_A1);
%      [ class,class_rate ] = Classification_function( test_data,tmp_A1,tmp_D1,tmp_D2,maxL,LABELS,per_D_size,sparsitythres,tast_labels ,X,test_A);
if strcmp(class_rate_way,'everytime')==1
switch lower(Classificationway)
    case {'fisher'}
%%
        nClass   =   maxL;
        weight   =   0.005;
        td1_ipts.D    =   tmp_D2;
        td1_ipts.tau1 =   lambda2;
        if size(td1_ipts.D,1)>=size(td1_ipts.D,2)
            td1_par.eigenv = 0.02*eigs(td1_ipts.D'*td1_ipts.D,1);
        else
            td1_par.eigenv = 0.02*eigs(td1_ipts.D*td1_ipts.D',1);
        end
        
        for indTest = 1:size(test_data,2)
            %     fprintf(['Totalnum:' num2str(size(test_data,2)) 'Nowprocess:' num2str(indTest) '\n']);
            td1_ipts.y          =      test_data(:,indTest);   
            [opts]              =      IPM_SC(td1_ipts,td1_par);
            s                   =      opts.x;
            test_A(:,indTest)   =      s;
            tmp_A1              =      tmp_A;
        end
%%        
    case {'homemade'}
        test_A = omp(tmp_D2'* test_data , G ,sparsitythres); 
end        
switch lower(Dataway)
    case {'lilun'}
        [ class,class_rate ] = Classification_function_lilun( test_data,tmp_A1,tmp_D1,tmp_D2,maxL,labels,per_D_size,sparsitythres,tast_labels ,X,test_A);
    case {'shiji'}
        [ class,class_rate ] = Classification_function_shiji( test_data,tmp_A1,tmp_D1,tmp_D2,maxL,labels,per_D_size,sparsitythres,tast_labels ,X,test_A);
    case {'zidingyi'}
        %可根据需要自己编辑
    otherwise 
        fprintf('Unkonw method of data.\n');
end
     class_ratee(1,n)=class_rate;
     class_rateee(n,:)=class;
     fprintf('class_rate=%f\n',class_rate);
end
end
%%   最后一次更新
fprintf('n=%f\n',n);
[tmp_A,JJ] = Update_coefficients( tmp_A,tmp_D1,tmp_D2,tmp_P,X,maxL,labels,lambda1,lambda2,lambda3,label_data,per_D_size ) ;
% [tmp_A,JJ] = Update_coefficients( tmp_A,tmp_D1,tmp_D2,tmp_P,X,maxL,labels,lambda1,lambda2,lambda3,label_data,per_D_size ) ;
J(n+1)=JJ;
[tmp_D1,tmp_D2]=Update_dictionary( tmp_A,tmp_D1,tmp_D2,tmp_P,X,maxL,per_D_size );
% [tmp_D1,tmp_D2]=Update_dictionary( tmp_A,tmp_D1,tmp_D2,tmp_P,X,maxL,per_D_size );     
G = tmp_D2'*tmp_D2;
     tmp_A1 = omp(tmp_D2'* label_data , G ,sparsitythres);%稀疏系数
     tmp_A1=full(tmp_A1);
%      [ class,class_rate ] = Classification_function( test_data,tmp_A1,tmp_D1,tmp_D2,maxL,LABELS,per_D_size,sparsitythres,tast_labels ,X,test_A);
switch lower(Classificationway)
    case {'fisher'}
%%
        nClass   =   maxL;
        weight   =   0.005;
        td1_ipts.D    =   tmp_D2;
        td1_ipts.tau1 =   lambda2;
        if size(td1_ipts.D,1)>=size(td1_ipts.D,2)
            td1_par.eigenv = 0.02*eigs(td1_ipts.D'*td1_ipts.D,1);
        else
            td1_par.eigenv = 0.02*eigs(td1_ipts.D*td1_ipts.D',1);
        end
        
        for indTest = 1:size(test_data,2)
            %     fprintf(['Totalnum:' num2str(size(test_data,2)) 'Nowprocess:' num2str(indTest) '\n']);
            td1_ipts.y          =      test_data(:,indTest);   
            [opts]              =      IPM_SC(td1_ipts,td1_par);
            s                   =      opts.x;
            test_A(:,indTest)   =      s;
            tmp_A1              =      tmp_A;
        end
%%        
    case {'homemade'}
        test_A = omp(tmp_D2'* test_data , G ,sparsitythres); 
end        
switch lower(Dataway)
    case {'lilun'}
        [ class,class_rate ] = Classification_function_lilun( test_data,tmp_A1,tmp_D1,tmp_D2,maxL,labels,per_D_size,sparsitythres,tast_labels ,X,test_A);
    case {'shiji'}
        [ class,class_rate ] = Classification_function_shiji( test_data,tmp_A1,tmp_D1,tmp_D2,maxL,labels,per_D_size,sparsitythres,tast_labels ,X,test_A);
    case {'zidingyi'}
        %可根据需要自己编辑
    otherwise 
        fprintf('Unkonw method of data.\n');
end
class_ratee(1,max_iteration)=class_rate;
class_rateee(max_iteration,:)=class;
n=max_iteration; 
fprintf('n=%f\n',n);
fprintf('class_rate=%f',class_rate); 
figure;
imagesc(class');
colormap(jet(2));
title('半监督判别字典');
toc