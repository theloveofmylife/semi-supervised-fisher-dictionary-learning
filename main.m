tic
addpath(genpath('.\ksvdbox'));  % add K-SVD box
addpath(genpath('.\OMPbox'));   % add sparse coding algorithem OMP
load('.\zao2_zidiandaxiao_25_yangben_50');
%%   �����趨   
sparsitythres     =   1;          %OMP������ϡ���

max_iteration     =   40;         %�����ֵ���ϵ���ĵ�������

per_D_size        =   25;        %ÿһ���ֵ�Ĵ�С

lambda1           =   0.001;      %F�ĳ߶Ȳ���

lambda2           =   0.005;%1.2/sqrt(size(X,1));     %ϡ��ȵĳ߶Ȳ���

lambda3           =   1;          %||A_i||F�����Ĳ���

threshold         =   0.7;        %����Pʱ����ֵ

o                 =   0.3;        %����P�е�sigema

Dataway           =   'lilun';    %���ݵ���Դ��Ŀǰ�����֣�lilunΪ����ģ�͵����۵������ݣ�shijiΪ�й�ĳ����ʵ�ʵ������ݣ�zidingyiΪ�û�������Ҫ�����б༭

Iniway            =   'pca';    %��ʼ��������Ŀǰ�����֣�suijiΪ�����ʼ����pcaΪ���ɷַ�����ʼ����ksvdΪKSVD�ֵ��ʼ����zidingyiΪ�û�������Ҫ�����б༭

Classificationway =   'homemade'; %��������ϡ��ϵ������ⷽ����Ŀǰ�����֣�homemadeΪ�Լ���д�ĳ���fisherΪ���FDDL�ķ���

class_rate_way    =   'everytime';%�Ƿ�ÿһ�ε�����������׼ȷ�ʣ�everytimeΪÿһ�ζ���⣬noeverytimeΪ���������һ�����
%%   ���ݻ�ȡ
% [ label_data,unlabel_data,labels,test_data,tast_labels ] = data_resource( trace,0,Dataway );%data_resource( ��ǩ����,δ��ǩ����,���ݵ���Դ)��ʹ����������ʱ�������ݾ�Ϊ��ǩ����
% X=[label_data,unlabel_data];% data, d*(N+M)
% X = X*diag(1./t(sum(X.*X)));
maxL = max(labels);
%%   ��ʼ��
size_Udata = size(unlabel_data);
ini_P = init_latent_P(labels,size_Udata);%��ʼ�� P
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
        %�ɸ�����Ҫ�Լ��༭
    otherwise 
        fprintf('Unkonw method of initialize.\n');
end
fprintf(' ini_D  ini_A done! \n')
ini_A=full(ini_A);
%ini_P = Update_confidence( ini_A,ini_D1,X,maxL,per_D_size ,label_data,threshold,o,ini_P);
%%   ��һ�θ���
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
     tmp_A1 = omp(tmp_D2'* label_data , G ,sparsitythres);%ϡ��ϵ��
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
        %�ɸ�����Ҫ�Լ��༭
    otherwise 
        fprintf('Unkonw method of data.\n');
end
 class_ratee(1,1)=class_rate;
 class_rateee(1,:)=class;
 n=1; 
 fprintf('n=%f\n',n);
 fprintf('class_rate=%f\n',class_rate);
end
%%   ��������
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
     tmp_A1 = omp(tmp_D2'* label_data , G ,sparsitythres);%ϡ��ϵ��
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
        %�ɸ�����Ҫ�Լ��༭
    otherwise 
        fprintf('Unkonw method of data.\n');
end
     class_ratee(1,n)=class_rate;
     class_rateee(n,:)=class;
     fprintf('class_rate=%f\n',class_rate);
end
end
%%   ���һ�θ���
fprintf('n=%f\n',n);
[tmp_A,JJ] = Update_coefficients( tmp_A,tmp_D1,tmp_D2,tmp_P,X,maxL,labels,lambda1,lambda2,lambda3,label_data,per_D_size ) ;
% [tmp_A,JJ] = Update_coefficients( tmp_A,tmp_D1,tmp_D2,tmp_P,X,maxL,labels,lambda1,lambda2,lambda3,label_data,per_D_size ) ;
J(n+1)=JJ;
[tmp_D1,tmp_D2]=Update_dictionary( tmp_A,tmp_D1,tmp_D2,tmp_P,X,maxL,per_D_size );
% [tmp_D1,tmp_D2]=Update_dictionary( tmp_A,tmp_D1,tmp_D2,tmp_P,X,maxL,per_D_size );     
G = tmp_D2'*tmp_D2;
     tmp_A1 = omp(tmp_D2'* label_data , G ,sparsitythres);%ϡ��ϵ��
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
        %�ɸ�����Ҫ�Լ��༭
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
title('��ල�б��ֵ�');
toc