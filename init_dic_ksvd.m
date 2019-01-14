function [ ini_Dic1,ini_Dic2,A ] = init_dic_ksvd( per_D_size,training_data,labels,sparsitythres,X)
%init_dic 初始化字典
%   D_size 字典的原子个数
%   training_data 训练数据
%   类别数
%   latent_P 样本的类别样本
maxL = max(labels);
sizeM = size(training_data);
ini_Dic1 = zeros(sizeM(1),per_D_size,maxL);
ini_Dic2 = zeros(sizeM(1),maxL*per_D_size);
tmp_k = 1;
 for i = 1:maxL
     [~,tmp_loc] = find(labels == i);
     params.data = training_data(:,tmp_loc);
    params.Tdata = sparsitythres; % spasity term
    params.iternum = 3;
    params.memusage = 'high';
    D_ext3 = rand(sizeM(1),per_D_size);
    D_ext3 = normcols(D_ext3); % normalization
    params.initdict = D_ext3;

    [D,~,~] = ksvd(params,'');
    ini_Dic1(:,:,i) = D;
    tmp_k_after = i*per_D_size;
    ini_Dic2(:,tmp_k:tmp_k_after) = D;
    tmp_k = tmp_k_after + 1;
end
%        ini_Dic1 = rand(sizeM(1),per_D_size,maxL);
%        ini_Dic1 = normcols(ini_Dic1);
       ini_Dic2=reshape(ini_Dic1,size(X,1),[]);
    G = ini_Dic2'*ini_Dic2;%%%%%%%%%%%%%%%%%%%%%%%
    A = omp(ini_Dic2'* X , G ,sparsitythres);%稀疏系数%%%%%%%%%%%%%%%%
%      A = rand(per_D_size*maxL,sizeM(2));
end

