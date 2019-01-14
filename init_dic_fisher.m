function [ini_D1,ini_D2,ini_A] = init_dic_fisher(per_D_size,label_data,labels,sparsitythres,X,lambda1,lambda2);
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
maxL = max(labels);
tmp_D1  =  zeros(size(X,1),per_D_size,maxL);
Dict_ini  =  []; 
Dlabel_ini = [];
for ci = 1:maxL
    cdat          =    label_data(:,labels==ci);
   [D,disc_value,Mean_Image]   =    Eigenface_f(cdat,per_D_size-1);
    D                           =    [Mean_Image./norm(Mean_Image) D];
    ini_D1(:,:,ci)  =  D;
    Dict_ini      =    [Dict_ini D];
    Dlabel_ini    =    [Dlabel_ini repmat(ci,[1 size(D,2)])];
end
ini_D2= Dict_ini;
ini_par.tau         =     lambda2;
ini_par.lambda      =    lambda1;
ini_ipts.D          =     Dict_ini;
coef = zeros(size(Dict_ini,2),size(X,2));
if size(Dict_ini,1)>size(Dict_ini,2)
      ini_par.c        =    1.05*eigs(Dict_ini'*Dict_ini,1);
else
      ini_par.c        =    1.05*eigs(Dict_ini*Dict_ini',1);
end
ini_ipts.X      =    X;
    [ini_opts]      =    FDDL_INIC (ini_ipts,ini_par);
    coef =    ini_opts.A;
ini_A = coef;
end

