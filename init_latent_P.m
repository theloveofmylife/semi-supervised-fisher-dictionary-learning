function [ ini_P ] = init_latent_P( labels, size_Un_data )
%init_latent_P 初始化latent_P
%   labels为标签样本的labels
%   size_Udata 为未标签样本的大小

tmp_size = size(labels,2);
maxP = max(labels);
ini_P_1 = zeros(tmp_size,maxP);

for i = 1:maxP
    [~,tmp_location] = find(labels == i);
    ini_P_1(tmp_location,i) = 1;
end
size_Un=[size_Un_data(2),maxP];
ini_P_2 = rand(size_Un);

ini_P =[ini_P_1;ini_P_2];
end

