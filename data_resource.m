function [ label_data,unlabel_data,labels,test_data,tast_labels ] = data_resource( training_feats,testing_feats,Dataway )
%DATA 数据获取
%   此处显示详细说明
switch lower(Dataway)
    case {'lilun'}
        trace=training_feats;
        [ label_data,unlabel_data,labels,test_data,tast_labels ] = lilun( trace );
    case {'shiji'}
        [ label_data,unlabel_data,labels,test_data,tast_labels ] = shiji( training_feats,testing_feats );
    case {'zidingyi'}
        %可根据需要自己编辑
    otherwise 
        fprintf('Unkonw method of data.\n');
end
end

