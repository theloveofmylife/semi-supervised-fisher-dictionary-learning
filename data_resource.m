function [ label_data,unlabel_data,labels,test_data,tast_labels ] = data_resource( training_feats,testing_feats,Dataway )
%DATA ���ݻ�ȡ
%   �˴���ʾ��ϸ˵��
switch lower(Dataway)
    case {'lilun'}
        trace=training_feats;
        [ label_data,unlabel_data,labels,test_data,tast_labels ] = lilun( trace );
    case {'shiji'}
        [ label_data,unlabel_data,labels,test_data,tast_labels ] = shiji( training_feats,testing_feats );
    case {'zidingyi'}
        %�ɸ�����Ҫ�Լ��༭
    otherwise 
        fprintf('Unkonw method of data.\n');
end
end

