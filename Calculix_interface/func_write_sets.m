function [  ] = func_write_sets( fout, eid, nset )

% Write node sets
temp = fieldnames(nset);
for ii = 1:numel(temp)
    currentFieldName = temp{ii};
    currentNSet = nset.(currentFieldName);
    fprintf(fout, '*NSET, NSET=nset_%s\n', currentFieldName);
    func_write_int_list(fout, currentNSet, 16);
end

% Write element sets
% Todo

% Write one elset for each element (to assign section properties)
ne = length(eid);
for ii = 1:ne
    currentSet = eid(ii);
    currentSetName = sprintf('elset_elem_%d', ii);
    fprintf(fout, '*ELSET, ELSET=%s\n', currentSetName);
    func_write_int_list(fout, currentSet, 16);
end

end