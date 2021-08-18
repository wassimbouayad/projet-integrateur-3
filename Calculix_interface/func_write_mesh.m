function [  ] = func_write_mesh( fout, coord, connec, eid, nset )

% Node
fprintf(fout, '*Node, NSET=nset_all\n');
fprintf(fout, '%d, %e, %e, %e\n', coord');
func_write_separator(fout, 'small');

% Elements
fprintf(fout, '*Element, TYPE=S8R, ELSET=elset_all\n');
fprintf(fout, '%d, %d, %d, %d, %d, %d, %d, %d, %d\n', connec');
func_write_separator(fout, 'small');

% Sets
func_write_sets(fout, eid, nset);
func_write_separator(fout, 'small');

end

