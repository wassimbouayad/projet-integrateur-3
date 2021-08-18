function [  ] = func_write_section_properties( fout, eid, h, heq )

ne = numel(eid);

for ii = 1:ne

    currentElset = sprintf('elset_elem_%d', ii);
    currentMaterialTop = sprintf('mymaterial_active_top_elem_%d', ii);
    currentMaterialMid = sprintf('mymaterial_active_mid_elem_%d', ii);
    currentMaterialBtm = sprintf('mymaterial_active_btm_elem_%d', ii);

%    fprintf(fout, '*SHELL SECTION, MATERIAL=%s,ELSET=%s, COMPOSITE\n', currentElset);
    fprintf(fout,'*SHELL SECTION,ELSET=%s,OFFSET=0, COMPOSITE\n',currentElset);
    fprintf(fout, '%g, ,%s\n', heq, currentMaterialTop);
    fprintf(fout, '%g, ,%s\n', h - 2*heq, currentMaterialMid);
    fprintf(fout, '%g, ,%s\n', heq, currentMaterialBtm);

end

func_write_separator(fout, 'small');

end
