function [  ] = func_write_materials( fout, eid, E, nu, epseq11,epseq22, xTop, xBtm, xMid)

ne = numel(eid);

for ii = 1:ne

    % Top face
    currentMaterialName = sprintf('mymaterial_active_top_elem_%d', ii);
    fprintf(fout, '*Material, name=%s\n', currentMaterialName);
    fprintf(fout, '*Elastic\n');
    fprintf(fout, '%g, %g\n', E, nu);
    fprintf(fout, '*Expansion, type=ANISO\n');
    fprintf(fout, '%g, %g, 0., 0., 0., 0.\n', epseq11*xTop([ii,ii]),epseq22*xTop([ii,ii]));



    % Mid plate
    currentMaterialName = sprintf('mymaterial_active_mid_elem_%d', ii);
    fprintf(fout, '*Material, name=%s\n', currentMaterialName);
    fprintf(fout, '*Elastic\n');
    fprintf(fout, '%g, %g\n', E, nu);
    fprintf(fout, '*Expansion, type=ANISO\n');
    fprintf(fout, '%g, %g, 0., 0., 0., 0.\n', epseq11*xMid([ii,ii]),epseq22*xMid([ii,ii]));




    % Btm face
    currentMaterialName = sprintf('mymaterial_active_btm_elem_%d', ii);
    fprintf(fout, '*Material, name=%s\n', currentMaterialName);
    fprintf(fout, '*Elastic\n');
    fprintf(fout, '%g, %g\n', E, nu);
    fprintf(fout, '*Expansion, type=ANISO\n');
    fprintf(fout, '%g, %g, 0., 0., 0., 0.\n', epseq11*xBtm([ii,ii]),epseq22*xBtm([ii,ii]));

end

fprintf(fout, '*Material, name=mymaterial_passive\n');
fprintf(fout, '*Elastic\n');
fprintf(fout, '%f, %f\n', E, nu);

func_write_separator(fout, 'small');

end
