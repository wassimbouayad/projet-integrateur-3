function [  ] = func_write_inp( fName, coord, connec, nset, h, E, nu, ...
    heq, epseq11, epseq22, xTop, xBtm,xMid, nlGeom, nOutputRequestIntervals )
% ...

% Inputs:
% fName:
% coord:
% connec:
% nset:
% elset:
% h:
% E:
% nu:
% heq:
% epseq:
% xTop:
% xBtm:

% ...
nn = size(coord, 1);
ne = size(connec, 1);
nid = 1:1:nn;
eid = 1:1:ne;
coord = [nid(:), coord];
connec = [eid(:), connec];

% ...
foutName = sprintf('%s.inp', fName);
fout = fopen(foutName, 'w');

% Write job-dependent blocks
func_write_mesh(fout, coord, connec, eid, nset);
func_write_materials(fout, eid, E, nu, epseq11, epseq22, xTop, xBtm, xMid);
func_write_section_properties(fout, eid, h, heq);

% BCs
fprintf(fout, '*BOUNDARY\n');
fprintf(fout, 'nset_encastre, 1,6\n');
##fprintf(fout, 'nset_xsym, 2\n');
##fprintf(fout, 'nset_xsym, 4\n');
##fprintf(fout, 'nset_xsym, 6\n');
##fprintf(fout, 'nset_ysym, 1\n');
##fprintf(fout, 'nset_ysym, 5\n');
##fprintf(fout, 'nset_ysym, 6\n');
func_write_separator(fout, 'small');

% Initial conditions
fprintf(fout, '*Initial Conditions, type=TEMPERATURE\n');
fprintf(fout, 'nset_all, 0.0\n');
func_write_separator(fout, 'big');

% Step
fprintf(fout, '*Step,%s\n', nlGeom);
fprintf(fout, '*Static\n');
fprintf(fout, '1.0, 1.0, 1e-05, 1.0\n');
func_write_separator(fout, 'small');

% Loads
fprintf(fout, '*Temperature\n');
fprintf(fout, 'NSET_ALL, 1.0\n');
func_write_separator(fout, 'small');

% Outputs
%fprintf(fout, '*Output, field, NUMBER INTERVAL=%d\n', nOutputRequestIntervals);
fprintf(fout, '*Node FILE\n');
fprintf(fout, 'U\n');
fprintf(fout, '*Node PRINT, NSET=NSET_ALL\n');
fprintf(fout, 'U\n');
%fprintf(fout, '*EL FILE\n');
%fprintf(fout, 'S\n');
func_write_separator(fout, 'small');

fprintf(fout, '*End Step\n');

fclose(fout);

end
