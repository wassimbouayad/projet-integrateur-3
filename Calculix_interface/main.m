close all
clear

##cd './Calculix_interface';
% Parameters
##arg_list = argv();

%to run mapped

jobName = 'mapped';
load('../mesh/mesh_8node_sym_mapped.mat');

coord=points;
connec=cells{1}+1;

%to run the test
##this section must be run before the mapped section.

##jobName = 'test';
##load('../mesh/mesh_8node_sym.mat');
##coord=points{1}
##connec=cells{1}+1;





symx=find(coord(:,2)==0)
symy=find(coord(:,1)==0)

nset.encastre = 901;
##nset.xsym = symx;
##nset.ysym = symy;
h = 1;
E = 71500;
nu = 0.33;
heq = 0.1;
% epseq = 4e-3, epseq;
dim_less_load=40;
##dim_less_load=str2num(arg_list{2});

##epseq = (2*dim_less_load*h**4)/(12*(36**2)*heq*(h-heq));
imperf=0.1;
epseq11 = ((2*dim_less_load*h**4)/(12*(36**2)*heq*(h-heq)))*(1+imperf);
epseq22 =((2*dim_less_load*h**4)/(12*(36**2)*heq*(h-heq)))*(1-imperf);

##patterns=sprintf('../patterns/pattern_%s.mat',jobName);
##load(patterns);
%xTop=reshape(pattern{1},[length(connec),1]);
##pattern{1}=flip(pattern{1},1);
xTop = ones(size(connec,1),1);
xMid = zeros(size(connec,1),1);
% xMid = -xTop;
% xBtm= xTop;
xBtm = zeros(size(connec,1),1);

nOutputRequestIntervals = 1;
NL=true;
if NL==true
  nlGeom =  'NLGEOM';
  else
  nlGeom='';
end


% Write inp file
##func_write_inp(jobName, coord, connec, nset, h, E, nu, heq, epseq, ...
##    xTop, xBtm,xMid, nlGeom, nOutputRequestIntervals);
func_write_inp(jobName, coord, connec, nset, h, E, nu, heq, epseq11,epseq22, ...
    xTop, xBtm,xMid, nlGeom, nOutputRequestIntervals);
% Run job
##cmd=sprintf('ccx %s > /dev/null 2>&1',jobName);
cmd=sprintf('ccx %s',jobName);
setenv('OMP_NUM_THREADS', '6');
system(cmd);

%dos_cmd = sprintf('abaqus job=%s interactive', jobName);
%status = dos(dos_cmd, '-echo');
