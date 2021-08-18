## Copyright (C) 2021 Wassim
## 
## This program is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see
## <https://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {} {@var{retval} =} compute (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Wassim <wassim@wassim>
## Created: 2021-06-16

function retval = compute (config)
  %you have two configs either test or mapped
  if config== 'mapped'
    jobName = 'mapped';
    load('../mesh_8node_sym_mapped.mat');
    coord=points;
    connec=cells{1}+1;
  elseif config=='test'
    jobName = 'test';
    load('../mesh/mesh_8node_sym.mat');
    coord=points{1};
    connec=cells{1}+1;

   end

  
symx=find(coord(:,2)==0)
symy=find(coord(:,1)==0)

nset.encastre = 901;
##nset.xsym = symx;
##nset.ysym = symy;
h = 1;
E = 71500;
nu = 0.33;
heq = 0.1;
% epseq = 4e-3;
dim_less_load=20;
##dim_less_load=str2num(arg_list{2});
epseq = (2*dim_less_load*h**4)/(12*(36**2)*heq*(h-heq));

##patterns=sprintf('../patterns/pattern_%s.mat',jobName);
##load(patterns);
%xTop=reshape(pattern{1},[length(connec),1]);
##pattern{1}=flip(pattern{1},1);
xTop=ones(size(connec,1),1);
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
func_write_inp(jobName, coord, connec, nset, h, E, nu, heq, epseq, ...
    xTop, xBtm,xMid, nlGeom, nOutputRequestIntervals);

% Run job
##cmd=sprintf('ccx %s > /dev/null 2>&1',jobName);
cmd=sprintf('ccx %s',jobName);
system(cmd);

%dos_cmd = sprintf('abaqus job=%s interactive', jobName);
%status = dos(dos_cmd, '-echo');

endfunction
