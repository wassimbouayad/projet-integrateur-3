function [  ] = func_write_int_list( fout, v, nItemPerLine )
% Write a comma-separated list of integers with nItemPerLine items per line

n = numel(v);

for ii = 1:n
   if ii == n % Last entry
       frmt = '%d\n';
   elseif (mod(ii, nItemPerLine) == 0) % End of line continuation
       frmt = '%d,\n';
   else
       frmt = '%d,';
   end
   fprintf(fout, frmt, v(ii));
end

end