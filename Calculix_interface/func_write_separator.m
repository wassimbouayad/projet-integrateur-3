function [   ] = func_write_separator( fout, type )

switch type
    case 'small'
        fprintf(fout, '**\n');
    case 'big'
        fprintf(fout, '**\n');
        fprintf(fout, '** --------------\n');
        fprintf(fout, '**\n');
end

end