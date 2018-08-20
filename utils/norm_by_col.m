function [W_norm] = norm_by_col(W)
% Functon normalizes input adjacency matrix by columns
% Input
% W - [n,n] - adjacency matrix
%
% Output
% W_norm - [n,n] - adjacency matrix normalized by columns
%
% Author: Aleksandr Katrutsa
% E-mail: aleksandr.katrutsa@phystech.edu
% Date: 20.11.2014

W_norm = W;
col_sum = sum(W);
idx_nonzero_sum = find(col_sum);
for i = idx_nonzero_sum
    W_norm(:, i) = W(:, i) / col_sum(1, i);
end
end
