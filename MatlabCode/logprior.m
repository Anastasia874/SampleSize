function lprior = logprior(w, pd)
% parameters are assumed independent, thus the lhd-s for prior are summed
lprior = zeros(1, numel(pd));
for i = 1:numel(pd)
   lprior(i) = log(pdf(pd{i}, w(i)));  
end
lprior = sum(lprior);

end