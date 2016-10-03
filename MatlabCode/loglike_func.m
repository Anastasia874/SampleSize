function lhd = loglike_func(y, X, probs, pd, w, parnames)

lhd = zeros(numel(pd), size(X, 2));
for i = 1:numel(pd)
    % assign parameter values from vector w, for parametrs of pd,
    % specified by parnames
    [pd{i}, lh] = assign_pars(pd{i}, w(1:numel(parnames{i})), parnames{i});
    if lh == 0
        break;
    end
    w = w(numel(parnames{i})+1:end);
    for j = 1:size(X, 2)
        lhd(i, j) = prod(pdf(pd{i}(j), X(y == i, j)));  
    end
end

lhd = log(probs*prod(lhd, 2));

end

function [pd, lh] = assign_pars(pd, w, parnames)

lh = 1;
for npar = 1:numel(parnames)
    % FIXIT replace with checkargs instead
    try 
    pd.(parnames{npar}) = w(npar);
    catch
    lh = 0;
    end
end
    
end