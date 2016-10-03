function [res, Aopt, Dopt, Gopt] = ADG_optimality(y, X, W, pd, pr_pd, parnames)


invI = inv(X'*X);
Aopt = trace(invI);
Dopt = det(invI);
Gopt = max(diag(X*(invI)*X'));

res = {};
res.A = Aopt;
res.D = Dopt;
res.G = Gopt;

end