function [ss, p, m] = KLSampleSize(X, y, pars)
% nonparametric version, July 2015
alpha = pars.alpha;
nSamples = pars.nSamples;
ss_final = 0;
p_final = zeros(1,length(y));
m_final = p_final;
count = p_final;
for i = 1:50
    stop = false;
    ss = length(y)-100;
    p = [];
    m = [0, length(y)];
    while ~stop & ss > 0
        m(end+1) = ss;
        [~, p(end + 1)] = kltest(X, y, ss, 100, alpha);  
        %ge(end+1) = calcError(X, y, ss);    
        if 1-p(end) > alpha
            ss = round((min(m(m > ss))+ss)/2);
            %stop = false;
        else
            if sum(p(1:end-1) < 1-alpha) > 0
                stop = true;
            else
            ss = round((ss + max(m(m < ss)))/2);
            end
        end    
    end
    m = m(3:end);
    [m, idx] = sort(m);
    p = smooth(p(idx))';
    idx = find(p > 1 - alpha);
    %
    if isempty(idx)
        idx = length(m);
    end
    %}
    ss_final = ss_final + m(idx(1));
    count(m) = count(m) + 1;
    p_final(m) = p_final(m) + p;
    m_final(m) = m;
end
ss = ss_final/i;
count = count(m_final~= 0);
p = p_final(m_final~= 0)./count;
m = m_final(m_final~= 0);
%{
if isempty(ss)
    ss = m(end);
end
%}
%h = fliplr(h);
%ge = fliplr(ge);
%m = fkiplr(m);
end

function [h, p] = kltest(X, y, ss, nSamples, alpha)

N_BINS = 20;
for i = 1:nSamples
    idx1 = randi(length(y),ss,1);
    idx2 = randi(length(y),ss,1);
    s1 = X(idx1,:);
    s2 = X(idx2,:);
    stat(i) = length(s1)*length(s2)*KLdiv(s1, s2, N_BINS)/(length(s1)+length(s2));    
end
chi1 = chi2inv(1 - alpha, 2*N_BINS);
h = chi1 > stat & 0 < stat;
p = mean(h);
h = chi1 > mean(stat) & 0 < mean(stat);
end

function dKL = KLdiv(X1, X2, nbins)

p1 = hist(X1, nbins-1)/length(X1);
p2 = hist(X2, nbins-1)/length(X2);

lstDKL = p1.*log(p1./p2);
idx = lstDKL == Inf | lstDKL == -Inf;
lstDKL = lstDKL.*(~idx) + idx*abs(max(lstDKL(~idx)));

dKL = nansum(lstDKL);

end