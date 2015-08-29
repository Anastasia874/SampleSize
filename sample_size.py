import sys
#from scipy.stats import norm
#import math
import numpy as np
from readArgs import * #read_args
import calcSampleSize #import *
#from readArgs import read_data


def gen_test_data(mu1, mu0, sigma, p):
    p_vec = np.random.rand(1, 50)
    y = (p_vec) > p
    x1 = (np.random.randn(1, 50)*sigma + mu1)*y
    x2 = (np.random.randn(1, 50)*sigma+mu0)*(1-y)
    data = np.array([y, x1 + x2])
    
    
    return data

delim = ','
msg = ''
default = {'method': 'Wald test', 'alpha':0.05,
           'power':0.95, 'delta':0.05}
methods2func = {'Equality test':'ss_equality_test', 
                'Normality test': 'sample_size_norm',
                'Wald test': 'sample_size_wald',
                'Superiority test':'ss_super_test'}

try:    
    filename = 'uploads/' #+ sys.argv[1]
    f = open(filename, 'r')
    text_data = f.readlines()
    data_by_rows = read_data(text_data, delim)
    data = np.matrix(data_by_rows)
except:    
    msg += 'Failed to open file ' + filename + '\n'
    data = gen_test_data(1, 0, 1, 0.4)


(arg_msg, pars) = read_args(sys.argv, default) 
msg += arg_msg


#print(len(text_data))

y = np.matrix(data[0,:].transpose())
x = np.matrix(data[1,:].transpose())
print(x.shape, y.shape)
#data_by_cols = zip(*data_by_rows)

try:
    func_name = methods2func[pars['method']]
except:
    func_name = 'sample_size_norm'
        
size = getattr(calcSampleSize, func_name)(y, x, pars)
#size = sample_size_norm(y, x, pars['alpha'])
print(msg + 'Estimated sample size is ' + str(size))


    
    
    

