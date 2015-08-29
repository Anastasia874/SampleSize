from scipy.optimize import fmin_bfgs
from numpy import *

def sigmoid(X):
    '''Compute the sigmoid function '''
    #d = zeros(shape=(X.shape))
 
    den = 1.0 + e ** (-1.0 * X) 
    d = 1.0 / den
 
    return d
 
 
def compute_cost(theta,X,y): #computes cost given predicted and actual values
    m = X.shape[0] #number of training examples
    #print('comp_cost: ' + str(type(theta)))
    if isinstance(theta, matrix):
        theta = squeeze(asarray(theta))    
          
         
    theta = reshape(theta,(len(theta),1))
    
    #y = reshape(y,(len(y),1))
    
    J = (1./m) * (-transpose(y).dot(log(sigmoid(X.dot(theta)))) - transpose(1-y).dot(log(1-sigmoid(X.dot(theta)))))
    
    grad = transpose((1./m)*transpose(sigmoid(X.dot(theta)) - y).dot(X))
    #optimize.fmin expects a single value, so cannot return grad
    return J[0][0]#,grad
 
 
def compute_grad(theta, X, y):
    m, n = X.shape
    #if not len(theta) == n:
    #    theta = []

    #print('comp_grad' + str(theta.shape))     
    theta.shape = (1, n) 
    grad = zeros(n) 
    h = sigmoid(X.dot(theta.T))
 
    delta = h - y 
    l = grad.size
 
    for i in range(l):
        sumdelta = delta.T.dot(X[:, i])
        grad[i] = (1.0 / m) * sumdelta * - 1
 
    theta.shape = (n,)
 
    return  grad


def optimize_theta(X, y):
    m, n = X.shape
    theta_0 = random.rand(n)
    res = fmin_bfgs(compute_cost, theta_0, fprime=compute_grad, 
        args=(X, y))
    
    return res
    
