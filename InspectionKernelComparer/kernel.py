import numpy as np
import numpy.linalg as la
np.seterr(divide='ignore', invalid='ignore')
#POLYNOMIAL KERNEL m E N, a>0 
def polynomial(x,y,gamma,a):
    m=gamma
    return (a*np.dot(x,y))**m

#RBF gamma>0, beta E (0,2]
def mrbf(x,y,gamma=1,a=2):
    beta=a
    sm=0
    for i, xi in enumerate(x):
        sm+=gamma*(xi-y[i])**beta        
    return np.exp(-sm)

#Hyperbolic tangent kernel a0>0, b<0
def hyperbolic(x,y,a=-1,gamma=1):
    b=a
    a0=gamma
    return np.tanh(a0*np.dot(x,y)+b)

#Triangle a>0
def triangle(x,y,a,gamma=0.1):    
    norm=la.norm(np.subtract(x, y))
    if norm<=a:
        return 1-norm/a    
    return 0   

#ANOVA gamma>0, m E N 
def radial_basic(x, y, gamma=1,a=1): 
    m=a   
    sm=0
    # for i, xi in enumerate(x):
    #     sm+=np.exp(-gamma*(xi-y[i])**2)        
    sm=np.sum(np.exp(-gamma*((x-y)**2)))
    return sm**m



#Rational quadratic a>0
def rquadratic(x,y,a=0.1,gamma=0.1):
    norm=la.norm(np.subtract(x, y))
    return 1-(norm**2)/(norm**2+a)

#Canberra gamma E (0,1]
def canberra(x,y,a,gamma=0.1):
    sm=0
    d=x.shape[0]
    if(gamma>1):
        gamma=1

    sm=np.sum(gamma*np.abs(x-y)/(np.abs(x)+np.abs(y)))        
    # for i, xi in enumerate(x):                        
    #     sm+=abs(xi-y[i])/(abs(xi)+abs(y[i]))                   
    return 1-sm/d
#Truncated gamma>0
def truncated(x,y,a,gamma=0.1):
    sm=0
    d=x.shape[0]    
    for i, xi in enumerate(x):
        val=1-np.abs(xi-y[i])/gamma
        if val>0:
            sm+=val
    return sm/d


def proxy_kernel(X, Y,a=0.1,gamma=0.1, K=triangle):   
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i, j] = K(x,y,a=a,gamma=gamma)
    # print("gram_matrix")        
    # print(gram_matrix)
    if(np.any(np.isnan(gram_matrix))):
        gram_matrix[np.isnan(gram_matrix)]=gram_matrix[~np.isnan(gram_matrix)].mean()
    return gram_matrix

#---------------------------------
#POLYNOMIAL
def polynomial_kernel(gamma,a):
    def pk(X,Y):
        return proxy_kernel(X,Y,a=a,gamma=gamma,K=polynomial)
    return pk
#MRBF
def mrbf_kernel(gamma=1,a=2):
    def mrbfk(X,Y):
        return proxy_kernel(X,Y,a=a,gamma=gamma,K=mrbf)
    return mrbfk
#HIPERBOLIC TAN
def hyperbolic_kernel(gamma=-1,a=0):
    def hk(X,Y):
        return proxy_kernel(X,Y,a=a,gamma=gamma,K=hyperbolic)
    return hk
#TRIANGLE
def triangle_kernel(a):    
    def tk(X,Y):
        return proxy_kernel(X,Y,a=a,K=triangle)
    return tk
#ANOVA
def radial_basic_kernel(gamma,a=1):    
    def rbk(X,Y):
        return proxy_kernel(X,Y,a=a,gamma=gamma,K=radial_basic)
    return rbk
#RATIONAL QUADRATIC
def rquadratic_kernel(gamma,a=0.1):    
    def rqk(X,Y):
        return proxy_kernel(X,Y,a=a,gamma=gamma,K=rquadratic)
    return rqk

#CANBERRA
def canberra_kernel(gamma=0.1):
    if(gamma>1):
        gamma=1/gamma
    def ck(X,Y):
        return proxy_kernel(X,Y,gamma=gamma,K=canberra)
    return ck

#TRUNCATED
def truncated_kernel(gamma=0.1):
    def trk(X,Y):
        return proxy_kernel(X,Y,gamma=gamma,K=truncated)
    return trk
