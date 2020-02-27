import numpy as np
import numpy.linalg as la

#Kernel functions definitions are at Belanche kernel Design paper
np.seterr(divide='ignore', invalid='ignore')

#POLYNOMIAL KERNEL m E N, a>0 
def polynomial(x,y, degree=3, gamma=0.01,coef0=0.0):
    m=degree
    a=gamma
    return (a*np.dot(x,y)+1)**m
#RBF gamma>0, beta E (0,2]
def mrbf(x,y, degree=3, gamma=0.01,coef0=0.0):
    beta=degree    
    sm=np.sum(gamma*(x-y)**beta)
    return np.exp(-sm)

#Hyperbolic tangent kernel a0>0, b<0
def hyperbolic(x,y, degree=3, gamma=0.01,coef0=0.0):
    b=coef0
    a=gamma
    return np.tanh(a*np.dot(x,y)+b)

#Triangle a>0
def triangle(x,y, degree=3, gamma=0.01,coef0=0.0):  
    a=gamma
    norm=la.norm(np.subtract(x, y))
    if norm<=a:
        return 1-norm/a    
    return 0   

#ANOVA gamma>0, m E N 
def radial_basic(x, y, degree=3, gamma=0.01,coef0=0.0): 
    m=degree   
    sm=0                
    sm=np.sum(np.exp(-gamma*((x-y)**2)))
    return sm**m
#Rational quadratic a>0
def rquadratic(x,y, degree=3, gamma=0.01,coef0=0.1):
    a=coef0
    norm=la.norm(np.subtract(x, y))    
    return 1-(norm**2)/(norm**2+a)
#Canberra gamma E (0,1]
def canberra(x,y, degree=3, gamma=0.01,coef0=0.0):
    sm=0
    d=x.shape[0]    
    r=gamma*np.abs(x-y)/(np.abs(x)+np.abs(y))
    sm=np.sum(r[~np.isnan(r)]) 
    
    return 1-sm/d
#Truncated gamma>0
def truncated(x,y, degree=3, gamma=0.01,coef0=0.0):
    sm=0
    d=x.shape[0]    
    val=1-np.abs(x-y)/gamma
    sm=np.sum(val[val>0])       
    return sm/d

#Gram Matrix
def proxy_kernel(X, Y,K, degree=3, gamma=0.01,coef0=0.0):
    if K is None:
        K=mrbf            

    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i, j] = K(x,y, degree, gamma,coef0)
    return gram_matrix
#POLYNOMIAL
#degree=3, gamma=0.01,coef0=0.0
#------------------------------------
#Gram matrix Kernel 
#POLYNOMIAL
#degree=3, gamma=0.01,coef0=0.0
def polynomial_kernel(degree, gamma):    
    def pk(X,Y):                    
        return proxy_kernel(X,Y,K=polynomial,degree=degree,gamma=gamma)
    return pk
#MRBF
def mrbf_kernel(degree=2, gamma=0.01):    
    def mrbfk(X,Y):              
        return proxy_kernel(X,Y,K=mrbf,degree=degree,gamma=gamma)
    return mrbfk
#HIPERBOLIC TAN
def hyperbolic_kernel(gamma,coef0):    
    def hk(X,Y):             
        return proxy_kernel(X,Y,K=hyperbolic,gamma=gamma,coef0=coef0)
    return hk
#TRIANGLE
def triangle_kernel(gamma):    
    def tk(X,Y):            
        return proxy_kernel(X,Y,K=triangle,gamma=gamma)
    return tk
#ANOVA
def radial_basic_kernel(degree, gamma):     
    def rbk(X,Y):            
        return proxy_kernel(X,Y,K=radial_basic,degree=degree,gamma=gamma)
    return rbk
#RATIONAL QUADRATIC
def rquadratic_kernel(degree, gamma,coef0):     
    def rqk(X,Y):
        return proxy_kernel(X,Y,K=rquadratic,degree=degree,gamma=gamma,coef0=coef0)
    return rqk
#CANBERRA
def canberra_kernel(gamma):   
    def ck(X,Y):
        return proxy_kernel(X,Y,K=canberra,gamma=gamma)
    return ck
#TRUNCATED
def truncated_kernel(gamma):    
    def trk(X,Y):
        return proxy_kernel(X,Y,K=truncated,gamma=gamma)
    return trk

#---------------------------------------
#Callable Functions
def c_polynomial_kernel(degree, gamma):    
    def pk(X,Y):                    
        return polynomial(X,Y, degree=degree, gamma=gamma)
    return pk
#MRBF
def c_mrbf_kernel(degree=2, gamma=0.01):    
    def mrbfk(X,Y):              
        return mrbf(X,Y,degree=degree,gamma=gamma)
    return mrbfk
#HIPERBOLIC TAN
def c_hyperbolic_kernel(gamma,coef0):    
    def hk(X,Y):             
        return hyperbolic(X,Y,gamma=0.01,coef0=0.0)
    return hk
#TRIANGLE
def c_triangle_kernel(gamma):    
    def tk(X,Y):            
        return triangle(X,Y,gamma=gamma)
    return tk
#ANOVA
def c_radial_basic_kernel(degree, gamma):    
    def rbk(X,Y):            
        return radial_basic(X,Y, degree=degree, gamma=gamma)
    return rbk
#RATIONAL QUADRATIC
def c_rquadratic_kernel(degree, gamma):     
    def rqk(X,Y):
        return rquadratic(X,Y, degree=degree, gamma=gamma)
    return rqk
#CANBERRA
def c_canberra_kernel(gamma):    
    def ck(X,Y):
        return canberra(X,Y,gamma=gamma)
    return ck
#TRUNCATED
def c_truncated_kernel(gamma):    
    def trk(X,Y):
        return truncated(X,Y,gamma=gamma)
    return trk