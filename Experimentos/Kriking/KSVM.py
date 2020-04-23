from KernelUtilities import *
from sklearn.svm import SVC,SVR
from sklearn.model_selection import train_test_split
#Clasificator
class KSVC(SVC):
    
    def __init__(self, C=1.0, kernel='rbf', degree=2, gamma='auto_deprecated',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 random_state=None,a=2):       

        super().__init__(
        kernel=kernel, degree=degree, gamma=gamma,
        coef0=coef0, tol=tol, C=C, shrinking=shrinking,
        probability=probability, cache_size=cache_size,
        class_weight=class_weight, verbose=verbose, max_iter=max_iter,
        decision_function_shape=decision_function_shape,
        random_state=random_state)
        
        self.a=a
        
    def fit(self, X, y, sample_weight=None):
        
        if(self.kernel=="linear" or self.kernel== "poly" or self.kernel== "rbf"):
            super().fit(X,y)
            return self
        else:            
            if(self.kernel=="mrbf"):                  
                self.kernel=mrbf_kernel(degree=self.degree, gamma=self.gamma)                   
                super().fit(X,y)
                return self
            if(self.kernel=="hyperbolic"):                
                self.kernel=hyperbolic_kernel(self.gamma,self.coef0)                
                super().fit(X,y)
                return self 
            if(self.kernel=="triangle"):                
                self.kernel=triangle_kernel(self.gamma)                
                super().fit(X,y)
                return self 
            if(self.kernel=="radial_basic"):                
                self.kernel=radial_basic_kernel(self.degree, self.gamma)                
                super().fit(X,y)
                return self            
            if(self.kernel=="rquadratic"):                
                self.kernel=rquadratic_kernel(self.degree, self.gamma, self.coef0)                
                super().fit(X,y)
                return self
            if(self.kernel=="can"):                
                self.kernel=canberra_kernel(self.gamma)                
                super().fit(X,y)
                return self 
            if(self.kernel=="tru"):                
                self.kernel=truncated_kernel(self.gamma)                
                super().fit(X,y)
                return self

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
#Regressor
class KSVR(SVR):
    
    def __init__(self, kernel='rbf', degree=2, gamma='scale',
                 coef0=0.0, tol=1e-3, C=1.0, epsilon=0.1, shrinking=True,
                 cache_size=200, verbose=False, max_iter=-1):

        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma,
            coef0=coef0, tol=tol, C=C, epsilon=epsilon, verbose=verbose,
            shrinking=shrinking, cache_size=cache_size,
            max_iter=max_iter)
        
    def fit(self, X, y, sample_weight=None):
        
        if(self.kernel=="linear" or self.kernel== "poly" or self.kernel== "rbf"):
            super().fit(X,y)
            return self
        else:            
            if(self.kernel=="mrbf"):                  
                self.kernel=mrbf_kernel(degree=self.degree, gamma=self.gamma)                   
                super().fit(X,y)
                return self
            if(self.kernel=="hyperbolic"):                
                self.kernel=hyperbolic_kernel(self.gamma,self.coef0)                
                super().fit(X,y)
                return self 
            if(self.kernel=="triangle"):                
                self.kernel=triangle_kernel(self.gamma)                
                super().fit(X,y)
                return self 
            if(self.kernel=="radial_basic"):                
                self.kernel=radial_basic_kernel(self.degree, self.gamma)                
                super().fit(X,y)
                return self            
            if(self.kernel=="rquadratic"):                
                self.kernel=rquadratic_kernel(self.degree, self.gamma, self.coef0)                
                super().fit(X,y)
                return self
            if(self.kernel=="can"):                
                self.kernel=canberra_kernel(self.gamma)                
                super().fit(X,y)
                return self 
            if(self.kernel=="tru"):                
                self.kernel=truncated_kernel(self.gamma)                
                super().fit(X,y)
                return self

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self