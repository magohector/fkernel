from KernelUtilities import *
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.kernel_approximation import Nystroem
#Clasificator
class KANNC(MLPClassifier):
    
    def __init__(self, hidden_layer_sizes=(100,), activation="identity",
                 solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate="constant",
                 learning_rate_init=0.001, power_t=0.5, max_iter=200,
                 shuffle=True, random_state=None, tol=1e-4,
                 verbose=False, warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=True,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, n_iter_no_change=10, max_fun=15000,kernel='rbf', degree=2, gamma=0.1,
                     coef0=0.0):
        
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation, solver=solver, alpha=alpha,
            batch_size=batch_size, learning_rate=learning_rate,
            learning_rate_init=learning_rate_init, power_t=power_t,
            max_iter=max_iter, shuffle=shuffle,
            random_state=random_state, tol=tol, verbose=verbose,
            warm_start=warm_start, momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
            n_iter_no_change=n_iter_no_change)  


        self.kernel=kernel
        self.gamma=gamma
        self.degree=degree
        self.coef0=coef0   
     
        
    def fit(self, X, y):
        self.feature_map_nystroem=None  
        self.isfit=False
        if(self.kernel=="linear"):
            super().fit(X,y)
            return self
        else:   
            
            if(self.kernel=="poly"):  
                self.feature_map_nystroem = Nystroem(kernel=c_polynomial_kernel(degree=self.degree, gamma=self.gamma))                                  
                
            if(self.kernel=="rbf"):                  
                self.feature_map_nystroem = Nystroem(kernel=c_mrbf_kernel(degree=self.degree, gamma=self.gamma)) 
            if(self.kernel=="hyperbolic"):                
                self.feature_map_nystroem = Nystroem(kernel=c_hyperbolic_kernel(gamma=self.gamma,coef0=self.coef0)) 
            if(self.kernel=="triangle"):                
                self.feature_map_nystroem = Nystroem(kernel=c_triangle_kernel(gamma=self.gamma)) 
            if(self.kernel=="radial_basic"):                
                self.feature_map_nystroem = Nystroem(kernel=c_radial_basic_kernel(degree=self.degree, gamma=self.gamma))            
            if(self.kernel=="rquadratic"):                
                self.feature_map_nystroem = Nystroem(kernel=c_rquadratic_kernel(degree=self.degree, gamma=self.gamma)) 
            if(self.kernel=="can"):                
                self.feature_map_nystroem = Nystroem(kernel=c_canberra_kernel(gamma=self.gamma))  
            if(self.kernel=="tru"):                
                self.feature_map_nystroem = Nystroem(kernel=c_truncated_kernel(gamma=self.gamma))                 
            
            if not self.feature_map_nystroem is None:
                data_transformed = self.feature_map_nystroem.fit_transform(X)
                super().fit(data_transformed,y)    
                self.isfit=True
                return self
            
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    def predict(self, X):  
        if self.isfit:
            X=self.mtransform(X)            
        return super().predict(X)
    def mtransform(self,X):
        if not self.feature_map_nystroem is None and self.isfit:
            return self.feature_map_nystroem.transform(X)
        return X

#Regresor
class KANNR(MLPRegressor):     
    
    def __init__(self, hidden_layer_sizes=(100,), activation="identity",
                 solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate="constant",
                 learning_rate_init=0.001,
                 power_t=0.5, max_iter=200, shuffle=True,
                 random_state=None, tol=1e-4,
                 verbose=False, warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, n_iter_no_change=10, max_fun=15000,kernel='rbf', degree=2, gamma=0.1,
                     coef0=0.0):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation, solver=solver, alpha=alpha,
            batch_size=batch_size, learning_rate=learning_rate,
            learning_rate_init=learning_rate_init, power_t=power_t,
            max_iter=max_iter, shuffle=shuffle,
            random_state=random_state, tol=tol, verbose=verbose,
            warm_start=warm_start, momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
            n_iter_no_change=n_iter_no_change)  


        self.kernel=kernel
        self.gamma=gamma
        self.degree=degree
        self.coef0=coef0   
     
        
    def fit(self, X, y):
        self.feature_map_nystroem=None  
        self.isfit=False
        if(self.kernel=="linear"):
            super().fit(X,y)
            return self
        else:   
            
            if(self.kernel=="poly"):  
                self.feature_map_nystroem = Nystroem(kernel=c_polynomial_kernel(degree=self.degree, gamma=self.gamma))                                  
                
            if(self.kernel=="rbf"):                  
                self.feature_map_nystroem = Nystroem(kernel=c_mrbf_kernel(degree=self.degree, gamma=self.gamma)) 
            if(self.kernel=="hyperbolic"):                
                self.feature_map_nystroem = Nystroem(kernel=c_hyperbolic_kernel(gamma=self.gamma,coef0=self.coef0)) 
            if(self.kernel=="triangle"):                
                self.feature_map_nystroem = Nystroem(kernel=c_triangle_kernel(gamma=self.gamma)) 
            if(self.kernel=="radial_basic"):                
                self.feature_map_nystroem = Nystroem(kernel=c_radial_basic_kernel(degree=self.degree, gamma=self.gamma))            
            if(self.kernel=="rquadratic"):                
                self.feature_map_nystroem = Nystroem(kernel=c_rquadratic_kernel(degree=self.degree, gamma=self.gamma)) 
            if(self.kernel=="can"):                
                self.feature_map_nystroem = Nystroem(kernel=c_canberra_kernel(gamma=self.gamma))  
            if(self.kernel=="tru"):                
                self.feature_map_nystroem = Nystroem(kernel=c_truncated_kernel(gamma=self.gamma))                 
            
            if not self.feature_map_nystroem is None:
                data_transformed = self.feature_map_nystroem.fit_transform(X)
                super().fit(data_transformed,y)    
                self.isfit=True
                return self
            
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    def predict(self, X):  
        if self.isfit:
            X=self.mtransform(X)            
        return super().predict(X)
    def mtransform(self,X):
        if not self.feature_map_nystroem is None and self.isfit:
            X=self.feature_map_nystroem.transform(X)            
            return X
        return X