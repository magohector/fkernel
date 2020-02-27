from evolutionary_search import EvolutionaryAlgorithmSearchCV
import sklearn.datasets
import numpy as np
import pandas as pd
from config import Config
from os import walk
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
import kernel as krn

def getkernels(lsk):
    lst_kernel=[krn.mrbf_kernel,krn.triangle_kernel,krn.radial_basic_kernel,krn.rquadratic_kernel,krn.canberra_kernel,krn.truncated_kernel]
    lst_kernelnames=["RBF","Tri","Rb","Rq","Can","Tru"]
    lst_k=[]
    for i,val in enumerate(lst_kernelnames):
        if val in lsk:
            lst_k.append(lst_kernel[i])
    return lst_k

df=pd.read_csv(".\Database\Classification\Iris.csv",";")

        
nr=df.shape[1]
X=df.iloc[:,0:nr-1]
Y=df.iloc[:,nr-1:nr]    
Y=Y.values.ravel()
print("load data")

paramgrid = {"kernel": krn.mrbf_kernel,
             "gamma": np.logspace(-9, 9, num=25, base=10),   
             "C"     : np.logspace(-9, 9, num=25, base=10),
             }


kfold = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)               
cv = GridSearchCV(estimator=SVC(),
                  param_grid=paramgrid,
                  scoring="accuracy",
                  cv=kfold,
                  verbose=1)
print("fit ini")
cv.fit(X, Y)
print("fit fin")

print("Best score ",cv.best_score_," Best param" ,cv.best_params_)

dfw=pd.DataFrame(cv.cv_results_).sort_values("mean_test_score", ascending=False)
dfw.to_csv(r'salida.csv')