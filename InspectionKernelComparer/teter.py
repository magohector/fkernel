print(__doc__)

import pandas as pd
import numpy as np
import numpy.linalg as la
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
import kernel as krn
import evaluateKernel as ek2



df=pd.read_csv("Database/iris.csv",",")
nr=df.shape[1]
X=df.iloc[:,0:nr-1]
Y=df.iloc[:,nr-1:nr]
y = Y.values.ravel()
Xf,Xt,yf,yt=model_selection.train_test_split(X,Y.values.ravel(),test_size=0.25)  


print("rbf")
rbf_bp=ek2.getBestParameter(-4,5,1,'^',Xf,yf,Xt,yt,krn.mrbf_kernel,"clasification")
print(rbf_bp)
