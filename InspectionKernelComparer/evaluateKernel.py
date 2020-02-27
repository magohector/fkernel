import numpy as np
import numpy.linalg as la
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import kernel as krn
np.seterr(divide='ignore', invalid='ignore')

def getBestParameter(li,ls,step,tstep,Xf,Yf,Xt,Yt,kernel,problemtype):    
    isfirst=True
    bs=[]
    bp=[]
    
    for p in frange(li,ls,step,tstep):        
        if problemtype=="prediction":
            svr=SVR(kernel=kernel(p), C=1e4)
            svr.fit(Xf, Yf)
            score=svr.score(Xt,Yt)
            bp.append(p)
            bs.append(score)
        elif problemtype=="classification":
            svc=SVC(kernel=kernel(p), C=1e4)
            svc.fit(Xf, Yf)            
            score=svc.score(Xt,Yt)                            
            bp.append(p)
            bs.append(score)            

    return bp,bs    


def frange(start, stop, step,tstep):
    i=start
    while i < stop:        
        if tstep=='+':
            yield i
            i += step
        if tstep=='*':
            yield i
            i*=step
        if tstep=='^':
            yield 10**i
            i+=step