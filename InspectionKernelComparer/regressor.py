#metrics https://scikit-learn.org/stable/modules/model_evaluation.html
import pandas as pd
import numpy as np
import numpy.linalg as la
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
import kernel as krn
import evaluateKernel as ek2
from config import Config
from os import walk
import os
import time
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler



def run(config):
    print("Regressor SVM")
    mdir, subdirs, files = next(walk(config.dbd+"/Regression"))  
    tformat="%Y_%m_%d_%H_%M_%S"          
    tdirectory=config.od+"/Regression/test_"+time.strftime(tformat)    
    os.mkdir(tdirectory)
    f=open(tdirectory+"/test.txt","w")
    f.write("REGRESSION SVM")
    myexport=[]
    for mfile in files:
       
        nfile=os.path.splitext(mfile)[0]            
        print(mdir+"/"+mfile)
        f.write("\n"+mfile+"\n")

        db_o_dir=tdirectory+"/"+nfile
        os.mkdir(db_o_dir)

        df=pd.read_csv(mdir+"/"+mfile,";")        
        nr=df.shape[1]
        X=df.iloc[:,0:nr-1]
        Y=df.iloc[:,nr-1:nr]    
        Y=Y.values.ravel()

        Xn=X
        Yn=Y
        normalizators=["None","Min_Max","Normalizer","Standard"]
        for nm in normalizators:
            nr_o_dir=db_o_dir+"/"+nm
            os.mkdir(nr_o_dir)
            print("Normalization:",nm)
            print("ERROR")
            f.write("\nNormalization:"+nm)
            f.write("\nERROR:")
            Xn=normalizerTransform(X,config.nm)
            Yn=normalizerTransform(Yn,config.nm)        
        
            Xf,Xt,yf,yt=model_selection.train_test_split(Xn,Y,test_size=0.25)          
            
            lst_kernelnames=config.kernels
            lst_kernel=getkernels(lst_kernelnames)  
            val=[]
            #class_names=np.array([str(i) for i in list(set(yf))])
            error_dir=nr_o_dir+"/Error"
            os.mkdir(error_dir)
            for i,kname in enumerate(lst_kernelnames):
                print(kname) 
                f.write("\n"+kname) 
                bp,bs=0,0                 
                if(kname!="Can"):
                    bp,bs=ek2.getBestParameter(-6,6,1,'^',Xf,yf,Xt,yt,lst_kernel[i],"prediction")                
                else:
                    bp,bs=ek2.getBestParameter(-12,1,1,'^',Xf,yf,Xt,yt,krn.canberra_kernel,"prediction")            
                
                mbs=max(bs)                
                mbp=bp[bs.index(mbs)]
                print(bp)
                print(bs)
                print("\n(",mbp,",",mbs,")")
                f.write("\n"+str(bp))
                f.write("\n"+str(bs))
                f.write("\n("+str(mbp)+","+str(mbs)+")")
                val.append(mbs)   

                drawError(error_dir+"/"+kname+".png",kname,bp,bs,mbp,mbs,config)
                            

            #acuracy sum(y_true==y_pred)/len(y_true)
            #recall sum(np.logical_and(y_true==a,y_pred==a))/sum(y_true==a)
            #precision sum(np.logical_and(y_true==a,y_pred==a))/sum(y_pred==a)
            scorings=['explained_variance','neg_mean_absolute_error','neg_mean_squared_error','neg_median_absolute_error','r2']        
            sm_dir=nr_o_dir+"/Scatter"             
            cv_dir=nr_o_dir+"/Cross_Validation"
            os.mkdir(sm_dir)            
            os.mkdir(cv_dir)
            
            
            print("Cross validation\n") 
            f.write("\nCross validation\n")
            results=[]
            scoring_keys=None
            for i,mkernel in enumerate(lst_kernel):
                kname=lst_kernelnames[i]
                print(kname)
                f.write("\n"+kname)
                model=SVR(kernel=mkernel(val[i]), C=1e4) 
                
                kfold = model_selection.KFold(n_splits=5, random_state=None, shuffle=False)               
                kname=lst_kernelnames[i]               
                cvr=cross_validate(model,X,Y,cv=kfold, scoring=scorings, return_train_score=True)
                mmetric=[]
                for key in cvr.keys():  
                    if scoring_keys==None:
                        scoring_keys=cvr.keys()          
                    cvrmean=cvr[key].mean()
                    stdmean=cvr[key].std()                        
                    msg = "%s: %f (±%f) [%f,%f]" % (key,cvrmean,stdmean,cvrmean-stdmean,cvrmean+stdmean)                    
                    
                    print("\n\n"+msg)                                        
                    print(cvr[key])  
                    f.write("\n\n"+msg)  
                    f.write("\n"+str(cvr[key])) 
                    cvs=np.array(cvr[key])
                    cvs=np.abs(cvs)        
                    cvs=sorted(cvs)
                    mmetric.append(cvs)
                    mydict={'db':mfile,'normalizer':nm,'kernel':kname,'metric':key,'avg':cvr[key].mean(),'std':cvr[key].std()}     
                    myexport.append(mydict)                
                    
                results.append(mmetric)
                  
                
                Yp=cross_val_predict(model,Xf,yf, cv=kfold) 
                drawPrediction(Yp,yf,sm_dir+"/pvt_"+kname+".png",kname)           
            
            for i,sc in enumerate(scoring_keys):
                res=[]
                for j,kname in enumerate(lst_kernelnames):
                    res.append(results[j][i])                
                drawBoxplot(res,lst_kernelnames,cv_dir+"/"+sc+".png",sc)
    df = pd.DataFrame(myexport)  
    df.to_csv(tdirectory+"/test.csv")                 
            

def getkernels(lsk):
    lst_kernel=[krn.mrbf_kernel,krn.triangle_kernel,krn.radial_basic_kernel,krn.rquadratic_kernel,krn.canberra_kernel,krn.truncated_kernel]
    lst_kernelnames=["RBF","Tri","Rb","Rq","Can","Tru"]
    lst_k=[]
    for i,val in enumerate(lst_kernelnames):
        if val in lsk:
            lst_k.append(lst_kernel[i])
    return lst_k



def normalizerTransform(x,nm):       
    if(nm=="Min_Max"):#mix-max
        scaler=MinMaxScaler()
        scaler.fit(x)
        return scaler.transform(x)
    if(nm=="Normalizer"):#normalizer
        scaler=Normalizer()
        scaler.fit(x)
        return scaler.transform(x)
    
    if(nm=="Standard"):#standard
        scaler=StandardScaler()
        scaler.fit(x)
        return scaler.transform(x)    
    return x

def drawPrediction(yp,yt,nfile,title):
    plt.xlabel('Y true') 
    plt.ylabel('Y predicted')    
    plt.title(title+" et=0.1")
    ytmin=min(yt)
    ytmax=max(yt)     
    p=(ytmax-ytmin)/10    
    y_t=np.arange(ytmin-p,ytmax+p,p)               
    plt.scatter(yt,yp)
    plt.plot(y_t,y_t,color="green",ls="-")
    plt.plot(y_t,y_t+1.1*ytmin,color="red",ls="--")
    plt.plot(y_t,y_t-1.1*ytmin,color="red",ls="--")
       
    plt.grid()
    plt.axis("equal")
    plt.savefig(nfile,bbox_inches='tight')   
    plt.close() 

def drawError(nfile,kname,bp,bs,mbp,mbs,config):
    plt.xlabel('parameter') 
    plt.ylabel('score') 
    plt.title(kname+" error")
    plt.xscale('log')                 
    plt.plot(bp,bs,color="blue",)  
    plt.scatter(mbp,mbs,color="red")  
    plt.grid()
    plt.ylim(-0.1,1.1)
    plt.annotate(r'$('+str(mbp)+','+str("%.2f"%mbs)+')$',
    xy=(mbp, mbs), xycoords='data',
    xytext=(mbp,mbs*0.8), fontsize=9,            
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.savefig(nfile,bbox_inches='tight')      
    plt.close()

def drawBoxplot(results,lst_kernelnames,nfile,scoring):
    plt.boxplot(results, labels=lst_kernelnames)
    plt.title(scoring, fontsize=10)    
    plt.savefig(nfile,bbox_inches='tight')      
    plt.close() 