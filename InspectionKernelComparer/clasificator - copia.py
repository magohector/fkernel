#metrics https://scikit-learn.org/stable/modules/model_evaluation.html
import pandas as pd
import numpy as np
import numpy.linalg as la
from sklearn.svm import SVC
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
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


def run(config):
    print("Clasification SVM")
    mdir, subdirs, files = next(walk(config.dbd+"/Classification"))  
    tformat="%Y_%m_%d_%H_%M_%S"          
    tdirectory=config.od+"/Classification/test_"+time.strftime(tformat)    
    os.mkdir(tdirectory)
    f=open(tdirectory+"/test.txt","w")
    f.write("CLASIFICATION SVM")
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
        normalizators=["None","Min_Max","Normalizer","Standard"]
        for nm in normalizators:
            nr_o_dir=db_o_dir+"/"+nm
            os.mkdir(nr_o_dir)
            print("Normalization:",nm)
            print("ERROR")
            f.write("\nNormalization:"+nm)
            f.write("\nERROR:")
            Xn=normalizerTransform(X,config.nm)        
        
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
                    bp,bs=ek2.getBestParameter(-6,6,1,'^',Xf,yf,Xt,yt,lst_kernel[i],"classification")                
                else:
                    bp,bs=ek2.getBestParameter(-12,1,1,'^',Xf,yf,Xt,yt,krn.canberra_kernel,"classification")            
                
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
            scorings=['accuracy','precision_macro','recall_macro','f1_macro']        
            cm_dir=nr_o_dir+"/Confusion_Matrix" 
            cmun_dir=cm_dir+"/CMUN"
            cmn_dir=cm_dir+"/CMN"
            cv_dir=nr_o_dir+"/Cross_Validation"
            os.mkdir(cm_dir)
            os.mkdir(cmun_dir)
            os.mkdir(cmn_dir)
            os.mkdir(cv_dir)
            
            
            print("Cross validation\n") 
            f.write("\nCross validation\n")
            results=[]
            scoring_keys=None
            for i,mkernel in enumerate(lst_kernel):
                kname=lst_kernelnames[i]
                print(kname)
                f.write("\n"+kname)
                model=SVC(kernel=mkernel(val[i]), C=1e4,probability=True)        
                kfold = model_selection.StratifiedKFold(n_splits=5, random_state=None, shuffle=False)               
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
                    
                results.append(mmetric)
                  
                
                Yp=cross_val_predict(model,Xt,yt, cv=kfold)
                cm = confusion_matrix(yt, Yp)    
                classes = [str(i) for i in unique_labels(yt,Yp)]
                cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Confusion matrix")                
                print(cm)
                f.write("\nConfusion matrix")
                f.write("\n"+str(cm))
                
                print("Confusion matrix norm") 
                print(cmn)
                f.write("\nConfusion matrix norm")
                f.write("\n"+str(cmn))
                class_report=classification_report(yt, Yp, target_names=classes)
                print("Classification report")
                print(class_report)
                f.write("\nClasification report")
                f.write("\n"+class_report)
                plot_confusion_matrix(cm,classes,config,kname,cmun_dir+"/"+kname+".png",nfile+" "+kname)                             
                plot_confusion_matrix(cmn,classes,config,kname,cmn_dir+"/"+kname+".png",nfile+" "+kname)
            
            
            for i,sc in enumerate(scoring_keys):
                res=[]
                for j,kname in enumerate(lst_kernelnames):
                    res.append(results[j][i])                
                drawBoxplot(res,lst_kernelnames,cv_dir+"/"+sc+".png",sc)              
            
                 
       
        
            

def getkernels(lsk):
    lst_kernel=[krn.mrbf_kernel,krn.triangle_kernel,krn.radial_basic_kernel,krn.rquadratic_kernel,krn.canberra_kernel,krn.truncated_kernel]
    lst_kernelnames=["RBF","Tri","Rb","Rq","Can","Tru"]
    lst_k=[]
    for i,val in enumerate(lst_kernelnames):
        if val in lsk:
            lst_k.append(lst_kernel[i])
    return lst_k


def plot_confusion_matrix(cm,classes,config,kname,nfile,title,cmap=plt.cm.Blues):    
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(nfile,bbox_inches='tight')      
    plt.close() 

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

def drawError(nfile,kname,bp,bs,mbp,mbs,config):
    plt.xlabel('parameter') 
    plt.ylabel('score') 
    plt.title(kname+" error")
    plt.xscale('log')                 
    plt.plot(bp,bs,color="blue",)  
    plt.scatter(mbp,mbs,color="red")  
    plt.grid()
    plt.ylim(-0.1,1.1)
    plt.annotate(r'$('+str(mbp)+','+str(mbs)+')$',
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