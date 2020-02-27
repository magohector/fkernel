from config import Config
import os
import clasificator
import regressor

def mainmenu():    
    conf=Config()
    op=-1
    while(op!=0):
        os.system("cls")
        print("Contrast SVM-K\n")
        print("1.)Manage Kernel")
        print("2.)Write data base directory")
        print("3.)Write output directory")
        print("4.)Choose learning model")
        print("5.)Choose normalization model")
        print("6.)Run test")
        print("7.)Show configuration")
        op=int(input("Write option:"))
        if(op==1):
            manageKernelMenu(conf)
        if(op==2):
            manageDbDirectory(conf)
        if(op==3):
            manageOuputDirectory(conf)
        if(op==4):
            manageLearningModel(conf)            
        if(op==5):
            manageNormalizationModel(conf)    
        if(op==6):
            runTest(conf)    
        if(op==7):
            os.system("cls")
            print("Configuration\n")
            print(conf)  
            input()
        
def manageKernelMenu(conf):
    
    op=-1
    while(op!=0):
        os.system("cls")
        print("Manage Kernel\n")        
        print("1.)Remove Kernel")
        print("2.)Insert Kernel")
        print("3.)List Kernel")
        print("0.)Esc")
        op=int(input("Write option:"))
        if(op==1 or op==2):  
            menuChooseKernel(conf,op)
        elif(op==3):
            print(conf.kernels)
            input()
            


def menuChooseKernel(conf,o):
    kernels=["RBF","Tri","Rb","Rq","Can","Tru"]
    kernelsD=["Gaussian","Triangle","Radial Basic","Rational quadratic","Canberra","Truncated"]
    n=len(kernelsD)
    op=-1
    while op!=0:
        os.system("cls")
        print("Kernels")  
        os.system("cls")
        if(o==1):
            print("Removed Kernel\n")
        if(o==2):
            print("Append Kernel\n")
        for i, val in enumerate(kernelsD):            
            print(i+1,".)",val)           
        print(n+1,".) All")
        print(0,".) Esc")
        op=int(input("Write option:"))
        for i in range(1,n+3):
            if(op==i):
                if op>=1 and op<=n:
                    if(kernels[op-1] in conf.kernels):
                        if(o==1):
                            conf.kernels.remove(kernels[op-1])
                            print(kernelsD[op-1],"has removed")                            
                    else:
                        if(o==2):
                            conf.kernels.append(kernels[op-1])
                            print(kernelsD[op-1],"has appended")
                    input()
                elif op==n+1:
                    if(o==1):
                        conf.kernels.clear()
                        print("kernels have been cleared")
                    if(o==2):
                        conf.kernels=kernels
                        print("kernels have been full")
                    input()
                elif op==n+2:
                    break




def manageDbDirectory(conf):    
    os.system("cls")            
    d=input("Write data base directory:")
    mmkdir(d)
    if(d!=""):
         conf.dbd=d 
    print("Database directory assignment successful")
    input()

def manageOuputDirectory(conf):        
    os.system("cls")
    d=input("Write output directory:")
    mmkdir(d)
    if(d!=""):
         conf.od=d                           
    print("Output directory assignment successful")
    input()

def mmkdir(d):
    try:
        os.stat(d)                   
    except:
        if(d!=""):
            os.mkdir(d)            
        
        

def manageLearningModel(conf):
    op=-1
    while(op!=0):
        os.system("cls")
        print("Learning Model\n")
        print("1.)Classification")
        print("2.)Regression")
        print("0.)Esc")
        op=int(input("Write option:"))
        if(op==1):
            conf.lm="Classification"
        elif(op==2):
            conf.lm="Regression"
    input()

def manageNormalizationModel(conf):
    op=-1
    while(op!=0):
        os.system("cls")
        print("Normalization Model\n")
        print("1.)Min_Max")
        print("2.)Normalizer")
        print("3.)Standard")
        print("4.)None")
        print("0.)Esc")
        op=int(input("Write option:"))
        if(op==0):
            return
        if(op==1):
            conf.nm="Min_Max"
        elif(op==2):
            conf.nm="Normalizer"
        elif(op==3):
            conf.nm="Standard"
        else:
            conf.nm="None"
                   
    input()

def runTest(conf):
    if(conf.lm=="Classification"):
        clasificator.run(conf)    
    elif(conf.lm=="Regression"):
        regressor.run(conf)
    input()
    








