class Config:    
    def __init__(self, kernels=["RBF","Tri","Rb","Rq","Can","Tru"],dbd="./Database", od="./Results",lm=0,nm=4):
        self.kernels=kernels
        self.dbd=dbd
        self.od=od        
        
        if(lm==0):
            self.lm="Classification"
        else:
            self.lm="Regression"
        if(nm==1):
            self.nm="Min_Max"
        elif(nm==2):
            self.nm="Normalizer"
        elif(nm==3):
            self.nm="Standard"
        else:
            self.nm="None"
        
    def __str__(self):
        return "kernels:"+str(self.kernels)+"\ndbd:'"+self.dbd+"'\nod:'"+self.od+"'\nlm:"+str(self.lm)+"\nnm:"+str(self.nm)