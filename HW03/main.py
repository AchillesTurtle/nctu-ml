import numpy as np
from ops import matrix_mul as matmul
from ops import matrix_transpose
from ops import amax
#from numpy.linalg import inv
from ops import matrix_inverse as inv
from generators import UniGaussianGen,PolyBasisGen
import matplotlib.pyplot as plt

class Datasummary:
    def __init__(self):
        self.mean=0
        self.var=0
        self.pvar=0
        self.data_count=0
    def update(self,new_data):
        new_mean=(self.mean*self.data_count+new_data)/(self.data_count+1)
        if self.data_count>0:
            self.var=(self.data_count-1)/self.data_count*self.var+(new_data-self.mean)**2/(self.data_count+1)
            self.pvar=self.pvar+((new_data-self.mean)*(new_data-new_mean)-self.pvar)/(self.data_count+1)
        self.mean=new_mean
        self.data_count+=1
        return self.mean,self.var,self.pvar
    def update_to_converge(self,gen_func):
        #only accepts numbers!
        new_summary=(self.mean,self.var,self.pvar)
        while True:
            old_summary=new_summary
            new_data=gen_func()
            new_summary=self.update(new_data)
            if(self.data_count%1000==0):
                print("Iteration:{}\nnew_data:{}".format(self.data_count,new_data))
                print("(mean,var,var):{}".format(new_summary))
            if self.data_count>10:
                if(sum([abs(new_summary[i]-old_summary[i]) for i in range(3)])<1e-5):
                    print("Iteration:{}\nnew_data:{}".format(self.data_count, new_data))
                    print("(mean,var,var):{}".format(new_summary))
                    return self.mean,self.var,self.pvar
#phi is horizontal
class Bayesianlinear:
    def __init__(self):
        #predefined?
        self.data_count=0
        self.noise_var=10
    def fit(self,b,new_data,a,basis_num=5):
        self.basis_num=basis_num
        self.phi = np.zeros((1, self.basis_num))
        for i in range(self.basis_num):
            self.phi[0,i]=new_data[0]**i
        self.precision=self.noise_var*matmul(matrix_transpose(self.phi),self.phi)+b*np.identity(self.basis_num)
        self.mean=self.noise_var*matmul(inv(self.precision),matrix_transpose(self.phi))*new_data[1]
        self.noise_var=1/a
        self.data_count+=1
        #print(self.precision,self.mean)
    def update(self,new_data):
        for i in range(self.basis_num):
            self.phi[0,i]=new_data[0]**i
        S=self.precision
        self.precision=self.noise_var*matmul(matrix_transpose(self.phi),self.phi)+S
        self.mean=matmul(inv(self.precision),(self.noise_var*matrix_transpose(self.phi)*new_data[1]+matmul(S,self.mean)))
        self.p_var=(1/self.noise_var)+matmul(matmul(self.phi,inv(self.precision)),matrix_transpose(self.phi))
        self.p_mean=matmul(matrix_transpose(self.mean),matrix_transpose(self.phi))
        self.data_count+=1
        #print(new_data,self.p_var,self.p_mean)
        #print(self.precision,self.mean,self.p_var,self.p_mean)
        return self.precision,self.mean,self.p_var,self.p_mean
    def update_to_converge(self,gen_func):
        #only accepts numbers!
        new_summary=[self.precision,self.mean]
        while True:
            old_summary=new_summary
            new_data=gen_func()
            temp=self.update(new_data)
            new_summary=[temp[0],temp[1]]
            if(self.data_count%1000==0):
                print("Iteration:{}".format(self.data_count))
                print("Posterior (precision,mean) :{}".format(temp[0], temp[1]))
                print("Predictive Distribution(var,mean):{},{}".format(temp[2], temp[3]))
                print("New data point(x,y):{}".format(list(new_data)))
            if self.data_count>10:
                if(amax(inv(temp[0]))<1e-3):
                    print("Iteration:{}".format(self.data_count))
                    print("Posterior (precision,mean) :{}\n{}".format(temp[0],temp[1]))
                    print("Predictive Distribution(var,mean):{},{}".format(temp[2],temp[3]))
                    print("New data point(x,y):{}".format(list(new_data)))
                    return self.precision, self.mean, self.p_var, self.p_mean
if __name__=="__main__":
    gauss=Datasummary()
    gauss.update_to_converge(lambda:UniGaussianGen(10,5))
    #print(gauss.data_count)
    w=[5,0.2,0.3,0.02]
    #x=np.zeros(100)
    #y=np.zeros(100)
    #for i in range(100):
    #    (x[i],y[i])=PolyBasisGen(4,0.5,w)
    #plt.scatter(x,y)
    #plt.show()

    bayes=Bayesianlinear()
    b=1
    a=1
    bayes.fit(b,PolyBasisGen(4, a, w),a)
    bayes.update_to_converge(lambda:PolyBasisGen(4,a,w))
    #for i in range(1000):
    #    print("--------{}---------".format(i))
    #    bayes.update(PolyBasisGen(4, 5, w))