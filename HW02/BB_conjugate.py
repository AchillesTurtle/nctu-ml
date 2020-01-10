from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
from math import factorial

def read_data(file_path='output'):
    lol = [[]]
    with open(file_path,'r') as fn:
        line=0
        for bin_char in fn.read():
            if bin_char=='0' or bin_char=='1':
                lol[line].append(int(bin_char))
            elif  bin_char=='\n':
                line+=1
                lol.append(list())
            else:
                print("File unexpected digit: {}".format(bin_char))
    return lol

def BBP(prior_a,prior_b,path=None):
    lol=read_data()
    #x = np.linspace(beta.ppf(0.01, prior_a, prior_b), beta.ppf(0.99, prior_a, prior_b), 100)
    #rv = beta(prior_a, prior_b)
    #plt.plot(x, rv.pdf(x))
    #plt.show()
    for line in lol:
        x = np.linspace(beta.ppf(0.01, prior_a, prior_b),beta.ppf(0.99, prior_a, prior_b), 100)
        rv=beta(prior_a,prior_b)
        plt.plot(x, rv.pdf(x))


        binolike=sum(line)/len(line)
        print(binolike)
        print(prior_a,prior_b)
        prior_a=prior_a+sum(line)
        prior_b=prior_b+len(line)-sum(line)
        print(prior_a,prior_b)
        print("-----")
    plt.show()
if __name__ == "__main__":
    #path=input("File path:")
    a=input("Value of a:")
    b=input("Value of b:")
    BBP(int(a),int(b))