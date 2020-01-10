#python 3.5
import numpy as np
import datetime
import matplotlib.pyplot as plt

def import_data(file_name):
    data=np.loadtxt(file_name, delimiter=',')
    return data[:,0][:,np.newaxis],data[:,1][:,np.newaxis]

def matrix_transpose(x):
    t_x=np.zeros((x.shape[1],x.shape[0]))
    for row in range(x.shape[1]):
        for col in range(x.shape[0]):
            t_x[row,col]=x[col,row]
    return t_x

def matrix_minor(x,row,col):
    #returns matrix withr (row,col) deleted
    return x[np.array(list(range(row))+list(range(row+1,x.shape[0])))[:,np.newaxis],
               np.array(list(range(col))+list(range(col+1,x.shape[1])))]
def matrix_det(x):
    #base case for 2x2 matrix
    if x.shape[0] == 2:
        return x[0,0]*x[1,1]-x[0,1]*x[1,0]
    det = sum( ((-1)**col)*x[0,col]*matrix_det(matrix_minor(x,0,col)) for col in range(x.shape[0]))
    return det

def triangular_det(x):
    #print(np.prod(list(x[i,i] for i in range(x.shape[0]))),np.linalg.det(x))
    return np.prod(list(x[i,i] for i in range(x.shape[0])))

def design_matrix(x,bases):
    #form design matrix
    d_mat=np.ones((x.shape[0],bases))
    for row in range(x.shape[0]):
        for col in range(bases):
            d_mat[row,col]=pow(x[row],bases-col-1)
    return d_mat

def matrix_mul(x,y):
    #perform matrix multiplication x*y

    if x.shape[1]==y.shape[0]:
        result = np.zeros([x.shape[0],y.shape[1]])
        for row in range(x.shape[0]):
            for col in range(y.shape[1]):
                result[row,col]=sum(x[row,:]*y[:,col])
                pass
        return result
    else:
        print(x.shape)
        print(y.shape)
        raise

def LU_decomp(x):
    #perform LU decomposition
    if x.shape[0] is not x.shape[1]:
        raise
    P = np.identity(x.shape[0])
    #in case for singularity problems
    if x[0,0] == 0:
        for row in range(x.shape[0]):
            if x[row,0] != 0:
                P[0,0]=0.
                P[0,row]=1.
                P[row,row]=0.
                P[row,0]=1.
                break

    Px=matrix_mul(P,x)

    L = np.zeros((x.shape[0],x.shape[0]))
    U = np.zeros((x.shape[0], x.shape[0]))
    for row in range(x.shape[0]):
        L[row,row]=1
        for Lcol in range(row):
            L[row,Lcol]=(Px[row,Lcol]-sum(L[row,i]*U[i,Lcol] for i in range(Lcol)))/U[Lcol,Lcol]

        for Ucol in range(x.shape[0]):
            U[row,Ucol]=Px[row,Ucol]-sum(L[row,i]*U[i,Ucol]for i in range(row))
    #a filter to filter out small error values
    L[abs(L)<1e-12]=0
    U[abs(U)<1e-12]=0
    return P,L,U

def matrix_inverse(x):
    #the original inverse
    #inverse
    det = matrix_det(x)
    #base 2x2 matrix:
    if x.shape[0] == 2:
        return np.array([[x[1,1],-x[0,1]],[-x[1,0],x[0,0]]])/det

    #cofactors
    cofactors = np.zeros([x.shape[0],x.shape[0]])
    for row in range(x.shape[0]):
        for col in range(x.shape[0]):
            minor = matrix_minor(x,row,col)
            cofactors[row,col]=(((-1)**(row+col)) * matrix_det(minor))
    cofactors = matrix_transpose(cofactors)/det
    return cofactors

def triangular_inverse(L,ul):
    L_det = triangular_det(L)
    #base 2x2 matrix:
    if L.shape[0] == 2:
        return np.array([[L[1,1],-L[0,1]],[-L[1,0],L[0,0]]])/L_det
    #cofactors
    cofactors = np.zeros([L.shape[0],L.shape[0]])
    for row in range(L.shape[0]):
        for col in range(L.shape[0]):
            minor = matrix_minor(L,row,col)
            if col==L.shape[0]-1 or row==L.shape[0]-1 or col==0 or row==0:
                #avoids the case of non-triangular matrixs
                cofactors[row, col] = (((-1) ** (row + col)) * matrix_det(minor))
            else:
                cofactors[row, col] = (((-1) ** (row + col)) * triangular_det(minor))
    inverse=matrix_transpose(cofactors)/L_det
    return inverse

def LU_matrix_inverse(x):
    #inverse by doing LU
    P,L,U=LU_decomp(x)
    U_inv = triangular_inverse(U,'U')
    L_inv = triangular_inverse(L,'L')
    return matrix_mul(matrix_mul(U_inv,L_inv),P)

def newton_opt(w,A,b):
    #newton optimizer
    gradientf=2*matrix_mul(matrix_mul(matrix_transpose(A),A),w)-2*matrix_mul(matrix_transpose(A),b)
    hessianf=2*matrix_mul(matrix_transpose(A),A)
    deltaw=matrix_mul(matrix_inverse(hessianf),gradientf)
    new_w=w-deltaw
    if sum(np.abs(deltaw))>1e-01:
        new_w=newton_opt(new_w,A,b)
    return new_w

def LSE_opt(A,b,lam):
    #LSE optimizer
    ATA=LU_matrix_inverse(matrix_mul(matrix_transpose(A),A)-lam*np.identity(A.shape[1]))
    weights=matrix_mul(matrix_mul(ATA,matrix_transpose(A)),b)
    return weights

def func(x):
    #function ofr data_gen
    return x**3+10*x**2-3*x+2

def data_gen(x_range=[-10,10],count=50):
    #sample data generator
    error=300
    x=np.random.rand(count,1)*abs(x_range[1]-x_range[0])+min(x_range)
    y=np.vectorize(func)(x)+(np.random.rand(count,1)-0.5)*error
    xaxis=np.arange(x_range[0],x_range[1],0.1)
    return x,y,xaxis

def print_func(x):
    #format print function
    for i,coef in enumerate(x):
        if i==x.shape[0]-1:
            print("{0:0.2f} = y".format(float(coef)))
        else:
            print("{0:0.2f} * x^{1} + ".format(float(coef),x.shape[0]-i-1),end='')

#input prompt
file_name=input("input data file path:")
bases=int(input("number of bases:"))
lam=int(input("lambda:"))

#file_name="test"
#bases=3
#lam=0

#datax,datay,xaxis=data_gen()
datax,datay=import_data('test')

"""
x=np.random.rand(10,10)
time1=datetime.datetime.now()
print(matrix_inverse(x))
time2=datetime.datetime.now()
print(LU_matrix_inverse(x))
time3=datetime.datetime.now()
print(time2-time1,time3-time2)
"""

#to draw lines
xaxis=np.arange(min(datax),max(datax),0.1)
A=design_matrix(datax,bases)
#get weights
weight_lse=LSE_opt(A,datay,lam)
weight_newton=newton_opt(np.zeros((bases,1)),A,datay)
#get errors
error_lse=sum((np.polyval(weight_lse,datax)-datay)**2)
error_newton=sum((np.polyval(weight_newton,datax)-datay)**2)
#plot
plt.plot(datax,datay,'ro',xaxis,np.polyval(weight_lse,xaxis),'b',linewidth=0.5)
plt.plot(xaxis,np.polyval(weight_newton,xaxis),'r--',linewidth=0.8)
print("LSE:")
print_func(weight_lse)
print("error: {0}".format(error_lse))
print("Newton:")
print_func(weight_newton)
print("error: {0}".format(error_newton))
print("\nError diff: {0}".format(error_lse-error_newton))

plt.show()

