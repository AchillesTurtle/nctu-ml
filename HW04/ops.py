import numpy as np
def amax(x):
    x=x.flatten()
    max=x[0]
    for i in range(x.size):
        if x[i]>max:
            max=x[i]
    return max

def matrix_transpose(x):
    t_x=np.zeros((x.shape[1],x.shape[0]))
    for row in range(x.shape[1]):
        for col in range(x.shape[0]):
            t_x[row,col]=x[col,row]
    return t_x

def L_inv(L):
    X=np.zeros((L.shape[0],L.shape[0]))
    for r in range(L.shape[0]):
        X[r,r]=1/L[r,r]
        for c in range(r):
            X[r,c]=sum(-L[r,0:r]*X[0:r,c])/L[r,r]
    return X
def U_inv(U):
    X=np.zeros((U.shape[0],U.shape[0]))
    for r in range(U.shape[0]):
        X[r,r]=1/U[r,r]
        for c in range(r+1,U.shape[0]):
            X[c, c] = 1 / U[c, c]
            X[r,c]=sum(-U[0:c,c]*X[r,0:c])/U[c,c]
    return X

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

def matrix_inverse(x):
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
    L[abs(L)<1e-15]=0
    U[abs(U)<1e-15]=0
    return matrix_mul(U_inv(U),L_inv(L))
