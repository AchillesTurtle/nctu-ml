import numpy as np

def func(x):
    return 0.2*x**3+2*x**2-3*x+2

def matrix_transpose_t(x):
    t_x=np.zeros(x.shape)
    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            t_x[row,col]=x[col,row]
    return t_x

def data_gen(x_range=[-1,1],count=50):
    error=300
    x=np.random.rand(count,1)*abs(x_range[1]-x_range[0])+min(x_range)
    y=np.vectorize(func)(x)+(np.random.rand(count,1)-0.5)*error
    xaxis=np.arange(x_range[0],x_range[1],0.1)
    return x,y,xaxis

x,y,_=data_gen([-1,1],count=100)
np.savetxt('test',np.concatenate((x,y),axis=1),delimiter=',')

def matrix_transpose(x):
    return np.transpose(x)

def matrix_minor(x,row,col):
    return x[np.array(list(range(row))+list(range(row+1,x.shape[0])))[:,np.newaxis],
               np.array(list(range(col))+list(range(col+1,x.shape[1])))]
def matrix_det(x):
    #base case for 2x2 matrix
    if x.shape[0] == 2:
        return x[0,0]*x[1,1]-x[0,1]*x[1,0]
    det = sum( ((-1)**col)*x[0,col]*matrix_det(matrix_minor(x,0,col)) for col in range(x.shape[0]))
    return det

def matrix_inverse(x):
    #DOESN'tWORK!!

    det = matrix_det(x)
    #special case for 2x2 matrix:
    if x.shape[0] == 2:
        return np.array([[x[1,1],-x[0,1]],[-x[1,0],x[0,0]]])/det

    #find matrix of cofactors
    cofactors = np.zeros([x.shape[0],x.shape[0]])
    for row in range(x.shape[0]):
        for col in range(x.shape[0]):
            minor = matrix_minor(x,row,col)
            cofactors[row,col]=(((-1)**(row+col)) * matrix_det(minor))
    cofactors = matrix_transpose(cofactors)/det
    return cofactors

x=np.random.randint(-20,20,size=(5,5))

#x=np.identity(3)
print(x.T)
print(matrix_transpose_t(x))
