from generators import MulGaussianGen
import numpy as np
from matplotlib import pyplot
from ops import matrix_transpose,matrix_inverse
from ops import matrix_mul as matmul
from math import exp
import numpy.linalg as linalg
import sys

D1x,D1y=MulGaussianGen(0.2,0.2,0.3,0.1)
D2x,D2y=MulGaussianGen(1,0.03,0.7,0.2)
x=np.reshape(np.asarray(D1x+D2x),(-1,1))
y=np.reshape(np.asarray(D1y+D2y),(-1,1))
data=np.concatenate((np.ones((x.shape[0],1)),x,y),axis=1)
feature_dim=2
weights=np.ones((feature_dim+1,1))
w_update=np.zeros((feature_dim+1,1))
predict=np.zeros((data.shape[0],1))+0.5
real_class=np.concatenate((np.zeros((len(D1x),1)),np.ones((len(D2x),1))),axis=0)

D=np.zeros((data.shape[0],data.shape[0]))

for opt_itr in range(5000):
    predict = 1 / (1 + np.exp((-1) * matmul(data, weights)))
    gradient = matmul(matrix_transpose(data), (predict - real_class))
    if linalg.cond(x) < 1/sys.float_info.epsilon:
        #invertible
        for i in range(data.shape[0]):
            D[i, i] = exp(-1 * matmul(data[i,:][np.newaxis], weights)) / ((1 + exp(-1 * matmul(data[i,:][np.newaxis], weights))) ** 2)
        H = matmul(matmul(matrix_transpose(data), D), data)
        H_inverse=matrix_inverse(H)
        w_update=matmul(H_inverse,gradient)
    else:
        #singular?
        w_update=gradient
    weights=weights-w_update
    if(np.amax(np.abs(gradient))<0.3):
        print("Iteration:{}".format(opt_itr))
        break

predict = 1 / (1 + np.exp((-1) * matmul(data, weights)))
con_mat=np.zeros((2,2))
for i in range(data.shape[0]):
    con_mat[int(real_class[i,0]),(0 if predict[i,0]<0.5 else 1)]+=1

for x in range(con_mat.shape[0]):
    print(con_mat[x,:])

print("Sensitivity: {}".format(con_mat[1,1]/(con_mat[1,1]+con_mat[1,0])))
print("Specificity: {}".format(con_mat[0,0]/(con_mat[0,1]+con_mat[0,0])))
pyplot.plot(D1x, D1y, 'bo')
pyplot.plot(D2x,D2y,'ro')
pyplot.plot([0,1],[(-weights[0])/weights[2],(-weights[0]-weights[1])/weights[2]])
pyplot.show()