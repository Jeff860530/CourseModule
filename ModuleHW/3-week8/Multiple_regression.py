import numpy as np
data=np.genfromtxt("RegressionExample.txt", delimiter=' ',dtype =float)
x=data[:,1:]
y=data[:,0]

def closed_form_solution(y,x):
    return np.linalg.inv((x.T@x))@(x.T)@y

B_head = closed_form_solution(y,x)
print(B_head.reshape(4,1))


