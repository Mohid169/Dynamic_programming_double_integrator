import numpy as np

#Define double integrator system variables
def double_integrator():
    A = np.array([[0 ,1], [0, 0]])
    B = np.array([0, 1])
    C = np.eye(2)
    D = np.zeros((2,1))
    return LinearSystem(A,B,C,D)



