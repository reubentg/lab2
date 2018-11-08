import numpy as np
import sys

aa = np.array([0.0, 0.01, 0.02, 0.04, 0.05, 0.06, 0.07, 0.2, 0.25, 0.3])
a = np.cumsum(aa) # [ 0.    0.01  0.03  0.07  0.12  0.18  0.25  0.45  0.7   1.  ]
# a = np.linspace(0, 1, 10) # array([ 0.        ,  0.11111111,  0.22222222,  0.33333333,  0.44444444,  0.55555556,  0.66666667,  0.77777778,  0.88888889,  1.        ])
b = np.append([0], a) # array([ 0.        , 0.        ,  0.11111111,  0.22222222,  0.33333333,  0.44444444,  0.55555556,  0.66666667,  0.77777778,  0.88888889,  1.        ])
b[-1::] += sys.float_info.epsilon # add small number epsilon
r = np.random.choice(101, 10) / 100.0
r = np.array([ 0.0 ,  0.005,  0.01,  0.02 ,  0.1 ])


bins_selected = np.digitize(r, b, right=False) # [ 2  2  2  2  2  3  4  9 10 10]
print "random values  ", r
print "boundaries     ", b
print "bin indices    ", bins_selected - 1


aa = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5])
# choices = np.random.choice(a, 10,p=aa)
# print choices
