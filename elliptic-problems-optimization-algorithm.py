import numpy as np 
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import scipy.integrate as integrate
import scipy.sparse as sparse 

# Discretization parameters

I = 50
h = 1/(I+1)


# Matrixes D and M (sparse)

D_diag_values = 2/h
D_side_diag_values = -1/h


D_side_diag = [D_side_diag_values for i in range(1,I+1)]
D_diag = [D_diag_values for i in range(1,I+1)]
D_data = np.array([D_side_diag, D_diag,D_side_diag])
D_offsets = np.array([-1, 0, 1])
D = sparse.dia_matrix((D_data, D_offsets), shape = (I,I))


M_diag_values = 2*h/3
M_side_diag_values = h/6

M_side_diag = [M_side_diag_values for i in range(1,I+1)]
M_diag = [M_diag_values for i in range(1,I+1)]
M_data = np.array([M_side_diag, M_diag,M_side_diag])
M_offsets = np.array([-1, 0, 1])
M = sparse.dia_matrix((M_data, M_offsets), shape = (I,I))

# Functions phi_i and vectorization

def phi(i,x):
    if (x < (i-1)*h) or (x> (i+1)*h):
        return 0
    else:
        return np.abs(x-i*h)/h

def phi_vec(x):
        return np.array([phi(i,x) for i in range(I)])

# Function f, in separated form (first example)

def f_1(x):
    return np.cos(2*np.pi*x)
def toIntegrate_1(i):
    return lambda x : f_1(x)*phi(i,x)



def f_2(y):
    return np.cos(4*np.pi*y)
def toIntegrate_2(j):
    return lambda y : f_2(y)*phi(j,y)


# Matrix F

F = np.zeros((I,I))
for i in range(I):
    for j in range(I):
        F[i,j] = integrate.quad(toIntegrate_1(i), 0, 1)[0] * integrate.quad(toIntegrate_2(j), 0, 1)[0]

# Function to minimize: arguments R and S are written as vectors of R^{2I} (concatenation)

def toMinimize(R_concatenate_S):
    R = R_concatenate_S[0:I]
    S = R_concatenate_S[I:2*I]
    premierTerme = 0
    if len(listR) > 0:
        for k in range(n-1):
            premierTerme += np.dot(R, D.dot(listR[k]))*np.dot(S, M.dot(listS[k])) + np.dot(R, M.dot(listR[k]))*np.dot(S, D.dot(listS[k]))
    secondTerme = np.dot(R, D.dot(R))*np.dot(S, M.dot(S)) + np.dot(R, M.dot(R))*np.dot(S, D.dot(S))
    troisiemeTerme = np.dot(R, F.dot(S))
    return premierTerme + secondTerme - troisiemeTerme

# Beginning of the loop: we solve problem (5) N times

N = 20
listR = [] # Contient les R^(k)
listS = [] # Contient les S^(k)
for n in range(N):
    Rn_concatenate_Sn = optimize.minimize(toMinimize, np.ones(2*I))
    toAppendR = Rn_concatenate_Sn.x[0:I]
    toAppendS = Rn_concatenate_Sn.x[I:2*I]
    listR.append(toAppendR)
    listS.append(toAppendS)


def approximation(x,y):
        res = 0
        for k in range(len(listR)):
                res += np.dot(listR[k], phi_vec(x)) * np.dot(listS[k], phi_vec(y))
        return res


# 3D display of the approximation

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.ticker import LinearLocator, FormatStrFormatter


fig = plt.figure()
ax = fig.gca(projection='3d')
    
    
X = np.linspace(0,1,100)
Y = np.linspace(0,1,100)
Z = np.array([np.array([approximation(x,y) for x in X]) for y in Y])
X, Y = np.meshgrid(X, Y)
surf = ax.plot_surface(X, Y, Z, cmap= cm.coolwarm, linewidth=0, antialiased=False)

ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


fig.colorbar(surf, shrink=0.5, aspect=5)
    
plt.show()




                