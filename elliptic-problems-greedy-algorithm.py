import numpy as np
import scipy.integrate as integrate
import scipy.sparse as sparse
import numpy.random as random
import matplotlib.pyplot as plt

# 1. Notations of question 8

I = 100
h = 1/(I+1)


def phi(i,x):
    if (x < (i-1)*h) or (x > (i+1)*h):
        return 0.
    else:
        return( 1- np.abs(x-i*h)/h )



# We use scipy.sparse for efficiency

D_diag_values = 2/h
D_side_diag_values = -1/h


D_side_diag = [D_side_diag_values for i in range(1,I+1)]
D_diag = [D_diag_values for i in range(1,I+1)]
D_data = np.array([D_side_diag, D_diag,D_side_diag])
D_offsets = np.array([-1, 0, 1])
D_sparse = sparse.dia_matrix((D_data, D_offsets), shape = (I,I))


M_diag_values = 2*h/3
M_side_diag_values = h/6

M_side_diag = [M_side_diag_values for i in range(1,I+1)]
M_diag = [M_diag_values for i in range(1,I+1)]
M_data = np.array([M_side_diag, M_diag,M_side_diag])
M_offsets = np.array([-1, 0, 1])
M_sparse = sparse.dia_matrix((M_data, M_offsets), shape = (I,I))



# Function f is defined by the f^p_1 et les f^p_2 for 0 < p < P+1. P = 1 for f et P = 2 for g
# We decompose f^p_1 and f^p_2 on the basis phi_i (Discretization)
# fp_alphaTab = np.array([f^p_alpha(i*h) for i in range(1, I+1)])

def f1_1(x):
    return np.cos(2*np.pi*x)
f1_1Tab = np.array([f1_1(i*h) for i in range(1, I+1)])


def f1_2(y):
    return np.cos(4*np.pi*y) 
f1_2Tab = np.array([f1_2(i*h) for i in range(1, I+1)])

def g1_1(x):
    return (np.sin(np.pi*x))**2
g1_1Tab = np.array([g1_1(i*h) for i in range(1, I+1)])

def g1_2(y):
    return np.sin(2*np.pi*y) 
g1_2Tab = np.array([g1_2(i*h) for i in range(1, I+1)])

def g2_1(x):
    return np.sin(10*np.pi*x)
g2_1Tab = np.array([g2_1(i*h) for i in range(1, I+1)])

def g2_2(y):
    return np.sin(np.pi*y)
g2_2Tab = np.array([g2_2(i*h) for i in range(1, I+1)])


# We can also obtain the vectors F^p_alpha using the fp_alphaTab and the matrix M_sparse

F1_1 = M_sparse.dot(f1_1Tab)
F1_2 = M_sparse.dot(f1_2Tab)
G1_1 = M_sparse.dot(g1_1Tab)
G1_2 = M_sparse.dot(g1_2Tab)
G2_1 = M_sparse.dot(g2_1Tab)
G2_2 = M_sparse.dot(g2_2Tab)

F = np.array([[F1_1,F1_2]])
G = np.array([[G1_1,G1_2],[G2_1,G2_2]])    #f(resp g) is given by F (resp G)




# We can finally define the matrix cursiveM(V) and the vectors cursiveF(n,V) and cursiveG(n,V)


def cursiveM(V):
    premierTerme = np.dot(V, D_sparse.dot(V)) * M_sparse # ((tV)DV)*M
    secondTerme = np.dot(V, M_sparse.dot(V)) * D_sparse # ((tV)MV)*D
    return(premierTerme + secondTerme)



# We suppose here that the R_k et S_k for 0<k<n are in the list of vectors listR et listS: listR[k-1] contains R_k and listS[k-1] contains S_k
# tabF here is either F either G (of length P)

def cursiveF(n,V,listR,listS,tabF):
    premierTerme = np.zeros(I)
    P = len(tabF)
    for i in range(P):
        premierTerme += np.dot(V,tabF[i][1]) * tabF[i][0]
    if n==1:
        return premierTerme
    secondTerme = np.zeros(I)
    for k in range(1, n):
        secondTerme += np.dot(V, D_sparse.dot(listS[k-1])) * M_sparse.dot(listR[k-1]) + np.dot(V, M_sparse.dot(listS[k-1])) * D_sparse.dot(listR[k-1])
    return(premierTerme - secondTerme)


def cursiveG(n,V,listR,listS, tabF):
    premierTerme = np.zeros(I)
    P = len(tabF)
    for i in range(P):
        premierTerme += np.dot(V,tabF[i][0]) * tabF[i][1]
    if n==1:
        return premierTerme
    secondTerme = np.zeros(I)
    for k in range(1, n):
        secondTerme += np.dot(V,D_sparse.dot(listR[k-1])) * M_sparse.dot(listS[k-1]) + np.dot(V, M_sparse.dot(listR[k-1])) * D_sparse.dot(listS[k-1])
    return(premierTerme - secondTerme)

# 2. Resolution of the fixed point method

N = 15 # N final value of n
M = 20 # M final value of m (we could also choose to stop when sufficiently close to a fixed point)
S0_n = np.ones(I) # Initial value for the fixed point method



#tabF as an argument for modularity reasons
#Nfinal as an argument to run the simulation for all possible values of n (to plot the error)

def simulation(tabF,Nfinal):
    
    listR = []
    listS = []
    for n in range(1,Nfinal+1):
        Smn = S0_n
        for m in range(1,M+1):
            
            rhs1 = cursiveF(n,Smn,listR,listS,tabF)
            mat1 = cursiveM(Smn)
            Rmn =  sparse.linalg.spsolve(mat1, rhs1)            
            rhs2 = cursiveG(n,Rmn,listR,listS,tabF)
            mat2 = cursiveM(Rmn)
            Smn =  sparse.linalg.spsolve(mat2, rhs2)         #Using the sparsity of the matrixes
        listR.append(Rmn)
        listS.append(Smn)
    return (listR,listS)
        
    

def evaluation(x,y,Nfinal, listR, listS): #computes u(x,y) (u given by listR, listS)
    res = 0
    phi_values_x = np.array([phi(j,x) for j in range(1, I+1)])
    phi_values_y = np.array([phi(j,y) for j in range(1, I+1)])
    for i in range(Nfinal):
        res += np.dot(listR[i],phi_values_x) * np.dot(listS[i],phi_values_y)
    return res

# 3D display of the obtained approximation

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def trace(tabF, Nfinal):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    listR, listS = simulation(tabF, Nfinal)
    
    X = np.linspace(0,1,100)
    Y = np.linspace(0,1,100)
    Z = np.array([np.array([evaluation(x,y, Nfinal, listR, listS) for x in X]) for y in Y])
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, Z, cmap= cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()


trace(F,N)



# We study the influence of n on the approximation quality (plotting the error wrt n)
# Other parameters (M and S0_n) are constant: M = 50 and S0_n = np.ones(I)
# We compare the approximations for n = 1, 2, 3, 4, 5 to the one obtained for n = N (15 for instance), supposed to be the exact solution
# The error metric is the euclidean distance  between the discretization values of the functions

    

valeurs_ref = np.zeros((I,I))   #Values of the "exact" solution

def initialisation(tabF):
    listR, listS = simulation(tabF,N)
    for i in range(I):
        for j in range(I):
            valeurs_ref[i][j]= evaluation(i*h,j*h, N, listR, listS)
            
    

def distance(listR, listS):  #distance of a solution characterized by listR, listS to the "exact" solution
    l = len(listR)
    s = 0
    for i in range(I):
        for j in range(I):
           s += (evaluation(i*h,j*h, l, listR, listS)-valeurs_ref[i,j])**2
    return (np.sqrt(s))
            
            
def tracer_erreur(tabF):
    
    initialisation(tabF)
    tab_erreurs = []
    for n in range(1,N):
        listR, listS = simulation(tabF, n)
        tab_erreurs.append( distance(listR, listS))
    X = np.linspace(1,N-1,N-1)
    plt.plot(X,tab_erreurs)
    plt.show()

tracer_erreur(F)










