from mpl_toolkits import mplot3d
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

# Déclaration de la figure
fig = plt.figure()
ax = plt.axes(projection='3d')

# rayon du brin (adimensionnel).
R = 0.8
# Paramètres
petit = 2
moyen = 4
grand = 5

# Fonctions de sélections des courbes
# La pomme de Touline se fait en 3 étapes.
# A chaque étape on fait trois boucles "ovoïdes".
# Etape 1 : les trois premières boucles sont faites dans le plan xz, plus large en x.
step1 = np.vectorize(lambda t : (t >= 0 and t <= 2*pi))
# Etape 2 : les trois suivantes sont dans le plan yz, plus large en z, de sorte à "encercler" les boucles de l'étape 1.
step2 = np.vectorize(lambda t : (t > 2*pi and t <= 4*pi))
# Etape 3 : les trois dernières sont dans le plan xy, plus large en y, de sorte à "encercler" les boucles de l'étape 2 et à passer dans celles de l'étape 1.
step3 = np.vectorize(lambda t : (t > 4*pi and t <= 6*pi))

# Equations paramétriques du noeud
dim = 6*pi
nPoints = 1000 # nombre de points à calculer
t = np.linspace(0, dim, nPoints)
#x = step1(t) * (grand*np.sin(3*t)) + step2(t) * (petit*np.cos(t/2)) + step3(t) * (moyen*np.cos(3*t))
#y = step1(t) * (petit*np.cos(t/2)) + step2(t) * (moyen*np.cos(3*t)) + step3(t) * (grand*np.sin(3*t))
#z = step1(t) * (moyen*np.cos(3*t)) + step2(t) * (grand*np.sin(3*t)) + step3(t) * (petit*np.cos(t/2))

#x = step1(t) * (grand*np.sin(3*t)) + step2(t) * (petit*(3*pi-t)/pi) + step3(t) * (moyen*np.cos(3*t))
#y = step1(t) * (petit*(pi-t)/pi) + step2(t) * (moyen*np.cos(3*t)) + step3(t) * (grand*np.sin(3*t))
#z = step1(t) * (moyen*np.cos(3*t)) + step2(t) * (grand*np.sin(3*t)) + step3(t) * (petit*(5*pi-t)/pi)

def correcteur(t) :
    return np.cos((t%(2*pi))/5 - pi/5)

x = step1(t) * (grand*np.sin(3*t)) + step2(t) * (petit*(3*pi-t)/pi) + step3(t) * (moyen*np.cos(3*t)*correcteur(t))
y = step1(t) * (petit*(pi-t)/pi) + step2(t) * (moyen*np.cos(3*t)*correcteur(t)) + step3(t) * (grand*np.sin(3*t))
z = step1(t) * (moyen*np.cos(3*t)*correcteur(t)) + step2(t) * (grand*np.sin(3*t)) + step3(t) * (petit*(5*pi-t)/pi)

ax.plot3D (x, y, z, 'green')

# valeurs différentielles
dt = dim/nPoints
dx = np.diff(x)
dy = np.diff(y)
dz = np.diff(z)

# calcul de la longueur (adimensionnelle)
ds = np.sqrt(dx**2 + dy**2 + dz**2)
longueur = np.sum(ds)

# Construction du repère de Serret-Frenet
# T est le vecteur tangeant à la courbure du noeud
Tx = dx/dt
Ty = dy/dt
Tz = dz/dt

# N est un vecteur normal à T (i.e. tel que T /\ N = 0)
Nx = Tz - Ty
Ny = Tx - Tz
Nz = Ty - Tx

# B est le produit vectoriel des vecteurs T et N (B = T /\ N)
Bx = Ty*Nz - Tz*Ny 
By = Tz*Nx - Tx*Nz
Bz = Tx*Ny - Ty*Nx

# On normalise pour avoir un vecteur orthonormé
Tnorm = np.sqrt(Tx**2 + Ty**2 + Tz**2)
Nnorm = np.sqrt(Nx**2 + Ny**2 + Nz**2)
Bnorm = np.sqrt(Bx**2 + By**2 + Bz**2)

(Tx, Ty, Tz) = (Tx/Tnorm, Ty/Tnorm, Tz/Tnorm)
(Nx, Ny, Nz) = (Nx/Nnorm, Ny/Nnorm, Nz/Nnorm)
(Bx, By, Bz) = (Bx/Bnorm, By/Bnorm, Bz/Bnorm)

# On utilise le repère de Serret-Frenet pour construire les torons du noeud en
# translatant le brin central par une série de vecteurs qui "tournent" autour du centre.

# Un rayon de 0.8 semble bonne pour que les torons soient bien "serrés", sans se chevaucher.
# C'est la partie la plus "fumeuse" car elle repose seulement sur l'observation de la figure.     
N = 50       # nombre de brins modélisés
# for k in range(N) :
#     Vx = R * (np.cos(k*2*pi/N)*Nx + np.sin(k*2*pi/N)*Bx)
#     Vy = R * (np.cos(k*2*pi/N)*Ny + np.sin(k*2*pi/N)*By)
#     Vz = R * (np.cos(k*2*pi/N)*Nz + np.sin(k*2*pi/N)*Bz)
#     ax.plot3D (x[:-1]+Vx, y[:-1]+Vy, z[:-1]+Vz, 'blue', alpha=0.1)

# On va essayer de déterminer le "rayon de serrage" de façon plus analytique...
# On cherche à déterminer une métrique de serrage.
# L'idée est d'imaginer une "peau" à l'interface entre deux brins.
# Au voisinage de chaque point :
# on souhaite maximiser la présence d'autres points dans la peau (indicateur de serrage)...
# ...tout en excluant les points en deça de la peau (indicateur de chevauchement).
def voisins(p1,p2,R, T,tolerance) :
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    # p2 doit être distant de p1 de moins de R
    prod_scalaire = dx*T[0] + dy*T[1] + dz*T[2]
    # le vecteur de p1 vers p2 doit être "orthogonal" à la direction du toron en p1
    return (norm <= R) and (abs(prod_scalaire)/norm <= tolerance)

def peau(p1,p2,R) :
    return voisins(p1,p2,1.2*R) and (not voisins(p1,p2,0.8*R))

points = np.transpose(np.array([x,y,z]))
vecT = np.transpose(np.array([Tx,Ty,Tz]))
offset1 = int(1.5*pi * nPoints / dim)
offset2 = int(2.5*pi * nPoints / dim)

def calculPeau_aux(R) :
    nClose = 0
    nFar = 0
    for i in range(nPoints) :
        #print("######################")
        for p in points[i+offset1:i+offset2] :
            nClose += voisins(points[i], p, 0.9*2*R, vecT[i], 0.01)
            nFar   += voisins(points[i], p, 1.1*2*R, vecT[i], 0.01)
    nPeau = nFar - nClose
    print(nClose,nFar,nPeau)
    return (nClose,nFar,nPeau)

calculPeau = np.vectorize(calculPeau_aux)

nStep = 10
R = np.linspace(0.5,1,nStep)
metrics = calculPeau(R)

fig2 = plt.figure()
ax2 = plt.axes()
ax2.plot(R,metrics[0], label="close")
ax2.plot(R,metrics[1], label="far")
ax2.plot(R,metrics[2], label="peau")
ax2.legend()

# for i in range(nStep) :
#     if metrics[2][i] <= metrics[0][i] :
#         R = R[i]
#         break

# Cette méthode montre un pic pour R =~ 0.87
R = 0.87

for k in range(N) :
    Vx = R * (np.cos(k*2*pi/N)*Nx + np.sin(k*2*pi/N)*Bx)
    Vy = R * (np.cos(k*2*pi/N)*Ny + np.sin(k*2*pi/N)*By)
    Vz = R * (np.cos(k*2*pi/N)*Nz + np.sin(k*2*pi/N)*Bz)
    ax.plot3D (x[:-1]+Vx, y[:-1]+Vy, z[:-1]+Vz, 'blue', alpha=0.1)
    
# Finalement on peut calculer le rapport longueur / diamètre
rapport = longueur / (2*R)
# Par exemple, on peut calculer la longueur (en mètres) d'une corde de 14 mm de diamètre :
longueur_m = rapport * 0.014
print(f"R = {R}\t L = {longueur}\t rapport = {rapport}\t longueur = {longueur_m} m")

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Pomme de Touline')
plt.show()
