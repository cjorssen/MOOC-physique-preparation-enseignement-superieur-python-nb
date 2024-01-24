<img src="media/CC-BY-NC-ND.png" alt="Drawing" style="width: 150px;"/> 

**Auteur** : Christophe Jorssen

Dans ce *notebook*, je te propose de mettre en œuvre la méthode que je t'ai présentée dans la vidéo qui permet de passer d'une déformation sur une longueur $L$ finie de corde à une déformation périodique de période $2L$ sur une longueur de corde infinie. On obtiendra ainsi les deux ondes progressives périodiques se propageant en sens opposés. Ces dernières pourront être décomposées en ondes progressives harmoniques qui, sommées deux à deux, donneront accès aux ondes stationnaires harmoniques.

Commençons par importer les bibliothèques nécessaires.


```python
%matplotlib inline
import numpy as np
from matplotlib import pyplot as plt
```

Commençons par définir la fonction `f_L` définie sur l'intervalle $[0, L]$. Je choisis ici la forme d'une corde de guitare pincée à une distance $\alpha L$ (`alpha * L`) avec un champ d'élongation $a$ (`a`). Il s'agit d'une fonction affine par morceaux. Les valeurs numériques sont choisies arbitrairement. Je prends $L = 1\,\mathrm{m}$ dans toute la suite.

Je te mets aussi en commentaire une forme initiale de type « impulsion » localisée. Si tu veux, tu pourras changer de forme initiale une fois que tu auras « joué » avec la fonction affine par morceaux.


```python
L = 1 # m
alpha = .25
a = .1 # m

def f_L(x):
    if x < alpha * L:
        return a * x / alpha
    else:
        return (a / (alpha - 1)) * ( x / L - 1)
    
#def f_L(x):
#    return a * np.exp(-((x - alpha * L) / .05)**2)
```

Ensuite, on définit la fonction `f_2L` définie sur l'intervalle $[-L, L]$ comme décrit dans la vidéo (si $x \in [0, L]$, alors $f_{2L}(x) = f_L(x)$, sinon $f_{2L}(x) = - f_L(-x)$).


```python
def f_2L(x):
    if x >= 0:
        return f_L(x)
    else:
        return -f_L(-x)
```

Enfin, on définit la fonction `f` définie sur $]-\infty, + \infty[$ comme décrit dans la vidéo en périodisant la fonction $f_{2L}$. Pour cela, on utilise la fonction partie entière (`np.floor`) qui permet de ramener un réel $x$ dans l'intervalle $[-L, L]$.


```python
def f(x):
    return f_2L(x - np.floor((x + L)/ (2 * L)) * 2 * L)
```

Pour nous faciliter la tâche, on transforme ces fonctions de telle sorte qu'elles soient facilement utilisables avec les objets `ndarray` de `numpy` en définissant des versions [« vectorisées »](https://docs.scipy.org/doc/numpy/reference/generated/numpy.vectorize.html).


```python
vf_L = np.vectorize(f_L)
vf_2L = np.vectorize(f_2L)
vf = np.vectorize(f)
```

On trace tout ça pour vérifier.


```python
Npt = 100

fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex = True, sharey = True)

x_L = np.linspace(0, L, Npt)
ax1.plot(x_L, vf_L(x_L), 'b')
ax1.grid(which = 'both')
ax1.set_ylabel(r'$f_L(x)$ (m)')

x_2L = np.linspace(-L, L, 2 * Npt)
ax2.plot(x_2L, vf_2L(x_2L), 'r')
ax2.grid(which = 'both')
ax2.set_ylabel(r'$f_{2L}(x)$ (m)')

x = np.linspace(-3 * L, 3 * L, 3 * 2 * Npt)
ax3.plot(x, vf(x), 'y')
ax3.grid(which = 'both')
ax3.set_ylabel(r'$f(x)$ (m)')
ax3.set_xlabel(r'$x$ (m)')

fig1.show()
        
```

Il ne reste plus qu'à faire une animation pour voir comment les ondes se comportent. En vert, on représente le champ de déplacement $u_z(x, t)$ de la corde. Il est obtenu à l'aide de la formule donnée dans la vidéo : $u_z(x, t) = \frac{1}{2}[u_{z, \nearrow}(x, t) + u_{z, \searrow}(x, t)]$, avec :
* $u_{z, \nearrow}(x, t) = f(x - ct)$ l'onde progressive $2L$-périodique se propageant vers les $x$ croissants (en orange) ;
* $u_{z, \searrow}(x, t) = f(x + ct)$ l'onde progressive $2L$-périodique se propageant vers les $x$ décroissants (en violet).


```python
from matplotlib.animation import FuncAnimation

fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex = True, sharey = True)

line1, = ax1.plot(x, vf(x), 'g')
ax1.grid(which = 'both')
ax1.set_ylabel(r'$u_z(x, t)$ (m)')

line2, = ax2.plot(x, .5 * vf(x), 'orange')
ax2.grid(which = 'both')
ax2.set_ylabel(r'$u_{z, \nearrow}(x, t)$ (m)')

line3, = ax3.plot(x, .5 * vf(x), 'purple')
ax3.grid(which = 'both')
ax3.set_ylabel(r'$u_{z, \searrow}(x, t)$ (m)')
ax3.set_xlabel(r'$x$ (m)')

def init():
    ax1.set_xlim(-3 * L, 3 * L)
    ax1.set_ylim(-a, a)
    ax2.set_xlim(-3 * L, 3 * L)
    ax2.set_ylim(-a, a)
    ax3.set_xlim(-3 * L, 3 * L)
    ax3.set_ylim(-a, a)
    return line1, line2, line3,

def update(frame):
    line1.set_data(x, .5 * (vf(x + frame / Npt) + vf(x - frame / Npt)))
    line2.set_data(x, .5 * vf(x - frame / Npt))
    line3.set_data(x, .5 * vf(x + frame / Npt))
    return line1, line2, line3,

ani = FuncAnimation(fig2, update, frames = 2 * Npt,
                    init_func = init, blit = False)
```

Focalise-toi bien sur la partie « utile » de la courbe verte, c'est-à-dire la partie correspondant à $x \in [0, L]$. As-tu remarqué que tout se passe comme si l'onde se *réfléchissait* aux extrémités, c'est-à-dire en $x = 0$ et en $x = L$ ?

Je te propose maintenant de passer à la décomposition de Fourier. Ne t'attarde pas sur les définitions des coefficients de Fourier que j'utilise : ce n'est pas le but ici.


```python
from scipy.integrate import quad
import functools

@functools.lru_cache(maxsize = 128)
def aFourier(i):
    Lambda = 2 * L
    tmp = quad(lambda x:2 * f(x) * np.cos(2 * np.pi * i * x / Lambda) / Lambda, -Lambda/2, Lambda/2)[0]
    if i == 0:
        return tmp / 2
    else:
        return tmp
    
@functools.lru_cache(maxsize = 128)
def bFourier(i):
    if i == 0:
        return 0
    Lambda = 2 * L
    return quad(lambda x:2 * f(x) * np.sin(2 * np.pi * i * x / Lambda) / Lambda, -Lambda/2, Lambda/2)[0]

def cFourier(i):
    return np.sqrt(aFourier(i)**2 + bFourier(i)**2)

def argFourier(i):
    return np.arctan2(bFourier(i), aFourier(i))
```

Je définis ensuite la fonction `Harm(i, x)` qui renvoie la valeur de l'harmonique de rang `i` à l'abscisse `x`.


```python
def Harm(i, x):
    return aFourier(i) * np.cos(2 * np.pi * i * x / (2 * L)) + bFourier(i) * np.sin(2 * np.pi * i * x / (2 * L))
```

Regarde. Plus on ajoute d'harmoniques, plus on s'approche de la fonction `f_L`.


```python
fig3, ax = plt.subplots()
fFourier = np.zeros(x_L.size)

N = 10

for i in range(N):
    fFourier += Harm(i, x_L)
    ax.plot(x_L, fFourier)
    
fig3.show()

```

Pour finir, je te propose une animation de synthèse. Il s'agit de tracer l'évolution temporelle des ondes stationnaires harmoniques résultant de la somme des ondes progressives harmoniques, se propageant en sens opposés, pour différents harmoniques (ici, on s'arrête à l'harmonique de rang `N`). Observe le caractère stationnaire des ondes stationnaires harmoniques. Observe la présence des nœuds et des ventres.


```python
fig4, ax = plt.subplots(N + 2, 3, sharex = True, sharey = 'row', figsize = (8, 15))

lines0 = [ax[i, 0].plot([], [])[0] for i in range(N + 1)]
lines1 = [ax[i, 1].plot([], [])[0] for i in range(N + 1)]
lines2 = [ax[i, 2].plot([], [])[0] for i in range(N + 1)]
lines3 = [ax[-1, j].plot([], [])[0] for j in range(3)]

ax[0, 0].set_xlim(0, L)
ax[-1, 0].set_xlabel(r'$x$ (m)')
ax[-1, 1].set_xlabel(r'$x$ (m)')
ax[-1, 2].set_xlabel(r'$x$ (m)')

for i in range(0, N + 1):
    ax[i, 0].set_ylabel(r'$u_{{z,{0}}}(x, t)$'.format(i))
    ax[i, 1].set_ylabel(r'$u_{{z,{0},\nearrow}}(x, t)$'.format(i))
    ax[i, 2].set_ylabel(r'$u_{{z,{0},\searrow}}(x, t)$'.format(i))
    
    if i > 0:
        for j in range(3):
            ax[i, j].set_ylim(np.min(Harm(i, x_2L)), np.max(Harm(i, x_2L)))

ax[-1, 0].set_ylabel(r'$u_z(x, t)$')
ax[-1, 1].set_ylabel(r'$u_{z, \nearrow}(x, t)$')
ax[-1, 2].set_ylabel(r'$u_{z, \searrow}(x, t)$')

for j in range(3):
    ax[-1, j].set_ylim(np.min(vf(x_2L)), np.max(vf(x_2L)))

def init():
    for i, line in enumerate(lines0):
        line.set_data(x_L, Harm(i, x_L))
    for i, line in enumerate(lines1):
        line.set_data(x_L, .5 * Harm(i, x_L))
    for i, line in enumerate(lines2):
        line.set_data(x_L, .5 * Harm(i, x_L))
        
    lines3[0].set_data(x_L, vf(x_L))
    lines3[1].set_data(x_L, .5 * vf(x_L))
    lines3[2].set_data(x_L, .5 * vf(x_L))
    
    return lines0 + lines1 + lines2 + lines3


def update(frame):
    for i, line in enumerate(lines0):
        line.set_data(x_L, .5 * (Harm(i, x_L + frame / Npt) + Harm(i, x_L - frame / Npt)))
    for i, line in enumerate(lines1):
        line.set_data(x_L, .5 * Harm(i, x_L - frame / Npt))
    for i, line in enumerate(lines2):
        line.set_data(x_L, .5 * Harm(i, x_L + frame / Npt))

    lines3[0].set_data(x_L, .5 * (vf(x_L + frame / Npt) + vf(x_L  - frame / Npt)))
    lines3[1].set_data(x_L, .5 * vf(x_L - frame / Npt))
    lines3[2].set_data(x_L, .5 * vf(x_L + frame / Npt))

    return lines0 + lines1 + lines2

ani = FuncAnimation(fig4, update, frames = 2 * Npt, interval = 10,
                    init_func = init, blit = False)

fig4.show()
```


```python

```
