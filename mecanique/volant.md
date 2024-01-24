<img src="media/CC-BY-NC-ND.png" alt="Drawing" style="width: 150px;"/> 

**Auteur** : Christophe Jorssen

Bienvenue dans cette activité informatique. 

Comme dans l'activité informatique précédente, je te propose d'utiliser le language Python et quelques unes de ses bibliothèques scientifiques. 

Commence par exécuter la cellule suivante sans te soucier du contenu. Tu te rappelles : il suffit de sélectionner la cellule et d'appuyer sur <kbd>SHIFT</kbd>+<kbd>ENTER</kbd> (ou <kbd>MAJ</kbd>+<kbd>ENTRÉE</kbd>). Tu vas voir apparaître une figure qui correspond à la trajectoire du volant de badminton, animé d'une vitesse initiale de norme $v_0$ et dont la direction forme un angle $\theta_0$ avec l'horizontale. Tu pourras faire varier les valeurs de $v_0$ et de $\theta_0$ à l'aide de curseur.

La suite sera consacrée au décorticage du code.


```python
%matplotlib inline
from ipywidgets import interact

import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt

m = 5E-3 # kg
g = 9.81 # m/s²
rho = 1.2 # kg/m³
S = 2.8E-3 # m²

def solve(v0 = 58, theta0 = 52, Cx = 0.6):
    theta0 = theta0 * np.pi / 180
    
    def loi_de_newton(dAdt, t):
        x = dAdt[0] ; vx = dAdt[1] ; y = dAdt[2] ; vy = dAdt[3]
        return [vx, 
                - 0.5 * rho * S * Cx * np.sqrt(vx**2 + vy**2) * vx / m,
                vy, 
                -g - 0.5 * rho * S * Cx * np.sqrt(vx**2 + vy**2) * vy / m]
    
    dAdt0 = [0, v0 * np.cos(theta0), 0, v0 * np.sin(theta0)]
    
    t = np.linspace(0, 3, 300)
    
    sol = odeint(loi_de_newton, dAdt0, t)

    x = sol[:, 0]
    y = sol[:, 2]
    
    fig, ax = plt.subplots(figsize = (6, 4))
    
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 12)
    
    ax.set_aspect('equal')
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    
    ax.plot(x[y >= 0], y[y >= 0])

interact(solve, v0 = (0., 70., 5), theta0 = (0., 90., 5.), Cx = (0., 2, .1));
```

# Bibliothèques

Les bibliothèques dont nous allons nous servir sont les mêmes que celles de l'activité informatique précédente.


```python
%matplotlib inline
import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
```

# Paramètres physiques

Définissons ensuite les variables `m`, `g`, `v0`, `theta0`, `rho`, `S` et `Cx` qui correspondent respectivement à :

* la masse du volant ;
* l'accélération de la pesanteur ;
* la norme de la vitesse initiale ;
* l'angle que forme le vecteur vitesse initiale avec l'horizontale ;
* la masse volumique de l'air ;
* le maître couple du volant ;
* le $C_x$ du volant. 

Les valeurs numériques de ces grandeurs sont exprimées dans les unités de base du système internationale. Nous donnons ici des valeurs correspondant à un volant en plume. J'ai utilisé les valeurs données dans [cet article](https://hal-polytechnique.archives-ouvertes.fr/hal-01214304/document).


```python
# Masse
m = 5E-3 # kg
# Accélération de la pesanteur
g = 9.81 # m/s²
# Norme de la vitesse initiale
v0 = 58 # m/s
# Angle de la vitesse initiale par rapport à l'axe (Ox)
theta0 = 52 # °
theta0 = theta0 * np.pi / 180 # Conversion en radians
# Masse volumique de l'air
rho = 1.2 # kg/m³
# Maître couple du volant
S = 2.8E-3 # m²
# Cx (coefficient de traînée)
Cx = 0.6 # Sans dimension
```

# Équation différentielle

On définit ensuite l'équation différentielle qui est obtenue à l'aide de la deuxième loi de Newton. La deuxième loi de Newton s'écrit : $m \vec{a} = m \vec{g} + \vec{F}$, où la force de frottement $\vec{F}$ s'écrit $\vec{F} = - \frac{1}{2} \rho S C_x v^2 \frac{\vec{v}}{\|\vec{v}\|}$.

En projection sur les vecteurs unitaires $\vec{e}_x$ horizontal et $\vec{e}_y$ vertical, on a : $$\left\{\begin{array}{ccl} \frac{dv_x}{dt} & = & - \frac{1}{2 m} \rho S C_x \sqrt{v_x^2 + v_y^2} v_x \\ \frac{dv_x}{dt} & = & -g - \frac{1}{2 m} \rho S C_x \sqrt{v_x^2 + v_y^2} v_y \end{array}\right..$$

En outre, les vitesses $v_x$ et $v_y$ sont définies par : $\left\{\begin{array}{ccc} \frac{dx}{dt} & = & v_x \\ \frac{dy}{dt} & = & v_y \end{array}\right.$.

C'est exactement ces quatre égalités-là qui découlent de la loi de Newton et qui définissent l'équation différentielle que nous allons résoudre numériquement. 

Pour cela, nous allons définir une fonction `loi_de_newton`. Elle prend deux arguments : 

* une `list`(liste), notée `dAdt`, de quatre éléments contenant, dans l'ordre, les valeurs de $x$, $v_x$, $y$ et $v_y$ ;
* un `float` (flottant), noté `t`, précisant la date $t$ à laquelle sont prises les valeurs de $x$, $v_x$, $y$ et $v_y$.

La fonction renvoie une `list` de quatre éléments qui correspondent, dans l'ordre, à la dérivée des éléments de `dAdt`, soit :

* $v_x$ pour $x$ ;
* $- \frac{1}{2 m} \rho S C_x \sqrt{v_x^2 + v_y^2} v_x$ pour $v_x$ ;
* $v_y$ pour $y$ ;
* $-g - \frac{1}{2 m} \rho S C_x \sqrt{v_x^2 + v_y^2} v_y$ pour $v_y$.


```python
def loi_de_newton(dAdt, t):
    x = dAdt[0] ; vx = dAdt[1] ; y = dAdt[2] ; vy = dAdt[3]
    return [vx, 
            - 0.5 * rho * S * Cx * np.sqrt(vx**2 + vy**2) * vx / m,
            vy, 
            -g - 0.5 * rho * S * Cx * np.sqrt(vx**2 + vy**2) * vy / m]
```

Tu as remarqué ? C'est très pratique : on peut aller à la ligne après chaque virgule lorsque l'on définit une `list`. C'est plus lisible comme ça !

# Conditions intiales

Définissons les conditions initiales : $x(0) = 0$, $v_x(0) = v_0 \cos \theta_0$, $y(0) = 0$ et $v_y(0) = v_0 \sin \theta_0$.


```python
dAdt0 = [0, v0 * np.cos(theta0), 0, v0 * np.sin(theta0)]
```

# Dates associées à la résolution numérique

On définit les dates à laquelle on cherche la solution numérique de l'équation différentielle. J'ai choisit d'avoir 300 dates sur une durée de $3\,\mathrm{s}$.


```python
t = np.linspace(0, 3, 300)
```

# Résolution de l'équation différentielle

Ça y est : tout est prêt. Utilisons `odeint` pour résoudre l'équation différentielle.


```python
sol = odeint(loi_de_newton, dAdt0, t)
```

Et voilà, c'est fait ! Il ne reste plus qu'à tracer ce qui nous intéresse. Je te propose de tracer la trajectoire dans le plan $(O, \vec{e}_x, \vec{e}_y)$, soit $y(t)$ en fonction de $x(t)$. 


```python
x = sol[:, 0]
vx = sol[:, 1]
y = sol[:, 2]
vy = sol[:, 3]
```

On finit par utiliser `matplotlib` et sa commande `plot` à laquelle on donne en argument l'abscisse `x` et l'ordonnée `y`. Il ne reste plus qu'à afficher !


```python
fig2, ax = plt.subplots(figsize = (6, 4))
# On définit les bornes des axes (en m)
ax.set_xlim(0, 15)
ax.set_ylim(0, 12)

# On fait en sorte que les deux axes aient la même échelle (afin que l'angle theta0 
# soit correctement « visible » sur le graphe)
ax.set_aspect('equal')

# On ajoute la légende des axes
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')

# On trace...
ax.plot(x, y)

# ...et on affiche
fig2.show()
```

En complétant la cellule ci-dessous, amuse-toi à tracer l'évolution du rapport des normes du poids et de la force de frottement. Que retrouves-tu ?


```python
fig3, ax = plt.subplots(figsize = (6, 4))

# Complète ci-dessous !


fig3.show()
```
