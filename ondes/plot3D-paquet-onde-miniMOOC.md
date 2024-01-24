<img src="media/CC-BY-NC-ND.png" alt="Drawing" style="width: 150px;"/> 

**Auteur** : Christophe Jorssen

Dans ce *notebook*, je te propose de visualiser simultanément la double dépendance en $x$ et en $t$ du champ de déplacement $u_z(x, t)$. Pour cela, nous allons, en particulier, tracer la surface représentant $u_z(x, t)$. Pour permettre le tracé en trois dimensions, il faut faire appel à un paquet complémentaire de la bibliothèque `matplotlib` : `mpl_toolkits.mplot3d.axes3d`.


```python
%matplotlib inline
from ipywidgets import interact
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d
```

Comme nous l'avons déjà fait dans le module de mécanique, la cellule ci-dessous va définir une figure interactive. Trois représentations graphiques vont apparaître : 

* la surface $u_z(x, t)$ pour $x \in [x_{\min}, x_{\max}]$ et $t \in [t_{\min}, t_{\max}]$ ;
* l'évolution au cours du temps du champ de déplacement $u_z(x_0, t)$ au point d'abscisse $x_0$ de la corde ;
* la forme de la corde à la date $t_0$, c'est-à-dire l'évolution du champ de déplacement $u_z(x, t_0)$ pour $x \in [x_{\min}, x_{\max}]$.

Dans un premier temps, tu n'as qu'à l'exécuter et à « jouer » avec les paramètres $x_0$ et $t_0$.

J'ai choisi de définir une forme initiale $f$ *via* la fonction $x \mapsto \mathrm{e}^{-x^2}$. Si tu veux, tu peux modifier et choisir une autre fonction. Tu peux aussi modifier la célérité $c$ que j'ai choisie (arbitrairement) égale à 1 m/s.

À chaque modification de $f$ ou $c$, n'oublie pas de réexécuter la cellule afin que le changement de définition soit pris en compte.


```python
def f(x):
    return np.exp(- x**2)

xmin = -5 ; xmax = 5 # m
tmin = 0 ; tmax = 5 # s
c = 1 # m/s


def plot_wave(x0 = 2, t0 = 3):

    fig = plt.figure()
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan = 2, rowspan = 2, projection = '3d')
    ax2 = plt.subplot2grid((3, 2), (2, 0))
    ax3 = plt.subplot2grid((3, 2), (2, 1))

    x = np.linspace(xmin, xmax, 50)
    t = np.linspace(tmin, tmax, 50)
    x_mesh, t_mesh = np.meshgrid(x, t)

    ax1.set_xlabel(r'$x$ (m)')
    ax1.set_ylabel(r'$t$ (s)')
    ax1.set_zlabel(r'$u_z(x, t)$ (m)')

    ax1.view_init(elev = 15, azim = 70)

    ax1.plot_surface(x_mesh, t_mesh, .5 * (f(x_mesh - c * t_mesh) + f(x_mesh + c * t_mesh)))

    ax1.plot([x0 for _ in range(len(t))], t, .5 * (f(x0 - c * t) + f(x0 + c * t)), 'r', lw = 2)

    ax1.plot(x, [t0 for _ in range(len(x))], .5 * (f(x - c * t0) + f(x + c * t0)), 'g', lw = 2)

    ax2.set_xlabel(r'$t$ (s)')
    ax2.set_ylabel(r'$u_z(x_0 = $' + '{0} m'.format(x0) + r'$, t)$ (m)')

    ax2.plot(t, .5 * (f(x0 - c * t) + f(x0 + c * t)), 'r')

    ax3.set_xlabel(r'$x$ (m)')
    ax3.set_ylabel(r'$u_z(x, t_0 = $' + '{0} s) (m)'.format(t0))
    ax3.yaxis.set_label_position('right')

    ax3.plot(x, .5 * (f(x - c * t0) + f(x + c * t0)), 'g')

    plt.show()
    
interact(plot_wave, x0 = (xmin, xmax, 0.5), t0 = (tmin, tmax, 0.5));
```


```python

```
