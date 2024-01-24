<img src="media/CC-BY-NC-ND.png" alt="Drawing" style="width: 150px;"/> 

**Auteur** : Christophe Jorssen

Dans ce *notebook*, je te propose de visualiser simultanément la double dépendance en $x$ et en $t$ du champ de déplacement $u_{z, \nearrow}(x, t) = A \cos\left(2 \pi \frac{x}{\lambda} - 2 \pi f t + \varphi \right)$ correspondant à une onde progressive harmonique se propageant selon les $x$ croissant.


```python
%matplotlib inline
from ipywidgets import interact
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d
```

Comme nous l'avons déjà fait précédemment, la cellule ci-dessous va définir une figure interactive. Trois représentations graphiques vont apparaître : 

* la surface $u_{z, \nearrow}(x, t)$ pour $x \in [x_{\min}, x_{\max}]$ et $t \in [t_{\min}, t_{\max}]$ ;
* l'évolution au cours du temps du champ de déplacement $u_{z, \nearrow}(x_0, t)$ au point d'abscisse $x_0$ de la corde ;
* la forme de la corde à la date $t_0$, c'est-à-dire l'évolution du champ de déplacement $u_{z, \nearrow}(x, t_0)$ pour $x \in [x_{\min}, x_{\max}]$.

Tu n'as qu'à exécuter la cellule ci-dessous et à « jouer » avec les paramètres $x_0$, $t_0$, $\lambda$ et $\varphi$. N'oublie pas que $\lambda$ et $f$ ne sont pas des grandeurs indépendantes. Elles sont liées par la relation de dispersion $\lambda f = c$. C'est la raison pour laquelle tu ne peux pas modifier $\lambda$ et $f$ séparément à célérité $c$ fixée.

Note technique : `lambda` est un mot-clé du language Python, c'est la raison pour laquelle j'appelle `Lambda` la longueur d'onde $\lambda$.


```python
A = 1 # m

def f(x, t, Lambda, phi):
    # Lambda (m)
    # f (Hz)
    # phi (°)
    
    f = c / Lambda
    
    return A * np.cos(2 * np.pi * (x / Lambda) - 2 * np.pi * f * t + phi * (np.pi / 180))

xmin = -5 ; xmax = 5 # m
tmin = 0 ; tmax = 5 # s
c = 1 # m/s


def plot_wave(x0 = 2, t0 = 3, Lambda = xmax, phi = 0):
    
    fig = plt.figure()
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan = 2, rowspan = 2, projection = '3d')
    ax2 = plt.subplot2grid((3, 2), (2, 0))
    ax3 = plt.subplot2grid((3, 2), (2, 1))

    x = np.linspace(xmin, xmax, 100)
    t = np.linspace(tmin, tmax, 100)
    x_mesh, t_mesh = np.meshgrid(x, t)

    ax1.set_xlabel(r'$x$ (m)')
    ax1.set_ylabel(r'$t$ (s)')
    ax1.set_zlabel(r'$u_{z, \nearrow}(x, t)$ (m)')

    ax1.view_init(elev = 15, azim = 70)

    ax1.plot_surface(x_mesh, t_mesh, f(x_mesh, t_mesh, Lambda, phi))

    ax1.plot([x0 for _ in range(len(t))], t, f(x0, t, Lambda, phi), 'r', lw = 2)

    ax1.plot(x, [t0 for _ in range(len(x))], f(x, t0, Lambda, phi), 'g', lw = 2)

    ax2.set_xlabel(r'$t$ (s)')
    ax2.set_ylabel(r'$u_{z, \nearrow}(x_0 = $' + '{0} m'.format(x0) + r'$, t)$ (m)')

    ax2.plot(t, f(x0, t, Lambda, phi), 'r')

    ax3.set_xlabel(r'$x$ (m)')
    ax3.set_ylabel(r'$u_{z, \nearrow}(x, t_0 = $' + '{0} s) (m)'.format(t0))
    ax3.yaxis.set_label_position('right')

    ax3.plot(x, f(x, t0, Lambda, phi), 'g')

    plt.show()
    
interact(plot_wave, x0 = (xmin, xmax, 0.5), t0 = (tmin, tmax, 0.5), Lambda = (0, xmax, .5), phi = (-180, 180, 10));
```


```python

```
