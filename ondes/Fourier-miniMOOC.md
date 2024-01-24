<img src="media/CC-BY-NC-ND.png" alt="Drawing" style="width: 150px;"/> 

**Auteur** : Christophe Jorssen

Dans ce *notebook*, je te propose de travailler de manière interactive sur la dernière figure que je t'ai présentée dans la vidéo. L'objectif ici n'est pas de comprendre en détail le code mais bien de « jouer » avec la figure interactive.

On commence par importer un certain nombre de bibliothèques.


```python
%matplotlib inline
from ipywidgets import interact
import numpy as np
from scipy.signal import square, sawtooth
from scipy.integrate import quad
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d
```

On définit ensuite la figure interactive. Tu peux choisir :
* la nature de la fonction `func` (cosinus, cosinus redressé, carré, dents de scie ou triangle) ;
* l'amplitude `A` ;
* la fréquence `f1` en hertz de la fonction (qui est aussi la fréquence du fondamental) ;
* la phase à l'origine des temps `phi` (en degrés) ;
* le rapport cyclique `alpha` ;
* l'offset `offset` ;
* le rang du dernier harmonique considéré `N`.

`func`, `A`, `f1`, `alpha` et `offset` sont typiquement des réglages que tu trouverais sur un générateur basses fréquences (GBF). L'amplitude et l'offset auraient alors la dimension d'une tension.

La courbe en vert est la fonction et la courbe en bleu est la fonction approchée par sa décomposition de Fourier limitée aux `N` premiers harmoniques.

Rappelle-toi que tu peux faire pivoter la figure en trois dimensions. 

Ton objectif ici est de bien comprendre les deux façons équivalentes de décrire un signal périodique :
* une description temporelle par la donnée explicite des variations temporelles de la fonction ;
* une description fréquentielle par la donnée des amplitudes et phases de chaque harmonique.


```python
tmin = 0 ; tmax = 1 # s
fmin = 0 ; fmax = 60 # Hz

def define_Fourier_expansion(func, A, f1, phi, alpha, offset):
    global a, b, c, arg
    def a(i):
        T = 1 / f1
        tmp = quad(lambda t:2 * func(t, A, f1, phi, alpha, offset) * np.cos(2 * np.pi * i * f1 * t) / T, -T/2, T/2)[0]
        if i == 0:
            return tmp / 2
        else:
            return tmp
    def b(i):
        if i == 0:
            return 0
        T = 1 / f1
        return quad(lambda t:2 * func(t, A, f1, phi, alpha, offset) * np.sin(2 * np.pi * i * f1 * t) / T, -T/2, T/2)[0]
    def c(i):
        return np.sqrt(a(i)**2 + b(i)**2)
    def arg(i):
        return np.arctan2(b(i), a(i)) * 180 / np.pi
    

def cosinus(t, A, f1, phi, alpha, offset):
    phi = phi * np.pi / 180
    return offset + A * np.cos(2 * np.pi * f1 * t + phi)

def cosinus_redresse(t, A, f1, phi, alpha, offset):
    return np.abs(cosinus(t, A, f1, phi, alpha, offset))

def carre(t, A, f1, phi, alpha, offset):
    phi = phi * np.pi / 180
    return offset + A * square(2 * np.pi * f1 * t + phi, alpha)

def dents_de_scie(t, A, f1, phi, alpha, offset):
    phi = phi * np.pi / 180
    return offset + A * sawtooth(2 * np.pi * f1 * t + phi, alpha)

def triangle(t, A, f1, phi, alpha, offset):
    phi = phi * np.pi / 180
    return offset + A * sawtooth(2 * np.pi * f1 * t + phi, .5)

def Fourier(func = carre, A = 1, f1 = 10, phi = 0, alpha = 0.5, offset = 1, N = 10):
    
    fmax = N * f1
    
    define_Fourier_expansion(func, A, f1, phi, alpha, offset)
    
    fig = plt.figure()
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan = 2, rowspan = 2, projection = '3d')
    ax2 = plt.subplot2grid((3, 2), (2, 0))
    ax3 = plt.subplot2grid((3, 2), (2, 1))

    t = np.linspace(tmin, tmax, 1000)
    f = np.linspace(fmin, fmax, 1000)
    t_mesh, f_mesh = np.meshgrid(t, f)

    ax1.set_xlabel(r'$t$ (s)')
    ax1.set_ylabel(r'$f$ (Hz)')
    ax1.set_ylim(fmin, fmax)
    ax1.set_zlabel('Amplitude')

    #ax1.view_init(elev = 15, azim = 70)

    ax1.plot(t, [-10 for _ in range(len(t))], func(t, A, f1, phi, alpha, offset), 'g', lw = 2)
    
    for i in range(N + 1):
        ax1.plot(t, [i * f1 for _ in range(len(t))], 
                     a(i) * np.cos(2 * np.pi * i * f1 * t) + b(i) * np.sin(2 * np.pi * i * f1 * t),
                'orange')
    
    for i in range(N + 1):
        ax1.plot([tmax, tmax], [i * f1, i * f1], [0, c(i)], 'r')

    ax2.set_xlabel(r'$t$ (s)')
    ax2.set_ylabel('Amplitude')

    ax2.grid(which = 'both')
    
    ax2.plot(t, func(t, A, f1, phi, alpha, offset), 'g')        
    
    func_Fourier = np.zeros(t.size)
    
    for i in range(N + 1):
        func_Fourier += a(i) * np.cos(2 * np.pi * i * f1 * t) + b(i) * np.sin(2 * np.pi * i * f1 * t)

    ax2.plot(t, func_Fourier, 'b')

    ax3.set_xlabel(r'$f$ (Hz)')
    ax3.set_ylabel('Amplitude')
    ax3.yaxis.set_label_position('right')
    
    ax3.grid(which = 'both')
    
    ax3.set_xlim(fmin, fmax)

    ax3.vlines([i * f1 for i in range(N + 1)], 
               [0], 
               [c(i) for i in range(N + 1)], 
               'r')

    plt.show()
    
    
interact(Fourier, 
         func = {'cosinus': cosinus, 
                 'cosinus redressé': cosinus_redresse,
                 'carré': carre, 
                 'dents de scie': dents_de_scie, 
                 'triangle': triangle},
         A = (0, 1, .1),
         f1 = (1, 3, 1), 
         phi = (-180, 180, 10), 
         alpha = (0, 1, .1), 
         offset = (-2, 2, .1), 
         N = (0, 20, 1));
```


```python

```
