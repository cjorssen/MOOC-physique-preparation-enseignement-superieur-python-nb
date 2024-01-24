<img src="media/CC-BY-NC-ND.png" alt="Drawing" style="width: 150px;"/> 

**Auteur** : Christophe Jorssen

Pour commencer, il faut [télécharger la vidéo](https://youtu.be/BWraEDaVXZM) produite par le _Technical Services Group_ du département de physique du [MIT](https://web.mit.edu/). Je ne peux pas te dire _comment_ télécharger cette vidéo, mais tu trouveras sans doute !

Il faut ensuite extraire les images qui nous intéressent. Pour cela, j'ai utilisé la boîte à outils [`FFMpeg`](https://www.ffmpeg.org/) qui fournit tout ce dont on a besoin pour manipuler une vidéo. `FFMpeg` présente en outre l'avantage d'être multi-plateforme, ce qui signifie que ces outils fonctionnent sous linux, macos ou windows. Un certain nombre de ces images sont stockées sur le serveur qui héberge ce _notebook_ dans le répertoire `media/echelle-perroquet/`. Elles sont nommées `TSG-perroquet000034.jpg`, `TSG-perroquet000035.jpg`, ..., `TSG-perroquet000056.jpg`. Ce sont des images successives extraites de la vidéo.

Nous allons avoir besoin d'un certain nombre de bibliothèques. Comme d'habitude, on utilise `numpy` et `matplotlib`. On aura aussi besoin du paquet [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html) de la bibliothèque `scipy` qui permettra d'accéder à [`linregress`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html#scipy.stats.linregress) afin de réaliser une régression linéaire.


```python
%matplotlib inline
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.stats import linregress
```

Pour réaliser le traitement des images, nous allons nous servir d'une nouvelle bibliothèque : [`skimage`](http://scikit-image.org/docs/stable/). 

* Dans le paquet [`skimage.io`](http://scikit-image.org/docs/stable/api/skimage.io.html), on se servira, d'une part, d'[`imread`](http://scikit-image.org/docs/stable/api/skimage.io.html#skimage.io.imread) (_image read_) qui permet de lire un fichier image et de stocker son contenu dans une variable et, d'autre part, d'[`imshow`](http://scikit-image.org/docs/stable/api/skimage.io.html#skimage.io.imshow) sui permet d'afficher une image.
* Dans le paquet [`skimage.color`](http://scikit-image.org/docs/stable/api/skimage.color.html), on se servira de [`rgb2gray`](http://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.rgb2gray) qui permet d'obtenir la luminance associée à une image.
* Dans le paquet [`skimage.feature`](http://scikit-image.org/docs/stable/api/skimage.feature.html), on se servira de [`blob_doh`](http://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_doh) qui permet de détecter la position de « points lumineux » dans une image étant donné sa luminance.


```python
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.feature import blob_doh
```

Je te propose dans un premier temps de visualiser une image « brute ». Par exemple, affichons la première de la série.

La prochaine cellule « stocke » l'image dans le `ndarray` `image`.


```python
image = imread('media/echelle-perroquet/TSG-perroquet000034.jpg')
```

L'attribut `shape` permet de connaître la forme, c'est-à-dire la dimension, du `ndarray`.


```python
image.shape
```

On constate qu'il s'agit d'une image de `720` pixels en hauteur sur `1280` pixels en largeur. Note que c'est d'abord l'ordonnée *puis* l'abscisse. 

Chaque pixel possède trois informations correspondant à la couleur : une information pour le rouge (_red_), une pour le vert (_green_) et une pour le bleu (_blue_). C'est le codage dit `rgb` de la couleur.

Ensuite, on définit une figure `fig` et un système d'axes (abscisses et ordonnées) _via_ la commande `subplots` de `pyplot` (tu te rappelles : `plt` est l'alias de `pyplot`). Une fois le système d'axes défini, on peut y afficher l'image `image` à l'aide d'`imshow`.


```python
fig, ax = plt.subplots()
ax.imshow(image, interpolation = 'nearest')
```

Et voilà ! Bon, on voit que toute l'image n'est pas utile pour l'exploitation de l'expérience. On va donc ne s'intéresser qu'à une portion de l'image. J'ai choisi, un peu arbitrairement, les pixels compris, en ordonnées, entre les indices `280` et `550` (non compris) et, en abscisses, entre les indices `150` et `1150` (non compris). C'est le sens de la notation `image[280:550, 150:1150, :]`. Le dernier `:` sans « bornes » signifie que l'on prend toute l'information de cette dimension du `ndarray`, c'est-à-dire toute l'information de couleur pour chaque pixel.


```python
image2 = image[280:550, 150:1150, :]
```

Affichons-la.


```python
fig2, ax2 = plt.subplots()
ax2.imshow(image2, interpolation = 'nearest')
```

Détectons les points lumineux à l'aide de `blob_doh` de la bibliothèque `skimage`. Je ne m'en sers pas tous les jours de cette bibliothèque. Je me suis donc nettement inspiré de ce que tu peux trouver dans la [documentation](http://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html). On commence par convertir l'image en noir et blanc (`rgb2gray`), puis on utilise `blob_doh`. Pour être tout à fait honnête, les valeurs données à `max_sigma` et à `threshold` ont été trouvées par une méthode rudimentaire d'essais/erreurs. Si tu veux, tu peux les modifier pour voir ce que ça change.


```python
image2_gray = rgb2gray(image2)
blob = blob_doh(image2_gray, max_sigma = 30, threshold = .005)
```

Le résultat de la recherche est stockée dans `blob`. Il s'agit d'un `ndarray` contenant trois informations par points lumineux détectés : son ordonnée, son abscisse et son rayon que l'on récupère respectivement dans `x`, `y` et `r`. 


```python
x = blob[:, 1]
y = blob[:, 0]
r = blob[:, 2]
```

Pour mettre en évidence les points lumineux détectés, on va les entourer par des cercles rouges. Exécute la cellule suivante et regarde comment cela a modifié l'image que tu as affichée ci-dessus dans la figure `fig2`.


```python
for i in range(len(y)):
    c = plt.Circle((x[i], y[i]), r[i], color = '#CE181E', linewidth = 2, fill = False)
    ax2.add_patch(c)
```

Remonte un peu dans cette page et observe les disques rouges qui sont apparus sur l'image de la figure `fig2`.

Une fois qu'on a bien vérifié que la détection automatique était (à peu près) conforme à ce à quoi on pouvait s'attendre, on peut se passer de l'image et faire une représentation graphique « classique » présentant l'abscisse et l'ordonnée de chaque point lumineux détecté.

`x0`et `y0` sont les indices du pixel qui correspond à l'origine choisie (position horizontale de l'échelle et position initiale du « pic »).

`scale` est le facteur d'échelle qui permet de passer d'un nombre de pixels à une distance (l'échelle de perroquet mesure 90 cm et occupe 950 pixels).

Notons que l'axe des ordonnées de l'image est orienté selon la verticale descendante. On prend l'opposé des ordonnées pour travailler avec un axe des ordonnées selon la verticale ascendante.


```python
fig3, ax3 = plt.subplots()
ax3.grid(True)

scale = 90e-2/950
y0 = 140
# xoffset = 24
x0 = 475 + 24

scat = ax3.scatter((x - x0) * scale, (-(y - y0)) * scale, color = '#CE181E')

ax3.set_xlabel(r'$x$ (m)')
ax3.set_ylabel(r'$u_z(x, t)$ (m)')
```

Maintenant que l'on a fait cette démarche pour une image, il suffit de recommencer pour les autres. Une structure de données adaptée à cette sitation est la `list`. On va donc construire cette `list` en reprenant et adaptant légèrement ce qui précède. 

On commence par initialiser la `list` `blobs` à la `list` vide `[]`. Cette `list` contiendra les caractéristiques des points lumineux détectés (`y`, `x` et `r`).


```python
blobs = []
```

À l'aide d'une boucle `for`, on parcourt les entiers compris entre `34` et `56` à l'aide de l'itérateur `range`. Le corps de la boucle `for` commence par les `:` à la fin de la ligne présentant le mot-clé `for` et s'achève lorsque l'indentation redevient identique à celle de la ligne présentant le mot-clé `for`. À noter l'utilisation de la méthode [`format`](https://docs.python.org/3/library/string.html#format-string-syntax) qui s'applique à une objet de type `str` (chaîne de caractères ou _string_). Le `{0}` est remplacé par la valeur de `i` dans la chaîne de caractère. La méthode `append` ajoute un élément à la `list`.


```python
for i in range(34, 57):
    image = imread('../media/echelle-perroquet/TSG-perroquet0000{0}.jpg'.format(i))
    image = image[280:550, 150:1150, :]
    image_gray = rgb2gray(image)

    blobs.append(blob_doh(image_gray, max_sigma = 30, threshold = .005))
```

Les lignes qui suivent servent à créer une animation à l'aide de `matplotlib`. C'est un peu technique, je ne te détaille pas tout. Tu trouveras [ici](https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/) ou [là](https://brushingupscience.com/2016/06/21/matplotlib-animations-the-easy-way/) des tutoriels qui complèteront ce que je te propose ci-dessous.


```python
fig4, ax4 = plt.subplots()
ax4.grid(True)

Delta_t = 33.3 # ms (29,97 images par seconde)

scat = ax4.scatter((blobs[0][:,1] - x0) * scale, (-(blobs[0][:,0] - y0)) * scale, color = '#CE181E')

ax4.set_ylim([-0.05, 0.15])

ax4.set_xlabel(r'$x$ (m)')
ax4.set_ylabel(r'$u_z(x, t)$ (m)')

title = ax4.text(0.5, 1.1, "", transform = ax4.transAxes, ha = "center")

def init_fig4():
    return scat, 

def animate_fig4(i):
    data = np.hstack(((blobs[i][:, 1, np.newaxis] - x0) * scale,
                      (-(blobs[i][:, 0, np.newaxis] - y0)) * scale))
    
    scat.set_offsets(data)
    
    title.set_text('t = {:10.1f} (ms)'.format(i * Delta_t))

    return scat,

anim4 = animation.FuncAnimation(fig4, animate_fig4, init_func = init_fig4, frames = len(blobs), 
                                interval = 200, blit = False, repeat = True)
```

On va maintenant se concentrer sur le déplacement des maxima. 

Initialisons les `list`s `xtop_right` et `xtop_left` dont les éléments vont être respectivement les abscisses des « pics » se déplaçant vers la droite et vers la gauche.


```python
xtop_right = []
xtop_left = []
```

Initialisons également une `list` appelée `t` dont les éléments seront les dates.


```python
t = []
```

À l'aide d'une boucle `for` parcourant l'ensemble des éléments de la `list` `blobs`, on va chercher l'évolution au cours du temps de l'abscisses des deux « pics ».

`for i, blob in enumerate(blobs)` parcourt les éléments de `blobs` et stocke la valeur de l'indice de l'élément courant dans `i` et l'élément courant dans `blob`. Chaque élément est séparé dans le temps d'un intervalle `Delta_t`.

La suite utilise le fait que `blob` est un `ndarray` défini par la bibliothèque `numpy`.

`blob[:,1]` correspond à toutes les abscisses des points lumineux. Parmi celles-ci, on ne s'intéresse qu'à celles qui sont dans la partie droite de l'échelle de perroquet, centrée en `x0`. Cela correspond au test : `blob[:,1] > x0`.

Comme il s'agit d'un `ndarray`, on peut passer ce test comme s'il s'agit d'un intervalle d'indices : `blob[blob[:,1] > x0, 0]` s'interprète comme toutes les ordonnées (`0`) des points lumineux dont l'abscisse est plus grande que `x0`.

Parmi ces ordonnées, on cherche la plus petite, c'est-à-dire celle qui correspond au « pic » (je te rappelle que l'axe des ordonnées est vertical descendant). On utilise alors `argmin` de la bibliothèque `numpy` qui donne accès à l'indice vérifiant ces deux conditions : on le stocke dans `index_right`.

On ajoute l'abscisse trouvée à la `list` `xtop_right`, en oubliant pas l'échelle.

On procède de même pour la partie gauche.


```python
for i, blob in enumerate(blobs):
    t.append(i * Delta_t * 1e-3) # s

    index_right = np.argmin(blob[blob[:,1] > x0, 0])
    xtop_right.append((blob[blob[:,1] > x0, 1][index_right] - x0) * scale)

    index_left = np.argmin(blob[blob[:,1] < x0, 0])
    xtop_left.append((blob[blob[:,1] < x0, 1][index_left] - x0) * scale)
```

Traçons les nuages de points obtenus.


```python
fig5, (ax5a, ax5b) = plt.subplots(1, 2, sharey = True)

ax5a.set_xlabel(r'$t$ (s)')
ax5a.set_ylabel(r'$x_{\max}$ (m)')
ax5a.grid(True)

ax5a.plot(t, xtop_right, '+', color = '#CE181E')

ax5b.set_xlabel(r'$t$ (s)')
ax5b.grid(True)

ax5b.plot(t, xtop_left, '+', color = '#CE181E')
```

On peut ensuite procéder à la régression linéaire. Pour cela, on se sert de [`linregress`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html) de la bibliothèque `scipy`. Elle prend en argument les abscisses et ordonnées du nuage de points. Elle renvoie un certain nombre de valeurs dont seules trois nous intéressent ici : la pente ou coefficient directeur (*slope* en anglais), l'ordonnée à l'origine (*intercept* en anglais) et le coefficient de corrélation $R$.


```python
slope_right, intercept_right, R_right, foo, bar = linregress(t, xtop_right)
slope_left, intercept_left, R_left, foo, bar = linregress(t, xtop_left)
print('c (droite) = {0:1.2e} m/s et c(gauche) = {1:1.2e} m/s'.format(slope_right, slope_left))
```

On peut alors tracer sur les figures ci-dessus les droites de regression obtenues.


```python
ax5a.plot([t[0], t[-1]], [slope_right * t[0] + intercept_right, slope_right * t[-1] + intercept_right], color = '#FFCC00')
ax5b.plot([t[0], t[-1]], [slope_left * t[0] + intercept_left, slope_left * t[-1] + intercept_left], color = '#FFCC00')
```

Et voilà !

Dernier commentaire. Dans le secondaire, tu as sans doute pris l'habitude de juger de la « qualité » d'une régression linéaire à l'aide du coefficient de corrélation $R^2$. Sache que ce n'est [pas toujours une bonne idée](https://stats.stackexchange.com/q/13314/16275). Juste pour information, on a ici :


```python
R_left**2, R_right**2
```


```python

```
