<img src="media/CC-BY-NC-ND.png" alt="Drawing" style="width: 150px;"/> 

**Auteur** : Christophe Jorssen

Observe la capture d'écran ci-dessous.

![ping](media/ping.png)

Il s'agit d'une capture d'écran réalisée à partir du terminal de mon ordinateur (je travaille sous [linux](https://www.archlinux.org/), mais tu peux reproduire la même chose sous windows ou macos). Tu peux y trouver le résultat de deux commandes  :

* `ping -c1 princeton.edu` ;
* `traceroute princeton.edu`.

Petite remarque au passage, pour être tout à fait clair : ces deux commandes n'ont rien à voir avec Python ! Mais patience, ça va venir.

La commande `ping -c1 princeton.edu` envoie un paquet d'onde électrique depuis mon ordinateur, situé au [lycée Jacques Decour](http://lyc-jacques-decour.scola.ac-paris.fr/v2/) à [Paris](https://goo.gl/maps/2T7k4EmrHMP2), vers un serveur de l'[université étatsunienne Princeton](https://www.princeton.edu/) située dans l'état du New Jersey, [pas très loin de New York](https://goo.gl/maps/WrfiXPr9zWq). L'affichage nous informe du fait que le paquet est bien arrivé et qu'il a mis $73,6\,\mathrm{ms}$ pour effectuer le trajet. On pourrait penser qu'il s'agit du retard dû à la propagation.

En fait, c'est un peu plus complexe que cela en raison de la structure de l'internet. Pour avoir une idée plus fine, on utilise la seconde commande : `traceroute princeton.edu`. Cette commande montre que le paquet d'onde voyage à travers différents « relais » (17 au total). Le transit au sein de ces relais peut prendre un peu de temps et donc augmenter artificiellement le retard dû à la propagation.

Si tu regardes attentivement ce que la commande `traceroute` renvoie, tu peux constater que l'essentiel de la durée de propagation se trouve entre le relais 13 `renater-lb1.mx1.par.fr.geant.net` qui se trouve à Paris (`par.fr`) et le relais 14 `et-3-1-0.102.rtsw.newy32aoa.net.internet2.edu` qui se trouve à New York (`newy`). Le paquet d'onde met donc une durée moyenne égale à $72,482 - 1,793 = 70,689\,\mathrm{ms}$ pour se propager depuis Paris jusqu'à New York.

Bon, je sens que tu as envie de faire un peu de Python ! Allons-y !

Je te propose d'utiliser la bibliothèque Python [`geopy`](https://geopy.readthedocs.io/en/stable/) qui permet d'extraire tout un tas d'information de géolocalisation. Pour être précis, `geopy` n'est qu'une interface entre Python et des fournisseurs d'information de géolocalisation comme GoogleMaps. J'ai fait le choix d'utiliser [Nominatim](https://nominatim.openstreetmap.org/), le service fourni par [OpenStreetMap](https://www.openstreetmap.org).


```python
from geopy.geocoders import Nominatim
```

Mais pourquoi au juste utilise-t-on geopy ici ? Quel est le rapport avec le retard dû à la propagation ? En fait, je te propose de vérifier que le paquet d'onde électrique se propage avec une célérité proche de celle de la lumière. Pour cela, il nous faut la distance qui sépare les relais 13 et 14 que nous avons identifiés précédemment. Importons ce qui nous sera nécessaire.


```python
from geopy.distance import vincenty
```

On définit les emplacements (approximatifs) des relais.


```python
geolocator = Nominatim()

relais13 = geolocator.geocode("Paris France")
relais14 = geolocator.geocode("New York USA")
```

On calcule la distance « à vol d'oiseau » que l'on convertit en mètres.


```python
distance = vincenty((relais13.latitude, relais13.longitude), 
                    (relais14.latitude, relais14.longitude)).kilometers * 1E3 # m
```

On calcule la célérité en exploitant le retard mesuré grâce à la commande `traceroute`.


```python
retard = 70.689E-3 # s

celerite = distance / retard
```

On peut dès lors afficher les résultats. Note l'utilisation de [`format`](https://docs.python.org/3/library/string.html#format-specification-mini-language) qui d'utiliser des « gommettes » (`{0}` ou `{1}`) à l'intérieur d'une chaîne de caractères et d'en contrôler le format (`:.3e`). Par exemple `{0:.3e}` sera remplacé dans la chaîne de caractères par le premier argument (d'indice `0`) de `format` en utilisant la notation scientifique (`e`) en laissant trois chiffres après la virgule (`.3`).


```python
print('distance = {0:.3e} m, celerite = {1:.3e} m/s'.format(distance, celerite))
```

On constate que la célérité à laquelle se propage le paquet d'onde électrique est bien voisine de celle de la lumière dans le vide. En outre, la distance que nous avons utilisée pour le calcul est très grossièrement estimée puisqu'il s'agit d'une distance « à vol d'oiseau ».

Au fait, tu peux te demander pourquoi je n'ai pas envoyé un paquet d'onde électrique au MIT ou à l'université d'Auckland. C'est parce que ces universités ont des serveurs délocalisés afin de minimiser leur temps de réponse. Quand on se trouve en Europe, c'est un serveur européen qui répond à la commande `ping`. Du coup, le retard dû à la propagation est trop petit pour être exploitable.
