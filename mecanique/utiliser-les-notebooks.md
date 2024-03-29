<img src="media/CC-BY-NC-ND.png" alt="Drawing" style="width: 150px;"/> 

**Auteurs originaux** : Arnaud Legout et Thierry Parmentelat (auteurs du MOOC *Python 3 : des fondamentaux aux concepts avancés du langage* disponible sur la plateforme FUN).

**Modifié et adapté par** : Christophe Jorssen (avec l'autorisation des auteurs).

# Rôle de l'informatique en Physique

L'informatique occupe une place de plus en plus importante en Physique, que ce soit dans son enseignement ou dans la recherche. En effet, du point de vue expérimental, les mesures à traiter sont de plus en plus nombreuses et leur analyse requiert une puissance de stockage et de calcul auxquels les ordinateurs et langages contemporains donnent accès très simplement. Du point de vue théorique, la simulation numérique permet de résoudre des problèmes qui seraient partiellement ou totalement inaccessible en calculant « à la main ».

L'enseignement de l'informatique en tant que science (les anglosaxons parlent de _computer science_) se développe et l'apprentissage d'un langage de programmation est devenu un « passage obligé » dans l'enseignement supérieur scientifique.

Dans l'état actuel des choses, le langage scientifique « à la mode » est le langage [Python](https://www.python.org/). Son utilisation se développe dans le secondaire et il sert, en particulier, de support à l'enseignement de l'informatique en CPGE.

# L'informatique dans ce MOOC

Nous avons choisi, dans ce MOOC, de te faire découvrir (ou redécouvrir) Python en ciblant quelques utilisations possibles de ce langage en Physique. Faire un « cours de Python », ou plus généralement un cours d'informatique, serait totalement hors du champ de ce MOOC. Nous pensons néanmoins qu'il est important de mettre en évidence l'importance du recours à l'informatique en Physique. 

Alors comment va-t-on faire ? Imagine-toi à la place de quelqu'un qui arriverait dans un pays dont il ne connaît pas la langue. Au début, on ne comprend rien. On se débrouille, on fait des signes, on finit par identifier des mots ou des phrases qui reviennent sans cesse et dont on finit par _deviner_ le sens (en se trompant parfois). 

Ce quelqu'un (toi) sera pris en charge à son arrivée par un autre qui parle les deux langues (moi), qui prendra soin de te traduire _la plupart_ des mots de la langue que tu ignores (Python), qui t'expliquera un peu la grammaire (la syntaxe), qui t'expliquera les coutumes locales (les bonnes habitudes ou _best practice_) et qui te fournira, au cas où, un dictionnaire (la documentation).

En résumé : ne t'attends pas, à la fin du MOOC, à tout comprendre et à tout connaître au langage Python et à son utilisation en Physique. Tu auras appris quelques rudiments du langage et, surtout, tu t'en seras servi de manière concrète dans des situations où le recours à l'outil informatique est un réel plus.

# Jupyter

Pour utiliser Python, il faut disposer d'un certain nombre d'outils que tu peux installer sur ton ordinateur personnel. Par exemple, tu peux installer la distribution Python appelée [anaconda](https://www.anaconda.com/download/). Elle est muti-plateforme, ce qui signifie qu'elle peut être installée sur des ordinateurs équipés de windows, macos ou linux. Elle dispose des bibliothèques scientifiques nécessaires. Néanmoins, cela nécessite quelques connaissances et cela peut aboutir à quelques difficultés techniques que nous voulions absolument éviter.

Aussi avons-nous avons choisi d'illustrer le recours au langage Python dans ce MOOC en utilisant [Jupyter](http://jupyter.org/). Très schématiquement, cela permet de modifier et d'exécuter des scripts Python, **directement dans le navigateur, sans aucune installation de ta part**. Cela nous a semblé beaucoup plus simple et motivant.

# _Notebooks_

Pour pouvoir utiliser Jupyter, il faut rédiger des documents « mixtes », contenant du texte et du code Python, que l'on appelle des _notebooks_. Ces lignes, que tu es en train de lire, font partie d'un _notebook_.

Dans la suite de ce _notebook_, on va utiliser du code Python. Pas d'inquiétude si tu ne le comprends pas. Ce code est uniquement destiné à valider le fonctionnement des notebooks et tu ne vas utiliser que des choses très simples.

## Avantages des _notebooks_

Comme tu le vois, ce support permet un format plus lisible que des commentaires dans un fichier de code.

J'attire ton attention sur le fait que **les fragments de code peuvent être évalués et modifiés** (la page qui s'affiche dans ton navigateur n'est pas une page statique : elle est _dynamique_). Ainsi tu peux facilement essayer des variantes autour du _notebook_ original.

Note bien également que le code Python est interprété **sur une machine distante** (un serveur distant), ce qui te permet de faire tes premiers pas avant même d'avoir procédé à l'installation de Python sur ton propre ordinateur (c'est tout l'intérêt de la chose).

## Comment utiliser les _notebooks_

En haut du _notebook_, tu dois voir une barre, contenant&nbsp;:
* un titre pour le _notebook_, avec un numéro de version ;
* une barre de menus avec les entrées `File`, `Insert`, `Cell`, `Kernel`;
* et une barre de boutons qui sont des raccourcis vers certains menus fréquemment utilisés. Si tu laisses ta souris au dessus d'un bouton, un petit texte apparaît, indiquant à quelle fonction correspond ce bouton.

Un _notebook_ est constitué d'une suite de cellules, soit textuelles, soit contenant du code. Les cellules de code sont facilement reconnaissables, elles sont précédées de `In [ ]:`. La cellule qui suit celle que tu es en train de lire est une cellule de code.

Pour commencer, sélectionne la cellule de code ci-dessous avec ta souris et appuie dans la barre de boutons sur celui en forme de flèche triangulaire vers la droite (Run).


```python
20 * 30
```

Comme tu le vois, la cellule est « exécutée » (on dira plus volontiers « évaluée »), et on passe à la cellule suivante.

Alternativement tu peux simplement taper au clavier ***Shift+Enter***, ou, selon les claviers, ***Maj-Entrée***, pour obtenir le même effet (<kbd>SHIFT</kbd>+<kbd>ENTER</kbd> ou <kbd>MAJ</kbd>+<kbd>ENTRÉE</kbd>). D'une manière générale, il est important d'apprendre et d'utiliser les raccourcis clavier, cela te fera gagner beaucoup de temps par la suite.


La façon habituelle d'*exécuter* l'ensemble du notebook consiste à partir de la première cellule, et à taper  <kbd>SHIFT</kbd>+<kbd>ENTER</kbd> (ou <kbd>MAJ</kbd>+<kbd>ENTRÉE</kbd>) jusqu'au bout du notebook, en n'allant pas trop vite, c'est-à-dire en attendant le résultat de l'exécution de chaque cellule.

Lorsqu'une cellule de code a été évaluée, Jupyter ajoute sous la cellule `In` une cellule `Out` qui donne le résultat du fragment Python, soit ci-dessus 600.

Jupyter ajoute également un nombre entre les crochets pour afficher, par exemple ci-dessus, `In [1]:`. Ce nombre te permet de retrouver l'ordre dans lequel les cellules ont été évaluées.

Tu peux naturellement modifier ces cellules de code pour faire des essais. Ainsi, tu peux te servir du modèle ci-dessous pour calculer la racine carrée de 3, ou essayer la fonction sur un nombre négatif et voir comment est signalée l'erreur.


```python
# math.sqrt (pour square root) calcule la racine carrée
import math
math.sqrt(2)
```

Tu peux également évaluer tout le _notebook_ en une seule fois en utilisant le menu *Cell -> Run All*.

## Attention à bien évaluer les cellules dans l'ordre

Il est important que les cellules de code soient évaluées dans le bon ordre. Si tu ne respectes pas l'ordre dans lequel les cellules de code sont présentées, le résultat peut être inattendu.

En fait, évaluer un programme sous forme de notebook revient à le découper en petits fragments. Si on exécute ces fragments dans le désordre, on obtient naturellement un programme différent.

On le voit sur cet exemple :


```python
message = "Il faut faire attention à l'ordre dans lequel on évalue les notebooks"
```


```python
print(message)
```

Si un peu plus loin dans le notebook on fait par exemple :


```python
# ceci a pour effet d'effacer la variable 'message'
del message
```

qui rend le symbole `message` indéfini, alors, bien sûr, tu ne peux plus évaluer la cellule qui fait `print` puisque la variable `message` n'est plus connue de l'interpréteur.

## Réinitialiser l'interpréteur

Si tu fais trop de modifications, ou si tu perds le fil de ce que tu as évalué, il peut être utile de redémarrer ton interpréteur. Le menu *Kernel → Restart* te permet de faire cela.

Le menu *Kernel → Interrupt* peut être quant à lui utilisé si ton fragment prend trop longtemps à s'exécuter (par exemple tu as écrit une boucle dont la logique est cassée et qui ne termine pas).

## Tu travailles sur une copie

Un des avantages principaux des _notebooks_ est de te permettre de modifier le code que j'ai écrit et de voir, par toi-même, comment se comporte le code modifié.

Pour cette raison, chaque participant dispose de sa **propre copie** de chaque _notebook_. Tu peux bien sûr apporter toutes les modifications que tu souhaites à tes notebooks sans affecter les autres étudiants.

## Revenir à la version du cours

Tu peux toujours revenir à la version « du cours » grâce au menu
*File → Reset to original*.

Attention, avec cette fonction, tu restaures **tout le _notebook_** et donc **tu perds tes modifications sur ce _notebook_**.

## Télécharger au format Python

Tu peux télécharger un _notebook_ au format Python sur ton ordinateur grâce au menu
*File → Download as → Python*

Les cellules de texte sont préservées dans le résultat sous forme de commentaires Python.

## Partager un _notebook_ en lecture seule

Enfin, avec le menu *File → Share static version*, tu peux publier une version en lecture seule de ton _notebook_ ; tu obtiens une URL que tu peux publier par exemple pour demander de l'aide sur le forum. Ainsi, les autres participants peuvent accéder en lecture seule à ton code.

Note que lorsque tu utilises cette fonction plusieurs fois, c'est toujours la dernière version publiée que verront les participants, l'URL utilisée reste toujours la même pour un étudiant et un _notebook_ donné.

## Ajouter des cellules

Tu poux ajouter une cellule n'importe où dans le document avec le bouton **+** de la barre de boutons.

Lorsque tu arrives à la fin du _notebook_, une nouvelle cellule est créée chaque fois que tu évalues la dernière cellule. De cette façon, tu disposes d'un brouillon pour tes propres essais.

À toi de jouer !

# En cas de problème...

Encore une chose... Si tout ne se passe pas comme prévu ou si tu rencontres des difficultés, n'hésite pas à utiliser le forum de discussion. 

Attention néanmoins... Pour que les participants du forum puissent t'aider à résoudre ton problème, il faut que tu le décrives correctement, d'abord avec des mots (évite des messages du style « Ça marche pas. ») et éventuellement avec des images (parfois, cela vaut mieux qu'un long discours). 

Aussi, je t'encourage vivement à être le plus précis possible dans la description de ton problème et à joindre une ou plusieures captures d'écran qui te permette de l'illustrer.


```python

```
