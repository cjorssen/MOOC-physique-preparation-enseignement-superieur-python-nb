<img src="media/CC-BY-NC-ND.png" alt="Drawing" style="width: 150px;"/> 

**Auteur** : Christophe Jorssen


```python
%matplotlib inline
import numpy as np
from scipy.stats import linregress
from matplotlib import pyplot as plt

x = [100, 200, 300] # m
Delta_t = [480E-9, 1000E-9, 1520E-9] # s

slope, intersept, R, foo, bar = linregress(Delta_t, x)

fig, ax = plt.subplots()

ax.plot(Delta_t, x, '+')
ax.plot([Delta_t[0], Delta_t[-1]], [slope * Delta_t[0] + intersept, slope * Delta_t[-1] + intersept])
fig.show()
```


```python
print('celerité = {0:1.2e} m/s'.format(slope))
```


```python

```
