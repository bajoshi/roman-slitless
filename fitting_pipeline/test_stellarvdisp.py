import numpy as np
import matplotlib.pyplot as plt
from fit_galaxy import model_galaxy

import matplotlib

matplotlib.rcParams['text.usetex'] = False

x = np.arange(7500, 18000, 10.0)
mv = model_galaxy(x, 0.75, 10.5, 3.0, np.log10(1.0), 0.2)
m  = model_galaxy(x, 0.75, 10.5, 3.0, np.log10(1.0), 0.2, stellar_vdisp=False)

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111)

ax.plot(x, m, color='mediumblue')
ax.plot(x, mv, color='tab:red')

plt.show()