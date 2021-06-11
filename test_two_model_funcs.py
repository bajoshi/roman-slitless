import numpy as np

import os
import sys

import matplotlib.pyplot as plt

home = os.getenv('HOME')
roman_slitless_dir = home + '/Documents/GitHub/roman-slitless/'

sys.path.append(roman_slitless_dir)
from test_pylinear_extractions import model_sn
from gen_sed_lst import get_sn_spec_path

# some test values for both funcs
redshift = 1.5
day = 20
av = 1.2

# Run the first model func
snpath = get_sn_spec_path(redshift, day_chosen=day, chosen_av=av)
# read in provided sn path
model1 = np.genfromtxt(snpath, dtype=None, names=True, encoding='ascii')

# Get same model from the other model func
modellam = np.arange(10000,19300,10.0)
model2 = model_sn(modellam, redshift, day, av)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(model1['lam'], model1['flux'], label='Model func 1')
ax.plot(modellam, model2, label='Model func 2')

ax.legend(loc=0, frameon=False, fontsize=12)

ax.set_xlim(10000, 19300)

plt.show()