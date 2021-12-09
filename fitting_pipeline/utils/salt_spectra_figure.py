import numpy as np
import matplotlib.pyplot as plt

# Read in SALT2 SN IA file  from Lou
salt2_spec = np.genfromtxt("templates/salt2_template_0.txt", 
    dtype=None, names=['day', 'lam', 'llam'], encoding='ascii')

day_arr = np.arange(-10, 21, 5)

# Get colors from some chosen colormap
col = plt.cm.viridis(np.linspace(0,1, len(day_arr)))

# Plot
fig = plt.figure(figsize=(9,5))
ax = fig.add_subplot(111)

ax.set_xlabel('Wavelength [A]', fontsize=15)
ax.set_ylabel('Normalized Flux', fontsize=15)

for d in range(len(day_arr)):

    # pull out spec
    day_idx = np.where(salt2_spec['day'] == day_arr[d])
    w = salt2_spec['lam'][day_idx]
    f = salt2_spec['llam'][day_idx]

    # Normalize 

    # plot 
    ax.plot(w, f, color=col[d], label='Phase: ' + str(day_arr[d]))

# Set limit to prism coverage
ax.set_xlim(4000, 20000)

ax.legend(loc=0, frameon=False)
plt.show()