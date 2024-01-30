import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from tueplots import bundles

current_dir = os.getcwd()
src_directory = current_dir + "/../../src/"
if src_directory not in sys.path:
    sys.path.append(src_directory)
from DataLoaderClass import DataLoader

dataloader = DataLoader()

# with scaled size and dropped last duration
data_array, years, durations, kids = dataloader.load_data(durations_to_drop=[26])
print(np.shape(data_array))

#plt.rcParams.update(bundles.icml2022(column="full", nrows=1, ncols=1, usetex=False))

x_val = []
y_val = []
z_val = []

count_val = []

for year in range(0, len(data_array)):
    for duration in range(0, len(data_array[year])):
        for kid in range(0, len(data_array[year][duration])):
            x_val.append(year)
            y_val.append(duration)
            z_val.append(kid)
            count_val.append(data_array[year][duration][kid])

# Create a 3D scatter plot
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=[10, 8])
sc = ax.scatter(np.array(x_val) + 1997, np.array(y_val) + 1,  z_val, c=np.array(count_val), cmap='viridis',
                s=np.array(count_val) / 6, alpha=0.8, edgecolors="black")

# Add colorbar
cbar = plt.colorbar(sc, ax=ax, shrink=0.8, location="left", pad=0.06)

ax.invert_zaxis()
ax.invert_xaxis()

ax.xaxis.set_major_locator(plt.FixedLocator([x_val[0] + 1997, x_val[-1] + 1997]))
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))

ax.yaxis.set_major_locator(plt.FixedLocator([y_val[0] + 1, y_val[-1] + 1]))
ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
ax.zaxis.set_major_locator(plt.MultipleLocator(1))

# Set labels and title
font_label = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 16}
font_title = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 16}
cbar.set_label('Number of Data Points', labelpad=20, fontdict=font_label)
ax.set_xlabel('Years', labelpad=20, fontdict=font_label)
ax.set_ylabel('Marriage Duration', labelpad=20, fontdict=font_label)
ax.set_zlabel('Number of Kids', labelpad=20, fontdict=font_label)
ax.set_title('Divorces in Germany across the years, marriage duration\nand involved kids', fontdict=font_title)
cbar.ax.tick_params(labelsize=15, pad=3)
ax.xaxis.set_tick_params(labelsize=15, pad=3)
ax.yaxis.set_tick_params(labelsize=15, pad=3)
ax.zaxis.set_tick_params(labelsize=15, pad=3)


#plt.show()

fig.savefig("fig_dataset_intuition.pdf")




