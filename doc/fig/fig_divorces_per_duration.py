import matplotlib.pyplot as plt
from tueplots import bundles
import matplotlib.colors as mcolors
import os
import sys
current_dir = os.getcwd()
src_directory = current_dir + "/../../src/"
if src_directory not in sys.path:
    sys.path.append(src_directory)
from DataLoaderClass import DataLoader

plt.rcParams.update(bundles.icml2022(column="half", nrows=1, ncols=1, usetex=False))


dataloader = DataLoader()

data_array, years, durations, kids = dataloader.load_data(years_to_drop=[1997])

data_array = data_array.swapaxes(0, 2).sum(axis=2)

sum_div_per_kid = data_array.sum(axis=1)

for kid in kids:
    data_array[kid] = data_array[kid] / sum_div_per_kid[kid]

kid0_color = mcolors.CSS4_COLORS['lightcoral']
kid1_color = mcolors.CSS4_COLORS['darkseagreen']
kid2_color = mcolors.CSS4_COLORS['darkviolet']
kid3_color = mcolors.CSS4_COLORS['darkcyan']

plt.plot(durations, data_array[0], color=kid0_color)
plt.plot(durations, data_array[1], color=kid1_color)
plt.plot(durations, data_array[2], color=kid2_color)
plt.plot(durations, data_array[3], color=kid3_color)

plt.fill_between(durations, data_array[0], color=kid0_color, alpha=0.7)
plt.fill_between(durations, data_array[1], color=kid1_color, alpha=0.7)
plt.fill_between(durations, data_array[2], color=kid2_color, alpha=0.7)
plt.fill_between(durations, data_array[3], color=kid3_color, alpha=0.7)


plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(1))

plt.ylabel("Frequency of Divorces")
plt.xlabel("Marriage Durations [years]")
plt.title("Frequency of divorces per duration for\ndifferent number of involved children")
plt.legend(["0 Children", "1 Child", "2 Children", ">= 3 Children"])

plt.savefig("fig_divorces_per_duration.pdf")
# plt.show()
