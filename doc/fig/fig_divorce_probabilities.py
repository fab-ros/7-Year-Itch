import numpy as np
import src.Bootstrap as Bt
import matplotlib.pyplot as plt
from tueplots import bundles
from tueplots.constants.color import rgb
import sys
import os
current_dir = os.getcwd()
src_directory = current_dir + "/../../src/"

if src_directory not in sys.path:
    sys.path.append(src_directory)
from src.DataLoaderClass import DataLoader

dataloader = DataLoader()

plt.rcParams.update(bundles.icml2022(column="half", nrows=1, ncols=1, usetex=True))

data_array, years, durations, kids = dataloader.load_data(years_to_drop=[1997], durations_to_drop=[26])

data_array = data_array.swapaxes(0, 1).sum(axis=2).sum(1)
sum_all_divs = data_array.sum()
probs_per_duration = data_array / sum_all_divs

mean = np.sum(np.array(durations) * probs_per_duration)
median = durations[np.argmax(np.cumsum(probs_per_duration) >= 0.5)]
mode = durations[np.argmax(probs_per_duration)]

num_simulations = 100
batch_size = 10000

means, medians, most_probables, estimates = Bt.bootstrap(num_simulations, batch_size, durations, probs_per_duration)

fig, ax = plt.subplots()

ax.axvline(mean, color=rgb.tue_orange, lw=2, label="estimated mean duration at divorce")
ax.axvline(median, color=rgb.tue_red, lw=2, label="estimated median duration at divorce")
ax.axvline(mode, color=rgb.tue_green, lw=2, label="estimated most likely duration at divorce")


for i in np.arange(num_simulations):
    ax.bar(durations, height=estimates[i, :] * 100, width=0.8, color=rgb.tue_gray, lw=0.5,
           alpha=0.1, align="edge")

ax2 = ax.twinx()
ax2.hist(means, bins=durations, color=rgb.tue_orange, lw=0.5, width=0.8, alpha=0.75,
         label="bootstrap sampling dist of \nestimated mean duration at divorce", align="mid", density=True)
ax2.hist(most_probables, bins=durations, color=rgb.tue_green, lw=0.5, width=0.8, alpha=0.75,
         label="bootstrap sampling dist of \nestimated most likely duration at divorce", align="mid", density=True)
ax2.hist(medians, bins=durations, color=rgb.tue_red, lw=0.5, width=0.8, alpha=0.75,
         label="bootstrap sampling dist of \nestimated median duration at divorce", align="mid", density=True)
ax2.set_ylim([0, 1])
ax2.yaxis.set_major_locator(plt.MultipleLocator(0.2))
ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax2.set_ylabel("bootstrap sampling density")

ax.scatter(np.array(durations) + 0.35, probs_per_duration * 100, color='red', label="probs of orginal data", s=5)
ax.bar(durations, probs_per_duration, width=0.8, color=rgb.tue_blue, lw=0.5,
       alpha=0.1, label="probability of divorce", align="edge")

ax.set_xlabel("marriage duration")
ax.set_ylabel("probability of divorce")
ax.set_ylim([0, 15])
ax.yaxis.set_major_locator(plt.MultipleLocator(2.5))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x:,.1f}%"))
ax.xaxis.set_major_locator(plt.MultipleLocator(5))
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))

#ax.grid(axis="both", which="both", color=rgb.tue_gray, linewidth=0.5)
ax.set_axisbelow(True)
fig.legend(loc=[0.5, 0.4], framealpha=1, facecolor="white", frameon=True, fontsize="x-small")
ax.set_title(f"Bootstrap sampling distribution of estimated mean, median and most likely \nduration at divorce based on "
             f"{num_simulations} simulations of {batch_size} marriages")
ax.set_xlim([0, 26])

#plt.show()

fig.savefig("fig_divorce_probabilities.pdf")
