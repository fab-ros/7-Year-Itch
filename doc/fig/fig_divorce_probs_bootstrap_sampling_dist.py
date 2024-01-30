import os
import sys
current_dir = os.getcwd()
src_directory = current_dir + "/../../src/"
if src_directory not in sys.path:
    sys.path.append(src_directory)
import numpy as np
import Bootstrap as Bt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from tueplots.constants.color import rgb
from DataLoaderClass import DataLoader

dataloader = DataLoader()

# plt.rcParams.update(bundles.icml2022(column="half", nrows=1, ncols=1, usetex=True))

num_simulations = 100
batch_size = 10000

''' first bootstrap: uncorrected data '''
data_array1, years1, durations1, kids1 = dataloader.load_data(years_to_drop=[1997])

data_array1 = data_array1.swapaxes(0, 1).sum(axis=2).sum(1)
sum_all_divs1 = data_array1.sum()
probs_per_duration1 = data_array1 / sum_all_divs1

probs_per_duration1 = probs_per_duration1 / np.sum(probs_per_duration1)

assert np.sum(probs_per_duration1) == 1, "does not sum to one"

mean1 = np.sum(durations1 * probs_per_duration1)
median1 = durations1[np.argmax(np.cumsum(probs_per_duration1) >= 0.5)]
most_likely1 = durations1[np.argmax(probs_per_duration1)]  # mode of the distribution

means1, medians1, most_probables1, estimates1 = Bt.bootstrap(num_simulations, batch_size, durations1,
                                                             probs_per_duration1)

''' second bootstrap: corrected data '''
data_array2, years2, durations2, kids2 = dataloader.load_data(years_to_drop=[1997], durations_to_drop=[26])

data_array2 = data_array2.swapaxes(0, 1).sum(axis=2).sum(1)
sum_all_divs2 = data_array2.sum()
probs_per_duration2 = data_array2 / sum_all_divs2

probs_per_duration2 = probs_per_duration2 / np.sum(probs_per_duration2)

assert np.sum(probs_per_duration2) == 1, "does not sum to one"

mean2 = np.sum(durations2 * probs_per_duration2)
median2 = durations2[np.argmax(np.cumsum(probs_per_duration2) >= 0.5)]
most_likely2 = durations2[np.argmax(probs_per_duration2)]  # mode of the distribution

means2, medians2, most_probables2, estimates2 = Bt.bootstrap(num_simulations, batch_size, durations2,
                                                             probs_per_duration2)

''' Plotting the two bootstraps '''
fig, ax = plt.subplots(figsize=[10, 6])

bt1_color = "lightcoral"
bt2_color = "darkseagreen"

bt1_mean_color = "firebrick"
bt1_median_color = "firebrick"
bt1_ml_color = "firebrick"

bt2_mean_color = "darkgreen"
bt2_median_color = "darkgreen"
bt2_ml_color = "darkgreen"

# bootstrap 1
mean1_line = ax.axvline(mean1, color=mcolors.CSS4_COLORS[bt1_mean_color], lw=2.5,
                        label="bt 1: est. mean duration at divorce", linestyle="-")
median1_line = ax.axvline(median1, color=mcolors.CSS4_COLORS[bt1_median_color], lw=3.5,
                          label="bt 1: est. median duration at divorce", linestyle="--")
ml1_line = ax.axvline(most_likely1, color=mcolors.CSS4_COLORS[bt1_ml_color], lw=3.5,
                      label="bt 1: est. most likely duration at divorce", linestyle=":")

for i in np.arange(num_simulations):
    ax.bar(np.array(durations1), height=estimates1[i, :] * 100, width=0.85, color=mcolors.CSS4_COLORS[bt1_color],
           lw=0.5,
           alpha=0.2, align="edge")

ax2 = ax.twinx()
hist11 = ax2.hist(means1, bins=np.array(durations1), color=mcolors.CSS4_COLORS[bt1_mean_color], lw=0, width=0.3,
                  alpha=0.70,
                  label="bt1 samp. dist. of est. mean duration at divorce", align="mid", density=True)
hist12 = ax2.hist(medians1, bins=np.array(durations1), color=mcolors.CSS4_COLORS[bt1_median_color], lw=0, width=0.3,
                  alpha=0.70,
                  label="bt1 samp. dist. of est. median\nduration at divorce", align="mid", density=True, hatch='\\')
hist13 = ax2.hist(most_probables1, bins=np.array(durations1), color=mcolors.CSS4_COLORS[bt1_ml_color], lw=0, width=0.3,
                  alpha=0.70,
                  label="bt1 samp. dist. of est. most likely\nduration at divorce", align="mid", density=True,
                  hatch='//')

# bootstrap 2
mean2_line = ax.axvline(mean2, color=mcolors.CSS4_COLORS[bt2_mean_color], lw=3.5,
                        label="bt 2: est. mean duration at divorce", linestyle="-")
median2_line = ax.axvline(median2, color=mcolors.CSS4_COLORS[bt2_median_color], lw=3.5,
                          label="bt 2: est. median duration at divorce", linestyle="--")
ml2_line = ax.axvline(most_likely2, color=mcolors.CSS4_COLORS[bt2_ml_color], lw=3.5,
                      label="bt 2: est. most likely duration at divorce", linestyle=":")

for i in np.arange(num_simulations):
    ax.bar(np.array(durations2), height=estimates2[i, :] * 100, width=0.55, color=mcolors.CSS4_COLORS[bt2_color],
           lw=0.5,
           alpha=0.2, align="edge")

hist21 = ax2.hist(means2, bins=np.array(durations2), color=mcolors.CSS4_COLORS[bt2_mean_color], lw=0.0, width=0.3,
                  alpha=0.70,
                  label="bt2 samp. dist. of est. mean duration at divorce", align="mid", density=True)
hist22 = ax2.hist(medians2, bins=np.array(durations2), color=mcolors.CSS4_COLORS[bt2_median_color], lw=0.0, width=0.3,
                  alpha=0.70,
                  label="bt2 samp. dist. of est. median\nduration at divorce", align="mid", density=True, hatch='\\')
hist23 = ax2.hist(most_probables2, bins=np.array(durations2), color=mcolors.CSS4_COLORS[bt2_ml_color], lw=0.0,
                  width=0.3, alpha=0.70,
                  label="bt2 samp. dist. of est. most likely\nduration at divorce", align="mid", density=True,
                  hatch='//')

data1 = ax.scatter(np.array(durations1), probs_per_duration1 * 100, color=mcolors.CSS4_COLORS[bt1_color],
                   label="probability of divorce original data", edgecolor='black', s=25)
data2 = ax.scatter(np.array(durations2), probs_per_duration2 * 100, color=mcolors.CSS4_COLORS[bt2_color],
                   label="probability of divorce corrected data", edgecolor='black', s=25)

handles_line_styles = [plt.Line2D([], [], linestyle="-", color="black"),
                       plt.Line2D([], [], linestyle="--", color="black"),
                       plt.Line2D([], [], linestyle=":", color="black")]
handles_line_labels = ["estimated mean\nduration at divorce",
                       "estimated median\nduration at divorce",
                       "estimated most likely\nduration at divorce"]

handles_scatter = [data1, data2]
handles_scatter_label = ["P(divorce) original data", "P(divorce) corrected data"]

handles_colors = [Patch(facecolor=mcolors.CSS4_COLORS[bt1_color]),
                  Patch(facecolor=mcolors.CSS4_COLORS[bt2_color])]
labels_colors = ['bootstrap 1\ncolorscheme', 'bootstrap 2\ncolorscheme']

handles_histogram = [Patch(facecolor=mcolors.CSS4_COLORS['white'], edgecolor="black"),
                     Patch(hatch="\\", facecolor=mcolors.CSS4_COLORS['white'], edgecolor="black"),
                     Patch(hatch='//', facecolor=mcolors.CSS4_COLORS['white'], edgecolor="black")]
handles_histogram_labels = ["sampling dist. of est. mean\nduration at divorce",
                            "sampling dist. of est. median\nduration at divorce",
                            "sampling dist. of est. most likely\nduration at divorce"]

# Add legends to the plot

ax.legend(handles=handles_colors, labels=labels_colors, loc='upper left', fontsize=11)

# Combine legends into one
current_handles, current_labels = plt.gca().get_legend_handles_labels()
plt.legend(handles_histogram + handles_line_styles + handles_scatter,
           handles_histogram_labels + handles_line_labels + handles_scatter_label,
           loc=[0.52, 0.36], fontsize=11)

# cosmetics
font_label = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 16}
font_title = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 16}

ax.set_xlabel("Marriage Duration [years]", labelpad=10, fontdict=font_label)
ax.set_ylabel("Probability of Divorce", labelpad=10, fontdict=font_label)
ax.set_xlim([0, 27])
ax.set_ylim([0, 15])
ax.yaxis.set_major_locator(plt.MultipleLocator(5))
ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x:,.1f}%"))
ax.xaxis.set_major_locator(plt.MultipleLocator(5))
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
ax.xaxis.set_tick_params(labelsize=15, pad=3)
ax.yaxis.set_tick_params(labelsize=15, pad=3)

ax2.set_ylim([0, 1])
ax2.yaxis.set_major_locator(plt.MultipleLocator(0.2))
ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax2.set_ylabel("bootstrap sampling density", labelpad=10, fontdict=font_label)
ax2.yaxis.set_tick_params(labelsize=15, pad=3)

ax.grid(axis="both", which="both", color=rgb.tue_gray, linewidth=0.5)
ax.set_axisbelow(True)

ax.set_title(f"Comparison of Bootstrap sampling distributions of estimated\nmean,"
             f" median and most likely duration at divorce", fontdict=font_title)

# plt.show()
fig.savefig("fig_divorce_probs_bootstrap_sampling_dist.pdf")
