# plot for divorces per kid per year
# code to load the dataset
import sys
import os
current_dir = os.getcwd()
src_directory = current_dir + "/../../src/"
if src_directory not in sys.path:
    sys.path.append(src_directory)

print(current_dir)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tueplots import bundles
from tueplots.constants.color import rgb
from DataLoaderClass import DataLoader

plt.rcParams.update(bundles.icml2022(column="half", nrows=1, ncols=1, usetex=False))

dataloader = DataLoader()
# load data
dataset_array, durations, years, kids = dataloader.load_data()

from io import StringIO

# Dataset of married couples in germany with and without kids per year
# after dropping the first to datapoints we copied the following lines
# which can be found in dat/pairs_with_wo_kids.csv

pairs_with_kids = "7508;7364;7264;7107;7036;6873;6729;6654;6476;6327;6132;5963;5846;5739;5699;5639;5589;5544;5697;5721;5643;5723"
pairs_wo_kids = "9406;9492;9600;9703;9748;9790;9847;9673;9681;9807;9809;9841;9854;9566;9661;9701;9702;9792;9714;9695;9815;9658"

# factor to get the acutal number of married couples
multiplication_factor = 1000

# write data into temporary csv file
temp_csv_file = StringIO()
temp_csv_file.write(pairs_with_kids)
temp_csv_file.seek(0)

# import data as NumPy-Array 
pairs_with_kids = np.fromstring(temp_csv_file.getvalue(), dtype=int, sep=';')

# to check 
print(pairs_with_kids*multiplication_factor)

# for married couples without kids
# write data into temporary csv file
temp_csv_file = StringIO()
temp_csv_file.write(pairs_wo_kids)
temp_csv_file.seek(0)

# import data as NumPy-Array 
pairs_wo_kids = np.fromstring(temp_csv_file.getvalue(), dtype=int, sep=';')

# to check
print(pairs_wo_kids*multiplication_factor)
print(len(pairs_with_kids))

# total number of married pairs it given by the sum of couples with and w/o kids
total_pairs = pairs_with_kids + pairs_wo_kids

# to build the plot 
# sum on axis=1 meaning to sum over all durations per #kids and year
# normalise by if possible
#years = np.arange(2020,2023)
# von Datenset kommen Daten von 1997 bis 2023
# von pairs_with_wo_kids kommen Daten von 1996 bis 2019
year_max = 2019
year_min = 1998
years = range(year_min,year_max+1)
year_max_offset = 2023-year_max-2
num_durations = 20


def divorces_kids(kids):
    data = np.array([])
    for year in range(25):
        # Initialize sum before the loop
        sum_value = 0
        
        for duration in range(num_durations):
            # Calculate the sum along the third axis (axis=0) for each year and duration
            sum_value += np.sum(dataset_array[year][duration][kids])

        # Append the result to the data list
        data = np.append(data,sum_value.astype(int))
    return data
# Now, 'data' contains the sum of the first element along the third axis for each year


kid_0 = divorces_kids(0) 
kid_0 = kid_0[:-year_max_offset]
kid_0 = kid_0[1:]

kid_1 = divorces_kids(1)
kid_1 = kid_1[:-year_max_offset]
kid_1 = kid_1[1:]

kid_2 = divorces_kids(2)
kid_2 = kid_2[:-year_max_offset]
kid_2 = kid_2[1:]

kid_3 = divorces_kids(3)
kid_3 = kid_3[:-year_max_offset]
kid_3 = kid_3[1:]



# cause of the fact that the second dataset only provides information about having or not having children
# we needed to aggregate the groups: one child, two children and three or more children into one group of HAVING CHILDREN
# normalization
print(total_pairs)
norm_wo_kids = 1/(pairs_wo_kids * 1000)
norm_with_kids =1/(pairs_with_kids * 1000)
print(norm_with_kids)
print(norm_wo_kids)

# divorces with kids 
divorces_with_kids = kid_1 +  kid_2 + kid_3
# divorces without kids
divorces_wo_kids = kid_0


norm_wo_kids = 1/(pairs_wo_kids * 1000)
norm_with_kids =1/(pairs_with_kids * 1000)
years = np.arange(1998,2020)
pairs_with_kids_weighted = pairs_with_kids / total_pairs
pairs_wo_kids_weighted = pairs_wo_kids / total_pairs


# Plotting
# Define RGB values in the range [0, 255]

#color1 = np.array([239,135,54])/255 #kind of orange
#color2 = np.array([60,117,176])/255 #kind of lightblue
color1 = mcolors.CSS4_COLORS['darkred']
color2 = mcolors.CSS4_COLORS['darkgreen']

fig, ax1 = plt.subplots()
# Plot for the first subplot
bar_width = 0.3  # Adjust the bar width as needed
ax1.bar(years, pairs_with_kids_weighted, width=bar_width, alpha=0.5, color=color2)
ax1.bar(np.array(years) + bar_width+0.1, pairs_wo_kids_weighted, width=bar_width, alpha=0.5, color=color1)
ax1.set_xlabel("Years")
ax1.set_ylabel("Married couples")
ax1.set_ylim([0.0,1.0])
ax1.legend(["Couples with kids","Couples without kids"], loc="upper left")

# Plot for the second subplot
ax2 = ax1.twinx()

ax2.plot(years, divorces_with_kids * norm_with_kids, color=color2)
ax2.plot(years, divorces_wo_kids * norm_wo_kids, alpha=1, color=color1)
ax2.set_title("Married couples with and w/o kids and \n divorces of married couples with and w/o kids (normalized) per year")
ax2.set_xlabel("Years")
ax2.set_ylabel("Divorces")
ax2.set_ylim([0.00,0.022])
ax2.legend(["Divorces with kids", "Divorces without kids"],loc="upper right")
ax2.set_yscale('linear')
ax2.xaxis.set_major_locator(plt.MultipleLocator(3))
ax2.xaxis.set_minor_locator(plt.MultipleLocator(1))


# to show the plot
#plt.show() 

fig.savefig("fig_divorces_per_kid_per_year.pdf")
