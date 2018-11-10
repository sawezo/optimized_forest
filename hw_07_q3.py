#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created Oct 2018
File to replicate simulation of a model to determine optimal firebreaks for timber yield.
Inspiration from Professor Dodds, University of Vermont
@author: Samuel Zonay
"""

# =============================================================================
# ==========================    IMPORT STATEMENTS   ===========================
# =============================================================================
# Data Science Imports
import numpy as np
from skimage import measure
import random as random
import math

# Plotting Imports
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors
import seaborn as sns # To improve graph aesthetic.
sns.set_style("darkgrid")

import matplotlib.animation as animation


# DEBUG Imports
import sys

# =============================================================================
# ==========================    INITIALIZE VARIABLES    =======================
# =============================================================================
L_values = [2, 32] # 2, 32, 64, 128 
D_values = [2] # [1,2,"L","L_squared"]

# D2L2records = {} # The dictionary representing the main data structure to record 
# # any generated data I compute below. 
# D2L2max_yields = {} # For recording and plotting max yield for each L, etc. 


# # TERMINAL OUTPUT OF PROGRESS (FOR USAGE DURING LONG SIMULATIONS)
# L_squares = [L**2 for L in L_values]
# L_total = np.sum(np.array(L_squares))  
# D_total = np.sum(np.array(D_values))  
# main_total = D_total * L_total
# User output incrementing.
# i = 0

# ========================      FUNCTIONS     =================================
def spark_probability_matrix(index_list):
    """
    - Function Description:
       Function takes in a list of lattice indeces and returns a matrix of the 
       spark probabilities for each of those indeces. 
    - Function Inputs:
        - initial_lattice: Initial lattice for which we are going to compute the spark
        probabilities for.
    - Function Output: 
        spark_probability_matrix: Normalized matrix of spark probabilities (numpy 2-D array). 
    """
    # We will start by adding the test tree index to the array we are curious about. 
    # The returned matrix should be of the same initial lattice size. 
    L = int(math.sqrt(len(index_list)))
    l = L/10 # The characteristic scale for this distribution. 

    # Creating the spark_probability lattice in memory to fill. 
    spark_probability_lattice = np.zeros((L, L))
    for index in index_list:
        spark_probability_lattice[index] = math.exp(-index[0]/l) * math.exp(-index[1]/l)
    
    # Now to normalize all of the values of the newly constructed matrix. 
    spark_probability_lattice = spark_probability_lattice/np.sum(spark_probability_lattice)    
    
    return spark_probability_lattice

def cost_calc(spark_probability_lattice, simulation_lattice, test_tree_index):
    """
    - Function Description:fdsa
       Function takes in simulation information and returns the average cost calculated when a 
       given test tree is placed in the current simulation lattice. The equation for average forest 
       fire size (used to compute the cost) is \sum_{i, j} P(i, j), S(i, j), where P represents spark 
       probability at location (i, j) and S represents the size of any current component at the same spot.
    - Function Inputs:
        spark_probability_lattice: Spark probability matrix for this simulation (numpy matrix).
        simulation_lattice: Current simulation lattice we are testing with (numpy matrix).
        test_tree_index: Index (list of integers) of index to test putting an additional tree at. 
    - Function Output: 
        calculated_average_cost: The estimated average cost of a fire (with cost here being fire size).
    """
    # Initializing the variable we are looking to compute. Its a summation so we will find - 
    # it via incrementation. 
    average_forest_fire_size = 0
    
    # We will start by adding the test tree index to the array we are curious about. 
    # First we are going to copy the current simulation lattice before adding a tree - 
    # theoretically, to stop any potential errors with overwrititng the initial lattice. 
    test_lattice = simulation_lattice.copy()
    test_lattice[test_tree_index[0],test_tree_index[1]] = 1

    # Calculating the connected components. 
    connected_components = measure.label(test_lattice, connectivity=1)
    # From skimage docs: "A labeled array, where all connected regions are -
    # - assigned the same integer value."
  
    # Now that we have anlayzed connected component data, we analyze potential fire sizes. 
    unique, counts = np.unique(connected_components, return_counts=True)
    components2size = dict(zip(unique, counts)) 
    
    # A list of the indeces where the lattice has trees (includes our test tree!). 
    tree_indeces = (np.argwhere(test_lattice==1)).tolist()    
    
    # Now for each tree in the test lattice we will compute cost at that lattice grid. 
    # Note indexes that represent empty spaces will have cost of zero to increment, so we skip them. 

    for tree_index in tree_indeces:
        # Getting the componenet size of the current test tree index we are considering. 
        tree_component_value = connected_components[tree_index[0], tree_index[1]]
        component_size = components2size[tree_component_value]

        spark_probability = spark_probability_lattice[tree_index[0], tree_index[1]]
        average_size_of_fire = (component_size * spark_probability)

        # Incrementing our calculated cost value. 
        average_forest_fire_size += average_size_of_fire # Note we have the average forest fire size equation. 

    return average_forest_fire_size

def fix_chosen_index(chosen_index):
    """
    fdsa Array parser 
    for manual edits of [0 2] vs [0, 2] error
    """
    items = list(str(chosen_index))
    
    i_value = ""
    j_value = ""

    number = 1
    first_space = 0
    for char in items:
        first_space -= 1
        if char == "[":
            first_space = 2
        elif (str(char).isdigit() == True) & (number == 1):
            if char == '':
                pass
            else:
                i_value += str(char)
        elif char == " ":
            if first_space == 1: # Preventing this from registering as the first whitespace.
                pass
            else:
                number = 2
        elif (str(char).isdigit() == True) & (number == 2):
            j_value += str(char)
        elif char == "]":
            pass

    fixed_chosen_index = [int(i_value), int(j_value)]

    return fixed_chosen_index

def animate(frame_index):
    """
    fdsa
    """
    simulation_lattice = trees_added2lattice[frame_index] # +1 since frame_index is based at zero. 
    index_zeros_len = len(np.argwhere(simulation_lattice==0).tolist())
    
    # This is to swap the color scheme to manually fix an error that would show the opposite plot. 
    if index_zeros_len < 1:
        color_map = colors.ListedColormap(['green', 'brown'])
        plt.imshow(simulation_lattice, cmap=color_map, animated=True)

    # If we aren't on the full forest, we do the same thing but with altered colorscheme. 
    else:
        color_map = colors.ListedColormap(['brown', 'green'])
        plt.imshow(simulation_lattice, cmap=color_map, animated=True)

# ========================      MAIN PROGRAM LOGIC     ========================
def main_iterator(D, L):
    """
    fdsa
    """
    # Addressing any issues from type(L) == String.
    if D == "L":
        D = L
    if D == "L_squared":
        D = L**2

    # We start this process by defining data structures we are going to be utilizing. 
    L_density2yields = []
    density2average_yield = {}
    average_yield2lattice = {}
    trees_added2lattice = {}
    average_yield2trees_added = {}
    trees_added = 0

    # Initializing both the spark and the empty simulation lattice (the latter being that we look to fill). 
    simulation_lattice = np.zeros((L,L)) # The lattice is initialized. 
    # We get a list of each index in the initialized lattice that is empty. 
    index_zeros = (np.argwhere(simulation_lattice==0)).tolist()
    spark_probability_lattice = spark_probability_matrix(index_zeros)

    # Adding the initial lattice to the animation records. 
    trees_added2lattice[0] = simulation_lattice.copy() 
    
    # We want to run this simulation as long as there are empty spaces available. 
    while len(index_zeros) > 0:  
        # Now to reset the perimiter variable we used as a dummy variable:
        if l_square_switch == True:
            D = L # Updating the perimiter size with the current L size. 
        if l_square_switch == True:
            D = L**2

        # For each draw we have (depending on the current value of D), we draw an empty index. 
        try:
            test_tree_indeces = random.sample(index_zeros, D)
        # Catching case of D > remaining number of indeces to potentially plant a tree at. 
        except ValueError:
            # The next best case is taking as many empty options as we can.
            test_tree_indeces = np.argwhere(simulation_lattice==0)

        # Now we iterate and calculate average cost based off of each test tree. 
        computed_sizes2tree_index = {}
        average_yield2index = {}

        trees_added += 1 # Incrementing this here since yield will need to be computed - 
        # - (used for density calculation).
        for test_tree_index in test_tree_indeces:                                
            # Computing and saving our average forest fire size for this test tree. 
            average_forest_fire_size = cost_calc(spark_probability_lattice, \
                                                simulation_lattice, test_tree_index)
            computed_sizes2tree_index[average_forest_fire_size] = test_tree_index
            density = (trees_added/(L**2))
            average_yield = density - (average_forest_fire_size/(L**2))    
            average_yield2index[average_yield] = test_tree_index

        highest_yield = max(average_yield2index.keys())
        chosen_index = average_yield2index[highest_yield]
        
        # Removing the chosen index from our empty index list and adding it - 
        # - to the simulation lattice.             
        simulation_lattice[chosen_index[0], chosen_index[1]] = 1

        # Since the test tree index has now been determined, we remove the - 
        # - index from the array of empty indeces. 
        try:
            index_zeros.remove(chosen_index)
        except ValueError: # From array error described more in fix_chosen_index() function description.
            fixed_chosen_index = fix_chosen_index(chosen_index)
            index_zeros.remove(fixed_chosen_index)

        # We calculate the yield associated with this tree from the cost.             
        # average_yield2lattice[highest_yield] = simulation_lattice
        # density2average_yield[density] = highest_yield  
        
        # For animation.
        trees_added2lattice[trees_added] = simulation_lattice.copy()
        average_yield2trees_added[highest_yield] = trees_added
        density2average_yield[density] = highest_yield

    L_density2yields.append(density2average_yield)

    if l_switch == True:
        D = "L" # Updating the perimiter size with the current L size. 
    if l_square_switch == True:
        D = "L_squared"

    # max_yield = max(trees_added2average_yield.values())
    # L2max_yield[L] = max_yield

    # max_yield_printer(max_yield)
    
    # Part b) Plots
    # print(density2average_yield.values())
    # highest_yield = max(density2average_yield.values())
    # print(average_yield2lattice[highest_yield])
    # print(highest_yield)
    # print('fo', np.argwhere(average_yield2lattice[highest_yield]==0).tolist())
    # plt.imshow(average_yield2lattice[highest_yield])
    # plt.show()


def main():
    """
    fdsa
    """
    for D in D_values:
        for L in L_values:
            main_iterator(D, L)

main()

# ========================      PLOTTING                  ========================
print("NOW PLOTTING")
# Animation.

# fig = plt.figure()
# frame_count = len(trees_added2lattice.keys())
# ani = animation.FuncAnimation(fig, animate, frames=frame_count, repeat=True, interval=.01)
# plt.show()

# Part a) Plot the forest at (approximate) peak yield.

peak_yield = max(average_yield2trees_added.keys())
trees_added_peak_yield = average_yield2trees_added[peak_yield]
peak_yield_lattice = trees_added2lattice[trees_added_peak_yield]

# Q3a_fig = plt.figure()
# color_map = colors.ListedColormap(['brown', 'green'])
# plt.imshow(peak_yield_lattice, cmap=color_map)
# plt.show()

# # Part b) Plot the yield curves for each value of D, and identify (approximately) the - 
# # - peak yield and the density for which peak yield occurs for each value of D.

# color_map = {2:"green", 32:"yellow", 64:"red", 128:"orange"}
# def yieldBYdensity_plotter(L_dictionary_datalist, D):
#     """
#     fdsa
#     """
#     # print(L_dictionary_datalist[0])
#     Q3pb = plt.figure()
#     axes = Q3pb.add_subplot(111)
#     for density2yields in L_density2yields:
#         x = list(density2yields.keys()) # Densities.
#         x.insert(0, 0)
#         y = list(density2yields.values())
#         y.insert(0, 0)

#         axes.set_title("Yield Curve by Density $p$ for $D = $ {D}".format(D=D))
#         axes.set_xlabel("Density $p$") # , ($log_{10}$)
#         axes.set_ylabel("Average Yield $Y$") # , ($log_{10}$)
#         green_patch = mpatches.Patch(color='green', label="$L=2$")
#         yellow_patch = mpatches.Patch(color='yellow', label="$L=32$")
#         red_patch = mpatches.Patch(color='red', label="$L=64$")
#         # cmap=color_map[L],
#         plt.legend(handles=[green_patch, yellow_patch, red_patch])
#         axes.plot(x, y, linestyle='-',  label='L = {L}'.format(L=L))
#         # plt.savefig('q3_b.png', bbox_inches='tight')
#     plt.show()

# yieldBYdensity_plotter(L_density2yields, D)

# Part c) Plot distributions of tree component sizes S at peak yield. Note: You will 
# have to rebuild forests and stop at the peak yield value of D to find these 
# distributions. By recording the sequence of optimal tree planting, this can be 
# done without running the simulation again.

 # Calculating the connected components. 
connected_components = measure.label(peak_yield_lattice, connectivity=1)
# From skimage docs: "A labeled array, where all connected regions are -
# - assigned the same integer value."

# Now that we have anlayzed connected component data, we analyze potential fire sizes. 
unique, counts = np.unique(connected_components, return_counts=True)
components2size = dict(zip(unique, counts))

freqBYcomponent_size = plt.figure()
axes = freqBYcomponent_size.add_subplot(111)
# color_map = {20:"black", 50:"green", 100:"yellow", 200:"orange", 500:"red",1000:"blue"}

#for L in L2runningAvg_pvals.keys():
x = components2size.values() # The x axis is .
y = components2size.keys() # The y axis is . 
axes.scatter(x, y) #, color=color_map[L]
    
# Labelling and making the plot output look nicer. 
# axes.set_title("$S_{avg}$ by Probability Value $p$")
axes.set_xlabel("Component Size") # , ($log_{10}$)
# axes.set_ylabel("$S_{avg}$") # , ($log_{10}$)
# axes.legend(['$L = 20$','$L = 50$','$L = 100$','$L = 200$','$L = 500$']) # ,'$L = 1000$'
plt.show()

# Part d) Extra level: Plot size distributions for D = L2 for varying tree densities
# ρ = 0.10, 0.20, . . . , 0.90. This will be an effort to reproduce Fig. 3b in [2].

# ========================      SCRAP                  ========================
# average_forest_fire_size = 0 # A counter for the average forest fire size at this iteration. 

# ========================      Q's/ OTHER                  ===================
# density = trees_added/(L**2)