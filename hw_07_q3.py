#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created Oct, 2018
File to replicate simulation of a model to determine optimal firebreaks for timber yield.
Inspiration from Professor Dodds, University of Vermont.
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
import seaborn as sns 
sns.set_style("darkgrid")
import matplotlib.animation as animation

# DEBUG Imports
import sys

# =============================================================================
# ==========================    "KNOB" VARIABLES    ===========================
# =============================================================================
L_values = [32] # 2, 4, 16, 32, 64, 128 
D_values = [2] # [1,2,"L","L_squared"]

# =============================================================================
# ========================      FUNCTIONS     =================================
# =============================================================================
def main(L_values, D_values):
    """
    - Function Description:
        Function takes in a list of values to run the simulation on and processes program calls.
    - Function Inputs:
        - L_values: List of L values (lattice perimiter), (type=list).
        - D_values: List of D values (design parameter), (type=list).
    - Function Output: 
        Function has no output.
    """
    print("--- NOW RUNNING SIMULATIONS ---")

    # Call to run animation functionality.
    display = True
    logg = True 

    for D in D_values:
        yields2trees_added_LIST = [] # Will reset this for each D value. 

        for L in L_values:
            trees_added2lattice, average_yield2trees_added, density2lattice = simulator(D, L)

            # Running the animation for this simulation. 
            run_animation(display, trees_added2lattice, L, D)

            # We are going to want to analyze the specific peak yield state. 
            peak_yield = max(average_yield2trees_added.keys())
            trees_added_peak_yield = average_yield2trees_added[peak_yield]
            peak_yield_lattice = trees_added2lattice[trees_added_peak_yield]

            # Plots of max yield (topographical map view).
            lattice_frame_grapher(display, peak_yield_lattice, D, L) 
            
            # We want to inquire more about 
            component_frequency_grapher(display, peak_yield_lattice, D, L, logg)

            # For plotting yields by densities later
            yields2trees_added_LIST.append(average_yield2trees_added) 
        # Now to plot the peak yield curves for this value of D.
        yieldBYdensity_grapher(display, D, yields2trees_added_LIST)
        # Finally we plot component size frequencies across densities. 
        # component_sizeBYdensity_grapher(display, D, density2lattice)

def simulator(D, L):
    """
    - Function Description:
        Function takes in specific D and L values and runs the simulation on each. 
    - Function Inputs:
        - D: The L value (lattice perimiter), (type=int).
        - D_values: The D value (design parameter), (type=int).
    - Function Output: 
        Function Outputs the following information for graphing purposes. 
            - trees_added2lattice: Dictionary of trees added (int) to lattice (numpy matrix). 
            - average_yield2trees_added: Dictionary of average yield (float) to trees added here (int).
            - density2lattice: Dictionary of density of forest (float) to the lattice only for 
              specified densities to save for later plotting. 
    """
    # Addressing any issues from type(L) == String.
    if D == "L":
        D = L
    if D == "L_squared":
        D = L**2

    # We start this process by defining data structures we are going to be utilizing. 
    trees_added2lattice = {} # For holding data to animate.
    average_yield2trees_added = {} # For plotting max yield data. 
    density2lattice = {} # For plotting.
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
        # For each draw we have (depending on the current value of D), we draw an empty index. 
        try:
            test_tree_indeces = random.sample(index_zeros, D)
        # Catching case of D > remaining number of indeces to potentially plant a tree at. 
        except ValueError:
            # The next best case is taking as many empty options as we can.
            test_tree_indeces = np.argwhere(simulation_lattice==0)

        # Now we iterate and calculate average cost based off of each test tree. 
        average_yield2index = {}

        trees_added += 1 # Incrementing this here since yield will need to be computed - 
        # - (used for density calculation) to select which tree to add.

        density = (trees_added/(L**2)) # It is important to note this was done after appending a tree - 
        # to our trees_added counter. 

        for test_tree_index in test_tree_indeces:                                
            # Computing and saving our average forest fire size for this test tree. 
            average_forest_fire_size = cost_calc(spark_probability_lattice, \
                                                simulation_lattice, test_tree_index)
        
            average_yield = density - (average_forest_fire_size/(L**2)) # We want cost in terms of density.    
            average_yield2index[average_yield] = test_tree_index

        # Selecting the index with highest yield.
        highest_yield = max(average_yield2index.keys())
        chosen_index = average_yield2index[highest_yield]
        
        # Removing the chosen index from our "empty" index list and adding it to the simulation lattice.             
        simulation_lattice[chosen_index[0], chosen_index[1]] = 1

        if (density % .25) == 0:
                density2lattice[density] = simulation_lattice.copy() # For graphing later. 

        # Since the test tree index has now been determined, we remove the - 
        # - index from the array of empty indeces. 
        try:
            index_zeros.remove(chosen_index)
        except ValueError: # From array error described more in fix_chosen_index() function description.
            fixed_chosen_index = fix_chosen_index(chosen_index)
            index_zeros.remove(fixed_chosen_index)

        trees_added2lattice[trees_added] = simulation_lattice.copy() # For animation.
        average_yield2trees_added[highest_yield] = trees_added # For plotting peak yield.

    return trees_added2lattice, average_yield2trees_added, density2lattice

def spark_probability_matrix(index_list):
    """
    - Function Description:
       Function takes in a list of lattice indeces and returns a matrix of the 
       spark probabilities for each of those indeces. 
    - Function Inputs:
        - index_list: List of indeces for which we need to compute the spark
        probabilities for (list).
    - Function Output: 
        - spark_probability_lattice: Normalized matrix of spark probabilities (numpy matrix). 
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
    - Function Description:
       Function takes in simulation information and returns the average cost calculated when a 
       given test tree is placed in the current simulation lattice. The equation for average forest 
       fire size (used to compute the cost) is $\sum_{i, j} P(i, j) S(i, j)$, where P represents spark 
       probability at location (i, j) and S represents the size of any current component at the same spot.
    - Function Inputs:
        - spark_probability_lattice: Spark probability matrix for this simulation (numpy matrix).
        - simulation_lattice: Current simulation lattice we are testing with (numpy matrix).
        - test_tree_index: Index (list of integers) of index to test putting an additional tree at. 
    - Function Output: 
        - calculated_average_cost: The estimated average cost of a fire (with cost here being fire size).
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
    - Function Description:
       Function ammends an error that occurs when there is only one item remaining in the list of 
       indeces with no trees. In particular, function changes an array being represented as [0 2]
       to [0, 2], including given double digits or additional whitespace padding. 
    - Function Inputs:
        - chosen_index: String representation of index we want to fix so we can work with. 
    - Function Output: 
        - fixed_chosen_index: The updated index that will no longer give the main looping an ValueErrors. 
    """
    items = list(str(chosen_index))
    
    # Containers incase there are multiple digits to a single numerical. 
    i_value = ""
    j_value = ""

    # Initial settings; note the switch signifying no number has been saved yet (out of two total) - 
    # - as shown by `number` being set to 1 to start. 
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

    # The chosen index is returned with correct forms. 
    fixed_chosen_index = [int(i_value), int(j_value)]

    return fixed_chosen_index

def run_animation(display, trees_added2lattice, L, D):
    """
    - Function Description:
       Function runs the animate function nested within (to utilize the trees added to lattice matrix 
       dictionary). 
    - Function Inputs:
        - display: Whether or not we want to display graphs to user terminal (Boolean).
        - trees_added2lattice: A dictionary mapping the count of trees added to the lattice at the addition. 
    - Function Output: 
        - Nothing, but if display=True an animation will appear. 
    """
    def animate(frame_index):
        """
        - Function Description:
            Function represents the animate function nested within (to utilize the trees
            added to the lattice matrix dictionary). During its runtime an animation will process, 
            with the option for it to print to the user's terminal. 
        - Function Inputs:
            - frame_index: the index of the frame to run (int), as required bt matplot Animate. 
        - Function Output: 
            - Nothing, but if display=True an animation will appear in the external animation 
        """
        # Internal Function Code.
        simulation_lattice = trees_added2lattice[frame_index] 
        index_zeros_len = len(np.argwhere(simulation_lattice==0).tolist())
    
        # This is to swap the color scheme to manually fix an error that would show the opposite plot. 
        if index_zeros_len < 1:
            color_map = colors.ListedColormap(['green', 'brown'])
            plt.imshow(simulation_lattice, cmap=color_map, animated=True)

        # If we aren't on the full forest, we do the same thing but with altered colorscheme. 
        else:
            color_map = colors.ListedColormap(['brown', 'green'])
            plt.imshow(simulation_lattice, cmap=color_map, animated=True)

    # External Function Code.
    fig = plt.figure()
    frame_count = len(trees_added2lattice.keys())
    ani = animation.FuncAnimation(fig, animate, frames=frame_count, repeat=True, interval=1)
    ani.save('animaz.gif', dpi=80, writer='imagemagick')

    if display == True:
        plt.show()
    

def lattice_frame_grapher(display, peak_yield_lattice, D, L):
    """
    - Function Description:
        Function graphs a lattice of a specified lattice (intended to be used for peak yield lattice). 
    - Function Inputs:
        display: Whether or not we want to display graphs to user terminal (Boolean).
        peak_yield_lattice: Lattice we wish to plot (numpy matrix).
        D: Current design parameter value (int). 
        L: Current perimiter size value (int). 
    - Function Output: 
        Nothing, but if display=True an animation will appear in the external animation and save
        user machine.  
    """
    Q3a_fig = plt.figure()
    axes = Q3a_fig.add_subplot(111)
    axes.set_title("Tree Space at Optimal Yield, Design Parameter $D$ = {D}".format(D=D))
    axes.set_xlabel("$L$ = {L}".format(L=L))
    green_patch = mpatches.Patch(color='green', label="tree")
    brown_patch = mpatches.Patch(color='brown', label="dirt")
    plt.legend(handles=[green_patch, brown_patch])

    color_map = colors.ListedColormap(['brown', 'green'])
    plt.imshow(peak_yield_lattice, cmap=color_map)

    plt.savefig("peak_yield_L{L}_D{D}.png".format(L=L, D=D))

    if display == True:
        plt.show()

def yieldBYdensity_grapher(display, D, yields2trees_added_LIST):
    """    
    - Function Description:
        Function graphs yield curves (Y = density - cost, <Y> = density - <cost>) for multiple L values for 
        a given D value. 
    - Function Inputs:
        display: Whether or not we want to display graphs to user terminal (Boolean).
        D: Current design parameter value (int). 
        yields2trees_added_LIST: A list of dictionarys of yields to number of trees added. 
    - Function Output: 
        Nothing, but if display=True an animation will appear in the external animation and save
        to user machine.             
    """
    # Plotting the yield curves for each value of D, and identifying (approximately) the - 
    # - peak yield and density for which peak yield occurs on each,
    Q3pb = plt.figure()
    axes = Q3pb.add_subplot(111)
    max_yield2density = {} # To fill and caluclate max yield over all L's to user terminal. 

    for average_yield2trees_added in yields2trees_added_LIST:        
        trees_added = np.array(list(average_yield2trees_added.values()))
        densities = trees_added/len(average_yield2trees_added.values())
        x = list(densities) # Densities.
        x.insert(0, 0)

        yields = list(average_yield2trees_added.keys())
        y = yields
        y.insert(0, 0)

        L = math.sqrt(len(average_yield2trees_added.keys()))

        axes.set_title("Yield Curve by Density $p$ for $D = $ {D}".format(D=D))
        axes.set_xlabel("Density $p$") 
        axes.set_ylabel("Average Yield $<Y> = p - <c>$") 

        green_patch = mpatches.Patch(color='green', label="$L=2$")
        yellow_patch = mpatches.Patch(color='yellow', label="$L=4$")
        orange_patch = mpatches.Patch(color='orange', label="$L=8$")
        red_patch = mpatches.Patch(color='red', label="$L=16$")
        purple_patch = mpatches.Patch(color='purple', label="$L=32$")
        black_patch = mpatches.Patch(color='black', label="$L=64$")

        plt.legend(handles=[green_patch, yellow_patch, orange_patch, red_patch, black_patch, purple_patch])
        cmap = {2:'green', 4:'yellow', 8:'orange', 16:'red', 32:'purple', 64:'black'}
        axes.plot(x, y, linestyle='-',  label='L = {L}'.format(L=L), color=cmap[L])

        plt.savefig("yield_by_density_D{D}.png".format(D=D))

        max_yield = max(yields)
        density_at_max = average_yield2trees_added[max_yield]/(L**2)

        max_yield2density[max_yield] = density_at_max
    
    peak_yield = max(max_yield2density.keys())
    print("Peak yield when D = {D} :".format(D=D), peak_yield)
    print("Density at this peak yield :", max_yield2density[peak_yield])

    if display == True:
        plt.show()

def component_frequency_grapher(display, peak_yield_lattice, D, L, logg):
    """   
    - Function Description:
        Function graphs yield curves (Y = density - cost, <Y> = density - <cost>) for multiple L values for 
        a given D value. 
    - Function Inputs:
        display: Whether or not we want to display graphs to user terminal (Boolean).
        peak_yield_lattice: Lattice we wish to plot (numpy matrix).
        D: Current design parameter value (int). 
        L: Current perimiter size value (int). 
        logg: Whether or not we want to plot logged graph also (Boolean). 
    - Function Output: 
        Nothing, but if display=True an animation will appear in the external animation and save
        to user machine.             
    """
    component_size2count = {}

    # Plotting distributions of tree component size `S` at peak yield. 
    # Calculating the connected components. 
    connected_components = measure.label(peak_yield_lattice, connectivity=1)

    # Now that we have anlayzed connected component data, we analyze potential fire sizes. 
    unique, counts = np.unique(connected_components, return_counts=True)
    components2size = dict(zip(unique, counts))

    # We don't want to consider the empty squares here. 
    del components2size[0]

    for component_id in components2size.keys():
        try:
            component_size2count[components2size[component_id]] += 1
        except KeyError:
            component_size2count[components2size[component_id]] = 1
    
    freqBYcomponent_size = plt.figure()
    axes = freqBYcomponent_size.add_subplot(111)

    # for L in L2runningAvg_pvals.keys():
    x = list(component_size2count.keys()) # The x axis is the sizes S of the components.
    y = list(component_size2count.values()) # The y axis is the number of components of this size in the peak lattice. 
    
    axes.scatter(x, y, color="purple", alpha=0.33) 
        
    # Labelling and making the plot output look nicer. 
    axes.set_title("Component Size $S$ Frequencies, D = {D}, L = {L}".format(D=D, L=L))
    axes.set_xlabel("Component Size $S$") # , ($log_{10}$)
    axes.set_ylabel("Count of Components of Size $S$") # , ($log_{10}$)
    
    plt.savefig("component_frequency{D}.png".format(D=D))

    if display == True:
        plt.show()

    if logg == True:
        freqBYcomponent_size_LOG = plt.figure()
        axes = freqBYcomponent_size_LOG.add_subplot(111)
        axes.scatter(np.log10(x), np.log10(y), color="purple", alpha=0.33)
        axes.set_title("Component Size $S$ Frequencies, D = {D}, L = {L}".format(D=D, L=L))
        axes.set_xlabel("Component Size $S$, ($log_{10}$)")
        axes.set_ylabel("Count of Components of Size $S$, ($log_{10}$)")
        plt.savefig("component_frequency_log{D}.png".format(D=D))
        if display == True:
            plt.show()

# def component_sizeBYdensity_grapher(display, D, density2lattice):
#     """
#     fdsa
#     """
#     # Plotting the yield curves for each value of D, and identifying (approximately) the - 
#     # - peak yield and density for which peak yield occurs on each,
#     Q3pd = plt.figure()
#     axes = Q3pd.add_subplot(111)  

#     x = list(density2lattice.keys()) # Densities.
#     y = list(density2lattice.values())

    # axes.set_title("")
    # axes.set_xlabel("Event Size $c$") 
    # axes.set_ylabel("Cumulative Probability $F(c)$") 

    # green_patch = mpatches.Patch(color='green', label="$L=2$")
    # yellow_patch = mpatches.Patch(color='yellow', label="$L=4$")
    # orange_patch = mpatches.Patch(color='orange', label="$L=8$")
    # red_patch = mpatches.Patch(color='red', label="$L=16$")
    # purple_patch = mpatches.Patch(color='purple', label="$L=32$")
    # black_patch = mpatches.Patch(color='black', label="$L=64$")

    # plt.legend(handles=[green_patch, yellow_patch, orange_patch, red_patch, black_patch, purple_patch])
    # axes.plot(x, y, linestyle='-')

    # plt.savefig("component_sizeBYdensity_D{D}.png".format(D=D))

    # if display == True:
    #     plt.show()

# =============================================================================
# ========================      MAIN LOGIC CALL        ========================
# =============================================================================
main(L_values, D_values)